#!/usr/bin/env python3
"""Fit and export the Medical selector from fully OOF locator predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nested_selector_audit as selector_audit
from nested_selector_audit import expanded, load_documents, tune_threshold_one_se


def answer_token_limit(
    rows: list[dict], documents: dict[str, dict], quantile: float
) -> int:
    lengths = []
    for row in rows:
        words = documents[f"conversation_{row['transcript_id']}.mp3"]["words"]
        start, end = row["gold"]
        lengths.append(
            sum(
                float(word["end"]) > start and float(word["start"]) < end
                for word in words
            )
        )
    return max(1, int(np.ceil(np.quantile(lengths, quantile))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nested-result", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-answer-quantile", type=float, default=0.99)
    args = parser.parse_args()

    result = json.loads(args.nested_result.read_text())
    if result["acceptance"]["generalization_gate"] != "PASS":
        raise RuntimeError("refusing to export a selector that failed the generalization gate")
    rows = [row for fold in result["folds"] for row in fold["outer_predictions"]]
    if len({row["question_id"] for row in rows}) != len(rows):
        raise RuntimeError("outer predictions are not unique by question")
    documents = load_documents(args.cache)
    candidate_names = (
        ("ft_raw",)
        if result.get("selector_candidates") == "raw"
        else ("ft_raw", "ft_clause", "ft_sentence")
    )
    selector_audit.CANDIDATES = candidate_names
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    threshold, threshold_diagnostics = tune_threshold_one_se(
        estimator, rows, documents
    )
    x, y, _ids, _names, _groups = expanded(rows, documents)
    estimator.fit(x, y)
    scaler: StandardScaler = estimator.named_steps["standardscaler"]
    ridge: Ridge = estimator.named_steps["ridge"]
    payload = {
        "format": "medical-conservative-selector-v1",
        "source": str(args.nested_result),
        "training_protocol": result["protocol"],
        "training_questions": len(rows),
        "training_conversations": len({row["transcript_id"] for row in rows}),
        "ridge_alpha": 10.0,
        "feature_count": int(x.shape[1]),
        "candidate_names": list(candidate_names),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "ridge_coef": ridge.coef_.tolist(),
        "ridge_intercept": float(ridge.intercept_),
        "switch_threshold": threshold,
        "threshold_diagnostics": threshold_diagnostics,
        "max_answer_quantile": args.max_answer_quantile,
        "max_answer_tokens": answer_token_limit(
            rows, documents, args.max_answer_quantile
        ),
        "nested_validation": {
            "baseline_score": result["baseline_score"],
            "selected_score": result["selected_score"],
            "fold_deltas": result["fold_deltas"],
            "bootstrap": result["bootstrap"],
            "acceptance": result["acceptance"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "switch_threshold": threshold,
                "max_answer_tokens": payload["max_answer_tokens"],
                "feature_count": payload["feature_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
