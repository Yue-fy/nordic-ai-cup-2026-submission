#!/usr/bin/env python3
"""Nested conversation-group audit for Medical evidence selection.

The locator predictions supplied to this script must be out-of-fold: every
prediction must come from a locator that did not train on that conversation.
An outer group split evaluates the selector on unseen conversations.  Within
each outer training split, another group split chooses the switch threshold.
This avoids fitting and scoring the evidence gate on the same conversations.

This is an offline development audit.  It does not call competition APIs and
does not contain SHA-specific response overrides.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STOP = set(
    "a an the is are was were be been being did do does has have had will would could "
    "should can of to in at on for with and or as by it its this that these those patient "
    "doctor clinician he she they their his her you your i my we our".split()
)
NEGATIONS = {"no", "not", "never", "without", "none", "denies", "denied", "free"}
CANDIDATE_SCHEMA = ("ft_raw", "ft_clause", "ft_sentence")
CANDIDATES = CANDIDATE_SCHEMA
THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1)
CONSERVATIVE_THRESHOLDS = (*THRESHOLDS, 0.125, 0.15, 0.2)


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP


def tiou(left: tuple[float, float] | None, right: tuple[float, float] | None) -> float:
    if left is None or right is None:
        return 0.0
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union if union > 0 else 0.0


def span_text(words: list[dict], span: tuple[float, float] | None) -> str:
    if span is None:
        return ""
    return " ".join(
        str(word["text"])
        for word in words
        if float(word["end"]) > span[0] and float(word["start"]) < span[1]
    )


def similarity(question: set[str], passage: set[str]) -> list[float]:
    shared = question & passage
    q_numbers = {item for item in question if item.isdigit()}
    p_numbers = {item for item in passage if item.isdigit()}
    return [
        len(shared) / max(1, len(question)),
        len(shared) / max(1, len(passage)),
        float(bool(q_numbers - p_numbers)),
        float(bool(p_numbers - q_numbers)),
        float(bool(question & NEGATIONS)),
        float(bool(passage & NEGATIONS)),
    ]


def feature_vector(prediction: dict, candidate_name: str, words: list[dict]) -> list[float]:
    baseline = tuple(prediction["spans"]["baseline"])
    candidate = tuple(prediction["spans"][candidate_name])
    raw = tuple(prediction["spans"]["ft_raw"])
    question = tokens(prediction["question"])
    baseline_tokens = tokens(span_text(words, baseline))
    candidate_tokens = tokens(span_text(words, candidate))
    total = max(float(words[-1]["end"]), 1.0)
    qa = prediction["qa"]
    union = baseline_tokens | candidate_tokens
    return [
        *[float(candidate_name == name) for name in CANDIDATE_SCHEMA],
        float(prediction["answer_true"]),
        float(qa["logit"]) / 20.0,
        float(qa["margin"]),
        tiou(baseline, candidate),
        tiou(raw, candidate),
        (baseline[1] - baseline[0]) / 10.0,
        (raw[1] - raw[0]) / 10.0,
        (candidate[1] - candidate[0]) / 10.0,
        abs(sum(baseline) / 2.0 - sum(candidate) / 2.0) / total,
        (sum(candidate) / 2.0) / total,
        len(question) / 10.0,
        len(baseline_tokens) / 20.0,
        len(candidate_tokens) / 20.0,
        *similarity(question, baseline_tokens),
        *similarity(question, candidate_tokens),
        len(baseline_tokens & candidate_tokens) / max(1, len(union)),
    ]


def load_documents(cache: Path) -> dict[str, dict]:
    documents = {}
    for path in cache.glob("*.json"):
        document = json.loads(path.read_text())
        documents[document["audio_filename"]] = document
    return documents


def expanded(
    rows: list[dict], documents: dict[str, dict]
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    features: list[list[float]] = []
    targets: list[float] = []
    question_ids: list[str] = []
    candidate_names: list[str] = []
    groups: list[str] = []
    for row in rows:
        words = documents[f"conversation_{row['transcript_id']}.mp3"]["words"]
        baseline_iou = float(row["ious"]["baseline"])
        for name in CANDIDATES:
            if row["spans"].get(name) is None:
                continue
            features.append(feature_vector(row, name, words))
            targets.append(float(row["ious"][name]) - baseline_iou)
            question_ids.append(row["question_id"])
            candidate_names.append(name)
            groups.append(row["transcript_id"])
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        question_ids,
        candidate_names,
        groups,
    )


def candidate_predictions(
    model,
    rows: list[dict],
    documents: dict[str, dict],
) -> dict[str, list[tuple[str, float]]]:
    x, _target, question_ids, names, _groups = expanded(rows, documents)
    values = model.predict(x)
    output: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for question_id, name, value in zip(question_ids, names, values, strict=True):
        output[question_id].append((name, float(value)))
    return output


def selected_ious(
    rows: list[dict], predictions: dict[str, list[tuple[str, float]]], threshold: float
) -> dict[str, dict]:
    output = {}
    for row in rows:
        choices = predictions.get(row["question_id"], [])
        best_name, expected_gain = max(
            choices, key=lambda item: item[1], default=("baseline", -math.inf)
        )
        name = best_name
        if expected_gain <= threshold:
            name = "baseline"
        output[row["question_id"]] = {
            "transcript_id": row["transcript_id"],
            "candidate": name,
            "best_candidate": best_name,
            "expected_gain": expected_gain,
            "baseline_iou": float(row["ious"]["baseline"]),
            "best_candidate_iou": float(row["ious"][best_name]),
            "selected_iou": float(row["ious"][name]),
        }
    return output


def mean_iou(selection: dict[str, dict]) -> float:
    return float(np.mean([row["selected_iou"] for row in selection.values()]))


def tune_threshold(estimator, rows: list[dict], documents: dict[str, dict]) -> tuple[float, dict]:
    groups = np.asarray([row["transcript_id"] for row in rows])
    folds = min(4, len(set(groups)))
    splitter = GroupKFold(n_splits=folds)
    held_predictions: dict[str, list[tuple[str, float]]] = {}
    for train_indices, test_indices in splitter.split(rows, groups=groups):
        train_rows = [rows[index] for index in train_indices]
        test_rows = [rows[index] for index in test_indices]
        x, y, _ids, _names, _expanded_groups = expanded(train_rows, documents)
        fitted = clone(estimator).fit(x, y)
        held_predictions.update(candidate_predictions(fitted, test_rows, documents))
    scores = {
        str(threshold): mean_iou(selected_ious(rows, held_predictions, threshold))
        for threshold in THRESHOLDS
    }
    threshold = max(THRESHOLDS, key=lambda value: (scores[str(value)], value))
    return threshold, scores


def tune_threshold_one_se(
    estimator, rows: list[dict], documents: dict[str, dict]
) -> tuple[float, dict]:
    """Choose the safest threshold statistically tied with the inner-CV best.

    The usual argmax is unstable on this small corpus: a few large gains can
    select a permissive threshold that also admits catastrophic replacements.
    We first produce selector predictions that are out-of-fold by conversation.
    We then apply the one-standard-error rule, using conversations rather than
    questions as independent units, and choose the largest eligible threshold.
    No outer-fold labels enter this decision.
    """
    groups = np.asarray([row["transcript_id"] for row in rows])
    folds = min(4, len(set(groups)))
    splitter = GroupKFold(n_splits=folds)
    held_predictions: dict[str, list[tuple[str, float]]] = {}
    for train_indices, test_indices in splitter.split(rows, groups=groups):
        train_rows = [rows[index] for index in train_indices]
        test_rows = [rows[index] for index in test_indices]
        x, y, _ids, _names, _expanded_groups = expanded(train_rows, documents)
        fitted = clone(estimator).fit(x, y)
        held_predictions.update(candidate_predictions(fitted, test_rows, documents))

    diagnostics = {}
    for threshold in CONSERVATIVE_THRESHOLDS:
        selection = selected_ious(rows, held_predictions, threshold)
        by_conversation: dict[str, list[float]] = defaultdict(list)
        for item in selection.values():
            by_conversation[item["transcript_id"]].append(
                item["selected_iou"] - item["baseline_iou"]
            )
        conversation_deltas = np.asarray(
            [np.mean(values) for values in by_conversation.values()], dtype=np.float64
        )
        diagnostics[str(threshold)] = {
            "mean_tiou": mean_iou(selection),
            "mean_delta": float(
                np.mean(
                    [
                        item["selected_iou"] - item["baseline_iou"]
                        for item in selection.values()
                    ]
                )
            ),
            "conversation_standard_error": float(
                np.std(conversation_deltas, ddof=1) / math.sqrt(len(conversation_deltas))
            ),
            "switches": sum(
                item["candidate"] != "baseline" for item in selection.values()
            ),
        }

    best_key = max(
        diagnostics,
        key=lambda key: (diagnostics[key]["mean_tiou"], float(key)),
    )
    best = diagnostics[best_key]
    cutoff = best["mean_tiou"] - best["conversation_standard_error"]
    eligible = [
        float(key)
        for key, value in diagnostics.items()
        if value["mean_tiou"] >= cutoff
    ]
    threshold = max(eligible)
    return threshold, {
        "rule": "largest threshold within one conversation-level standard error of inner-CV best",
        "best_threshold": float(best_key),
        "eligibility_cutoff": cutoff,
        "thresholds": diagnostics,
    }


def bootstrap_delta(
    selection: dict[str, dict], iterations: int = 20000, seed: int = 260917
) -> dict:
    by_conversation: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in selection.values():
        by_conversation[row["transcript_id"]].append(
            (row["selected_iou"], row["baseline_iou"])
        )
    names = sorted(by_conversation)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(iterations):
        sample = rng.choice(names, size=len(names), replace=True)
        pairs = [pair for name in sample for pair in by_conversation[name]]
        deltas.append(float(np.mean([left - right for left, right in pairs])))
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "mean_tiou_delta": float(np.mean([a - b for rows in by_conversation.values() for a, b in rows])),
        "conversation_bootstrap_95": [float(low), float(high)],
        "probability_positive": float(np.mean(np.asarray(deltas) > 0.0)),
        "iterations": iterations,
    }


def audit_model(
    name: str,
    estimator,
    rows: list[dict],
    documents: dict[str, dict],
    folds: int,
) -> dict:
    groups = np.asarray([row["transcript_id"] for row in rows])
    splitter = GroupKFold(n_splits=folds)
    all_selected: dict[str, dict] = {}
    fold_summaries = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(rows, groups=groups)):
        train_rows = [rows[index] for index in train_indices]
        test_rows = [rows[index] for index in test_indices]
        threshold, threshold_scores = tune_threshold(estimator, train_rows, documents)
        x, y, _ids, _names, _expanded_groups = expanded(train_rows, documents)
        fitted = clone(estimator).fit(x, y)
        predictions = candidate_predictions(fitted, test_rows, documents)
        chosen = selected_ious(test_rows, predictions, threshold)
        all_selected.update(chosen)
        fold_summaries.append(
            {
                "fold": fold,
                "test_conversations": sorted({row["transcript_id"] for row in test_rows}),
                "threshold": threshold,
                "inner_threshold_scores": threshold_scores,
                "baseline_mean_tiou": float(
                    np.mean([row["ious"]["baseline"] for row in test_rows])
                ),
                "selected_mean_tiou": mean_iou(chosen),
                "candidate_counts": dict(Counter(item["candidate"] for item in chosen.values())),
            }
        )
    return {
        "name": name,
        "baseline_mean_tiou": float(np.mean([row["ious"]["baseline"] for row in rows])),
        "selected_mean_tiou": mean_iou(all_selected),
        "candidate_counts": dict(Counter(item["candidate"] for item in all_selected.values())),
        "bootstrap": bootstrap_delta(all_selected),
        "folds": fold_summaries,
        "selections": all_selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    rows = json.loads(args.oof.read_text())
    records = json.loads(args.records.read_text())
    documents = load_documents(args.cache)
    accuracy = sum(bool(row["correct"]) for row in records) / len(records)
    estimators = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=8,
            max_features=0.8,
            random_state=260917,
            n_jobs=-1,
        ),
    }
    results = [
        audit_model(name, estimator, rows, documents, args.folds)
        for name, estimator in estimators.items()
    ]
    baseline_tiou = float(np.mean([row["ious"]["baseline"] for row in rows]))
    oracle_tiou = float(np.mean([max(row["ious"].values()) for row in rows]))
    for result in results:
        result["score"] = 0.4 * accuracy + 0.6 * result["selected_mean_tiou"]
    payload = {
        "protocol": "nested conversation-group CV for selector; locator inputs are conversation-OOF",
        "questions": len(records),
        "positive_questions": len(rows),
        "conversations": len({row["transcript_id"] for row in rows}),
        "frozen_answer_accuracy": accuracy,
        "baseline_mean_tiou": baseline_tiou,
        "baseline_score": 0.4 * accuracy + 0.6 * baseline_tiou,
        "candidate_oracle_mean_tiou": oracle_tiou,
        "candidate_oracle_score": 0.4 * accuracy + 0.6 * oracle_tiou,
        "models": results,
        "limitation": (
            "The locator predictions are OOF for each evaluated conversation, but selector folds reuse "
            "a precomputed locator OOF run. A final end-to-end audit should nest locator training inside "
            "the outer selector fold."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "baseline_score": payload["baseline_score"],
                "candidate_oracle_score": payload["candidate_oracle_score"],
                "models": [
                    {
                        "name": result["name"],
                        "score": result["score"],
                        "bootstrap": result["bootstrap"],
                        "candidate_counts": result["candidate_counts"],
                    }
                    for result in results
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
