#!/usr/bin/env python3
"""End-to-end nested group CV for the Medical evidence pipeline.

For every outer held-out conversation group:

1. train a locator only on the outer training conversations;
2. create selector-training predictions with inner locator folds;
3. choose the selector threshold with grouped cross-validation;
4. evaluate locator plus selector on the untouched outer conversations.

This script never calls the competition API.  It measures generalization to
new conversations and writes a checkpoint after every outer fold.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from finetune_qa_span_cv import Features, evaluate, training_features
from model_regularization import freeze_lower_encoder_layers
from structured_tiou_locator_cv import structured_loss, structured_training_features
import nested_selector_audit as selector_audit
from nested_selector_audit import (
    THRESHOLDS,
    bootstrap_delta,
    candidate_predictions,
    expanded,
    load_documents,
    mean_iou,
    selected_ious,
    tune_threshold,
    tune_threshold_one_se,
)


def train_locator(
    rows: list[dict],
    documents: dict[str, dict],
    alternate_documents: dict[str, dict],
    tokenizer,
    model_name: str,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    freeze_lower_layers: int,
    loss_mode: str = "exact",
    structured_weight: float = 0.5,
    structured_temperature: float = 0.05,
    structured_max_answer_tokens: int = 81,
):
    features = []
    for row in rows:
        filename = f"conversation_{row['transcript_id']}.mp3"
        feature_builder = (
            structured_training_features if loss_mode == "structured_tiou" else training_features
        )
        features.extend(feature_builder(row, documents[filename], tokenizer))
        alternate = alternate_documents.get(filename)
        if alternate is not None:
            features.extend(feature_builder(row, alternate, tokenizer))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        Features(features),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    torch.manual_seed(seed)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    parameter_summary = freeze_lower_encoder_layers(model, freeze_lower_layers)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )
    model.train()
    losses = []
    amp_dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    for _epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=device == "cuda"
            ):
                if loss_mode == "structured_tiou":
                    model_batch = {
                        key: batch[key].to(device)
                        for key in ("input_ids", "attention_mask")
                    }
                    targets = {
                        key: value.to(device)
                        for key, value in batch.items()
                        if key not in model_batch
                    }
                    result = model(**model_batch)
                    loss, _hard, _structured = structured_loss(
                        result.start_logits.float(),
                        result.end_logits.float(),
                        targets,
                        structured_weight=structured_weight,
                        temperature=structured_temperature,
                        max_answer_tokens=structured_max_answer_tokens,
                    )
                else:
                    result = model(**{key: value.to(device) for key, value in batch.items()})
                    loss = result.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    model.eval()
    return model, len(features), losses[-1], parameter_summary


def auxiliary_training_rows(real_rows: list[dict], args) -> list[dict]:
    """Select a balanced, fold-local subset of auxiliary locator examples."""
    if not args.auxiliary_rows or args.auxiliary_ratio <= 0:
        return []
    conversation_ids = {row["transcript_id"] for row in real_rows}
    grouped: dict[str, list[dict]] = {}
    for row in args.auxiliary_rows:
        if row["transcript_id"] in conversation_ids:
            grouped.setdefault(row["transcript_id"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["question_id"])
    target = min(
        sum(len(rows) for rows in grouped.values()),
        int(math.ceil(len(real_rows) * args.auxiliary_ratio)),
    )
    selected = []
    depth = 0
    names = sorted(grouped)
    while len(selected) < target:
        added = False
        for name in names:
            if depth < len(grouped[name]):
                selected.append(grouped[name][depth])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        depth += 1
    return selected


def independent_auxiliary_training_rows(real_rows: list[dict], args) -> list[dict]:
    """Select independent synthetic conversations without using held-out data.

    These examples have their own transcript IDs and documents, so they do not
    belong to any real outer or inner fold.  Selection is balanced by synthetic
    conversation and capped relative to the number of real positives.
    """
    if not args.independent_auxiliary_rows or args.independent_auxiliary_ratio <= 0:
        return []
    grouped: dict[str, list[dict]] = {}
    for row in args.independent_auxiliary_rows:
        grouped.setdefault(row["transcript_id"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["question_id"])
    target = min(
        sum(len(rows) for rows in grouped.values()),
        int(math.ceil(len(real_rows) * args.independent_auxiliary_ratio)),
    )
    selected = []
    depth = 0
    names = sorted(grouped)
    while len(selected) < target:
        added = False
        for name in names:
            if depth < len(grouped[name]):
                selected.append(grouped[name][depth])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        depth += 1
    return selected


def make_selector():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def answer_token_limit(
    rows: list[dict], documents: dict[str, dict], quantile: float | None
) -> int:
    """Estimate a span-length prior from training conversations only."""
    if quantile is None:
        return 81
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


def inner_locator_oof(
    outer_train: list[dict],
    documents: dict[str, dict],
    tokenizer,
    args,
    device: str,
    outer_fold: int,
) -> tuple[list[dict], list[dict]]:
    groups = np.asarray([row["transcript_id"] for row in outer_train])
    splitter = GroupKFold(n_splits=args.inner_folds)
    predictions = []
    summaries = []
    for inner_fold, (train_indices, test_indices) in enumerate(
        splitter.split(outer_train, groups=groups)
    ):
        started = time.monotonic()
        train_rows = [outer_train[index] for index in train_indices]
        test_rows = [outer_train[index] for index in test_indices]
        seed = args.seed + outer_fold * 100 + inner_fold
        auxiliary_rows = auxiliary_training_rows(train_rows, args)
        independent_rows = independent_auxiliary_training_rows(train_rows, args)
        model, feature_count, last_loss, parameter_summary = train_locator(
            [*train_rows, *auxiliary_rows, *independent_rows],
            documents,
            args.alternate_documents,
            tokenizer,
            args.model,
            device,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            seed,
            args.freeze_lower_layers,
            args.locator_loss,
            args.structured_weight,
            args.structured_temperature,
            args.structured_max_answer_tokens,
        )
        max_answer_tokens = answer_token_limit(
            train_rows, documents, args.max_answer_quantile
        )
        fold_predictions = evaluate(
            test_rows,
            documents,
            tokenizer,
            model,
            device,
            max_answer_tokens=max_answer_tokens,
        )
        predictions.extend(fold_predictions)
        summaries.append(
            {
                "inner_fold": inner_fold,
                "train_conversations": len({row["transcript_id"] for row in train_rows}),
                "test_conversations": sorted({row["transcript_id"] for row in test_rows}),
                "training_features": feature_count,
                "auxiliary_rows": len(auxiliary_rows),
                "independent_auxiliary_rows": len(independent_rows),
                "last_loss": last_loss,
                "parameter_summary": parameter_summary,
                "max_answer_tokens": max_answer_tokens,
                "seconds": time.monotonic() - started,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    predictions.sort(key=lambda row: row["question_id"])
    return predictions, summaries


def partial_payload(args, accuracy: float, folds: list[dict]) -> dict:
    selections = {
        question_id: selection
        for fold in folds
        for question_id, selection in fold["selections"].items()
    }
    baseline_tiou = float(
        np.mean([row["baseline_iou"] for row in selections.values()])
    ) if selections else None
    selected_tiou = mean_iou(selections) if selections else None
    bootstrap = bootstrap_delta(selections) if len(folds) == args.outer_folds else None
    fold_deltas = [
        float(fold["selected_mean_tiou"] - fold["baseline_mean_tiou"])
        for fold in folds
    ]
    sensitivity = []
    if len(folds) == args.outer_folds:
        for threshold in (*THRESHOLDS, 0.125, 0.15, 0.2):
            values = []
            baselines = []
            per_fold = []
            switches = 0
            for fold in folds:
                fold_values = []
                fold_baselines = []
                for row in fold["selections"].values():
                    active = row["expected_gain"] > threshold
                    fold_values.append(
                        row.get("best_candidate_iou", row["selected_iou"])
                        if active
                        else row["baseline_iou"]
                    )
                    fold_baselines.append(row["baseline_iou"])
                    switches += int(active)
                values.extend(fold_values)
                baselines.extend(fold_baselines)
                per_fold.append(float(np.mean(fold_values) - np.mean(fold_baselines)))
            sensitivity.append(
                {
                    "fixed_threshold": threshold,
                    "score": 0.4 * accuracy + 0.6 * float(np.mean(values)),
                    "mean_tiou_delta": float(
                        np.mean(np.asarray(values) - np.asarray(baselines))
                    ),
                    "fold_deltas": per_fold,
                    "switches": switches,
                }
            )
    passes = bool(
        bootstrap
        and bootstrap["conversation_bootstrap_95"][0] > 0.0
        and min(fold_deltas) >= -0.02
    )
    return {
        "protocol": "fully nested conversation-group CV for locator and selector",
        "model": args.model,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "freeze_lower_layers": args.freeze_lower_layers,
        "locator_loss": args.locator_loss,
        "structured_weight": args.structured_weight,
        "structured_temperature": args.structured_temperature,
        "structured_max_answer_tokens": args.structured_max_answer_tokens,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "selector_mode": args.selector_mode,
        "selector_candidates": args.selector_candidates,
        "max_answer_quantile": args.max_answer_quantile,
        "auxiliary_records": str(args.auxiliary_records) if args.auxiliary_records else None,
        "auxiliary_ratio": args.auxiliary_ratio,
        "alternate_training_cache": (
            str(args.alternate_training_cache) if args.alternate_training_cache else None
        ),
        "independent_auxiliary_records": (
            str(args.independent_auxiliary_records)
            if args.independent_auxiliary_records
            else None
        ),
        "independent_auxiliary_cache": (
            str(args.independent_auxiliary_cache)
            if args.independent_auxiliary_cache
            else None
        ),
        "independent_auxiliary_ratio": args.independent_auxiliary_ratio,
        "completed_outer_folds": len(folds),
        "frozen_answer_accuracy": accuracy,
        "baseline_mean_tiou": baseline_tiou,
        "selected_mean_tiou": selected_tiou,
        "baseline_score": 0.4 * accuracy + 0.6 * baseline_tiou if selections else None,
        "selected_score": 0.4 * accuracy + 0.6 * selected_tiou if selections else None,
        "bootstrap": bootstrap,
        "fold_deltas": fold_deltas,
        "worst_fold_delta": min(fold_deltas) if fold_deltas else None,
        "threshold_sensitivity_posthoc": sensitivity,
        "acceptance": {
            "bootstrap_lower_bound_positive": bool(
                bootstrap and bootstrap["conversation_bootstrap_95"][0] > 0.0
            ),
            "worst_fold_delta_at_least_minus_0_02": bool(
                fold_deltas and min(fold_deltas) >= -0.02
            ),
            "generalization_gate": "PASS" if passes else "FAIL",
            "decision": "eligible" if passes else "do_not_deploy",
        },
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="deepset/roberta-base-squad2")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--locator-loss",
        choices=("exact", "structured_tiou"),
        default="exact",
        help="Evidence-locator training objective; exact preserves the established pipeline.",
    )
    parser.add_argument("--structured-weight", type=float, default=0.5)
    parser.add_argument("--structured-temperature", type=float, default=0.05)
    parser.add_argument("--structured-max-answer-tokens", type=int, default=81)
    parser.add_argument(
        "--freeze-lower-layers",
        type=int,
        default=0,
        help="Freeze embeddings and this many lower encoder layers during locator fine-tuning.",
    )
    parser.add_argument("--seed", type=int, default=260917)
    parser.add_argument(
        "--selector-mode",
        choices=("argmax", "one_se"),
        default="argmax",
        help="Inner-CV threshold rule; one_se favors conservative replacement.",
    )
    parser.add_argument(
        "--selector-candidates",
        choices=("all", "raw"),
        default="all",
        help="Candidate families considered by the conservative selector.",
    )
    parser.add_argument(
        "--max-answer-quantile",
        type=float,
        default=None,
        help=(
            "Learn the maximum extractive span length from this quantile of "
            "training-fold gold word counts; omitted preserves the 81-token cap."
        ),
    )
    parser.add_argument(
        "--auxiliary-records",
        type=Path,
        help="Transcript-grounded auxiliary locator rows; filtered by training-fold conversation.",
    )
    parser.add_argument(
        "--auxiliary-ratio",
        type=float,
        default=0.0,
        help="Maximum auxiliary rows per real positive training row.",
    )
    parser.add_argument(
        "--alternate-training-cache",
        type=Path,
        help="Optional second ASR cache used only to augment locator training features.",
    )
    parser.add_argument(
        "--independent-auxiliary-records",
        type=Path,
        help="Synthetic locator rows from conversations independent of all real folds.",
    )
    parser.add_argument(
        "--independent-auxiliary-cache",
        type=Path,
        help="Timestamped transcripts for independent synthetic locator rows.",
    )
    parser.add_argument(
        "--independent-auxiliary-ratio",
        type=float,
        default=0.0,
        help="Maximum independent synthetic rows per real positive training row.",
    )
    args = parser.parse_args()

    if args.auxiliary_ratio < 0 or args.independent_auxiliary_ratio < 0:
        raise ValueError("auxiliary ratios must be nonnegative")
    args.auxiliary_rows = (
        json.loads(args.auxiliary_records.read_text()) if args.auxiliary_records else []
    )
    args.independent_auxiliary_rows = (
        json.loads(args.independent_auxiliary_records.read_text())
        if args.independent_auxiliary_records
        else []
    )
    if bool(args.independent_auxiliary_records) != bool(args.independent_auxiliary_cache):
        raise ValueError(
            "independent-auxiliary-records and independent-auxiliary-cache must be provided together"
        )

    if args.selector_candidates == "raw":
        selector_audit.CANDIDATES = ("ft_raw",)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    all_records = json.loads(args.records.read_text())
    rows = [row for row in all_records if row["label"]]
    accuracy = sum(bool(row["correct"]) for row in all_records) / len(all_records)
    documents = load_documents(args.cache)
    if args.independent_auxiliary_cache:
        independent_documents = load_documents(args.independent_auxiliary_cache)
        overlap = set(documents) & set(independent_documents)
        if overlap:
            raise RuntimeError(f"synthetic document IDs overlap real documents: {sorted(overlap)[:3]}")
        documents.update(independent_documents)
    args.alternate_documents = (
        load_documents(args.alternate_training_cache)
        if args.alternate_training_cache
        else {}
    )
    missing = sorted(
        f"conversation_{row['transcript_id']}.mp3"
        for row in rows
        if f"conversation_{row['transcript_id']}.mp3" not in documents
    )
    if missing:
        raise RuntimeError(f"missing ASR documents: {missing[:3]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    groups = np.asarray([row["transcript_id"] for row in rows])
    outer_splitter = GroupKFold(n_splits=args.outer_folds)
    folds = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for outer_fold, (train_indices, test_indices) in enumerate(
        outer_splitter.split(rows, groups=groups)
    ):
        started = time.monotonic()
        outer_train = [rows[index] for index in train_indices]
        outer_test = [rows[index] for index in test_indices]
        inner_predictions, inner_summaries = inner_locator_oof(
            outer_train, documents, tokenizer, args, device, outer_fold
        )
        selector = make_selector()
        threshold_tuner = (
            tune_threshold_one_se if args.selector_mode == "one_se" else tune_threshold
        )
        threshold, threshold_scores = threshold_tuner(selector, inner_predictions, documents)
        x, y, _ids, _names, _groups = expanded(inner_predictions, documents)
        selector.fit(x, y)

        outer_auxiliary = auxiliary_training_rows(outer_train, args)
        outer_independent = independent_auxiliary_training_rows(outer_train, args)
        outer_model, feature_count, last_loss, parameter_summary = train_locator(
            [*outer_train, *outer_auxiliary, *outer_independent],
            documents,
            args.alternate_documents,
            tokenizer,
            args.model,
            device,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.seed + outer_fold * 100 + 99,
            args.freeze_lower_layers,
            args.locator_loss,
            args.structured_weight,
            args.structured_temperature,
            args.structured_max_answer_tokens,
        )
        outer_max_answer_tokens = answer_token_limit(
            outer_train, documents, args.max_answer_quantile
        )
        outer_predictions = evaluate(
            outer_test,
            documents,
            tokenizer,
            outer_model,
            device,
            max_answer_tokens=outer_max_answer_tokens,
        )
        expected = candidate_predictions(selector, outer_predictions, documents)
        selections = selected_ious(outer_predictions, expected, threshold)
        folds.append(
            {
                "outer_fold": outer_fold,
                "train_conversations": len({row["transcript_id"] for row in outer_train}),
                "test_conversations": sorted({row["transcript_id"] for row in outer_test}),
                "threshold": threshold,
                "inner_threshold_scores": threshold_scores,
                "inner_locator_folds": inner_summaries,
                "inner_predictions": inner_predictions,
                "outer_training_features": feature_count,
                "outer_auxiliary_rows": len(outer_auxiliary),
                "outer_independent_auxiliary_rows": len(outer_independent),
                "outer_last_loss": last_loss,
                "outer_parameter_summary": parameter_summary,
                "outer_max_answer_tokens": outer_max_answer_tokens,
                "baseline_mean_tiou": float(
                    np.mean([row["baseline_iou"] for row in selections.values()])
                ),
                "selected_mean_tiou": mean_iou(selections),
                "candidate_counts": dict(
                    Counter(row["candidate"] for row in selections.values())
                ),
                "outer_predictions": outer_predictions,
                "selections": selections,
                "seconds": time.monotonic() - started,
            }
        )
        args.output.write_text(
            json.dumps(partial_payload(args, accuracy, folds), ensure_ascii=False, indent=2)
            + "\n"
        )
        print(
            json.dumps(
                {
                    "outer_fold": outer_fold,
                    "baseline_mean_tiou": folds[-1]["baseline_mean_tiou"],
                    "selected_mean_tiou": folds[-1]["selected_mean_tiou"],
                    "threshold": threshold,
                    "seconds": folds[-1]["seconds"],
                }
            ),
            flush=True,
        )
        del outer_model, selector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = partial_payload(args, accuracy, folds)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "baseline_score": payload["baseline_score"],
                "selected_score": payload["selected_score"],
                "bootstrap": payload["bootstrap"],
                "gpu_peak_gib": torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0.0,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
