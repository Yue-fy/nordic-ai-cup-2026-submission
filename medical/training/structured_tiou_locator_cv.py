#!/usr/bin/env python3
"""Conversation-grouped CV for a temporal-IoU-aware evidence locator.

The existing locator is trained with exact start/end token cross entropy.  That
objective treats a one-token boundary miss like a completely unrelated span,
although the competition scores continuous temporal IoU.  This experiment adds
a structured distribution over every valid span in a transcript window.  The
target distribution is derived from each candidate's temporal IoU with the
gold interval and is used only in the training conversations.

Inference is unchanged: the model emits ordinary Hugging Face QA start/end
logits, so a successful model can replace the current locator without adding a
new runtime dependency or latency-heavy decoder.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from extractive_qa_span import AUDIT, CACHE, transcript
from finetune_qa_span_cv import evaluate, gold_character_span


class StructuredFeatures(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def _token_times(
    sequence_ids: list[int | None],
    token_offsets: list[tuple[int, int]],
    word_offsets: list[tuple[int, int]],
    words: list[dict],
) -> tuple[list[float], list[float], list[bool]]:
    starts = [-1.0] * len(token_offsets)
    ends = [-1.0] * len(token_offsets)
    mask = [False] * len(token_offsets)
    for token_index, (sequence_id, (char_start, char_end)) in enumerate(
        zip(sequence_ids, token_offsets, strict=True)
    ):
        if sequence_id != 1 or char_end <= char_start:
            continue
        overlapping = [
            word_index
            for word_index, (word_start, word_end) in enumerate(word_offsets)
            if word_end > char_start and word_start < char_end
        ]
        if not overlapping:
            continue
        starts[token_index] = float(words[overlapping[0]]["start"])
        ends[token_index] = float(words[overlapping[-1]]["end"])
        mask[token_index] = True
    return starts, ends, mask


def structured_training_features(row: dict, document: dict, tokenizer) -> list[dict]:
    words = document["words"]
    context, word_offsets = transcript(words)
    gold = (float(row["gold"][0]), float(row["gold"][1]))
    target_start, target_end = gold_character_span(words, word_offsets, gold)
    encoded = tokenizer(
        row["question"],
        context,
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    output: list[dict] = []
    for feature_index in range(len(encoded["input_ids"])):
        sequence_ids = encoded.sequence_ids(feature_index)
        context_tokens = [index for index, sid in enumerate(sequence_ids) if sid == 1]
        offsets = encoded["offset_mapping"][feature_index]
        cls = encoded["input_ids"][feature_index].index(tokenizer.cls_token_id)
        start = end = cls
        has_answer = False
        if context_tokens:
            window_start = offsets[context_tokens[0]][0]
            window_end = offsets[context_tokens[-1]][1]
            if window_start <= target_start and window_end >= target_end:
                start = next(index for index in context_tokens if offsets[index][1] > target_start)
                end = next(
                    index for index in reversed(context_tokens) if offsets[index][0] < target_end
                )
                has_answer = True
        token_starts, token_ends, context_mask = _token_times(
            sequence_ids, offsets, word_offsets, words
        )
        output.append(
            {
                "input_ids": torch.tensor(encoded["input_ids"][feature_index], dtype=torch.long),
                "attention_mask": torch.tensor(
                    encoded["attention_mask"][feature_index], dtype=torch.long
                ),
                "start_positions": torch.tensor(start, dtype=torch.long),
                "end_positions": torch.tensor(end, dtype=torch.long),
                "token_starts": torch.tensor(token_starts, dtype=torch.float32),
                "token_ends": torch.tensor(token_ends, dtype=torch.float32),
                "context_mask": torch.tensor(context_mask, dtype=torch.bool),
                "gold_start": torch.tensor(gold[0], dtype=torch.float32),
                "gold_end": torch.tensor(gold[1], dtype=torch.float32),
                "has_answer": torch.tensor(has_answer, dtype=torch.bool),
            }
        )
    return output


def temporal_iou_matrix(
    token_starts: torch.Tensor,
    token_ends: torch.Tensor,
    gold_start: torch.Tensor,
    gold_end: torch.Tensor,
) -> torch.Tensor:
    starts = token_starts[:, None]
    ends = token_ends[None, :]
    intersection = (torch.minimum(ends, gold_end) - torch.maximum(starts, gold_start)).clamp_min(0)
    union = torch.maximum(ends, gold_end) - torch.minimum(starts, gold_start)
    return intersection / union.clamp_min(1e-6)


def structured_loss(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    structured_weight: float,
    temperature: float,
    max_answer_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hard = (
        F.cross_entropy(start_logits, batch["start_positions"])
        + F.cross_entropy(end_logits, batch["end_positions"])
    ) / 2
    losses = []
    for item in range(start_logits.shape[0]):
        if not bool(batch["has_answer"][item]):
            continue
        indices = torch.nonzero(batch["context_mask"][item], as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        first = indices[:, None]
        last = indices[None, :]
        valid = (last >= first) & (last - first + 1 <= max_answer_tokens)
        scores = start_logits[item, indices][:, None] + end_logits[item, indices][None, :]
        utilities = temporal_iou_matrix(
            batch["token_starts"][item, indices],
            batch["token_ends"][item, indices],
            batch["gold_start"][item],
            batch["gold_end"][item],
        )
        scores = scores[valid]
        utilities = utilities[valid]
        target = torch.softmax(utilities / temperature, dim=0)
        log_prob = torch.log_softmax(scores, dim=0)
        losses.append(-(target * log_prob).sum())
    structured = torch.stack(losses).mean() if losses else hard.new_zeros(())
    combined = (1.0 - structured_weight) * hard + structured_weight * structured
    return combined, hard.detach(), structured.detach()


def bootstrap(values: dict[str, list[float]], seed: int = 260920, draws: int = 20000) -> dict:
    rng = np.random.default_rng(seed)
    per_conversation = np.asarray([np.mean(group) for group in values.values()])
    samples = rng.choice(
        per_conversation, size=(draws, len(per_conversation)), replace=True
    ).mean(axis=1)
    return {
        "mean_delta": float(per_conversation.mean()),
        "ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "probability_positive": float(np.mean(samples > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--structured-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-answer-tokens", type=int, default=81)
    parser.add_argument("--reference-oof", type=Path)
    parser.add_argument("--seed", type=int, default=260920)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    records = json.loads((AUDIT / "records.json").read_text())
    rows = [row for row in records if row["label"]]
    accuracy = sum(bool(row["correct"]) for row in records) / len(records)
    documents = {}
    for path in CACHE.glob("*.json"):
        document = json.loads(path.read_text())
        documents[document["audio_filename"]] = document

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    groups = np.asarray([row["transcript_id"] for row in rows])
    splitter = GroupKFold(n_splits=args.folds)
    predictions = []
    fold_summaries = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(rows, groups=groups)):
        fold_started = time.monotonic()
        train_rows = [rows[index] for index in train_indices]
        test_rows = [rows[index] for index in test_indices]
        features = [
            feature
            for row in train_rows
            for feature in structured_training_features(
                row, documents[f"conversation_{row['transcript_id']}.mp3"], tokenizer
            )
        ]
        loader = DataLoader(
            StructuredFeatures(features),
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + fold),
            num_workers=0,
        )
        torch.manual_seed(args.seed + fold)
        locator = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
        optimizer = torch.optim.AdamW(locator.parameters(), lr=args.learning_rate, weight_decay=0.01)
        losses = []
        hard_losses = []
        structured_losses = []
        locator.train()
        for _epoch in range(args.epochs):
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                model_batch = {
                    key: batch[key].to(device) for key in ("input_ids", "attention_mask")
                }
                targets = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if key not in model_batch
                }
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
                ):
                    result = locator(**model_batch)
                    loss, hard, structured = structured_loss(
                        result.start_logits.float(),
                        result.end_logits.float(),
                        targets,
                        structured_weight=args.structured_weight,
                        temperature=args.temperature,
                        max_answer_tokens=args.max_answer_tokens,
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(locator.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                hard_losses.append(float(hard.cpu()))
                structured_losses.append(float(structured.cpu()))
        locator.eval()
        fold_predictions = evaluate(
            test_rows,
            documents,
            tokenizer,
            locator,
            device,
            max_answer_tokens=args.max_answer_tokens,
        )
        for item in fold_predictions:
            item["fold"] = fold
        predictions.extend(fold_predictions)
        fold_summaries.append(
            {
                "fold": fold,
                "train_conversations": len({row["transcript_id"] for row in train_rows}),
                "test_conversations": sorted({row["transcript_id"] for row in test_rows}),
                "train_features": len(features),
                "last_loss": losses[-1],
                "last_hard_loss": hard_losses[-1],
                "last_structured_loss": structured_losses[-1],
                "seconds": time.monotonic() - fold_started,
            }
        )
        del locator, optimizer, loader, features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"fold {fold + 1}/{args.folds} complete", flush=True)

    predictions.sort(key=lambda item: item["question_id"])
    names = list(predictions[0]["ious"])
    mean_tiou = {
        name: float(np.mean([row["ious"][name] for row in predictions])) for name in names
    }
    fold_deltas = []
    for fold in range(args.folds):
        subset = [row for row in predictions if row["fold"] == fold]
        fold_deltas.append(
            float(np.mean([row["ious"]["ft_raw"] - row["ious"]["baseline"] for row in subset]))
        )
    deltas: dict[str, list[float]] = {}
    for row in predictions:
        deltas.setdefault(row["transcript_id"], []).append(
            row["ious"]["ft_raw"] - row["ious"]["baseline"]
        )
    summary = {
        "experiment_id": "medical-20260920-structured-tiou-locator",
        "score_label": "development estimate",
        "architecture": "RoBERTa QA with exact-boundary CE plus temporal-IoU structured span distribution",
        "model": str(args.model),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": device,
        "folds": args.folds,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "structured_weight": args.structured_weight,
        "temperature": args.temperature,
        "max_answer_tokens": args.max_answer_tokens,
        "questions": len(records),
        "positive_questions": len(rows),
        "conversations": len(set(groups)),
        "accuracy_frozen": accuracy,
        "mean_tiou": mean_tiou,
        "scores": {name: 0.4 * accuracy + 0.6 * value for name, value in mean_tiou.items()},
        "paired_raw_vs_baseline": bootstrap(deltas),
        "fold_raw_vs_baseline_deltas": fold_deltas,
        "worst_fold_raw_vs_baseline": min(fold_deltas),
        "oracle_score": 0.4 * accuracy
        + 0.6 * float(np.mean([max(row["ious"].values()) for row in predictions])),
        "fold_summaries": fold_summaries,
        "runtime_seconds": time.monotonic() - started,
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0,
        "integrity": {
            "conversation_grouped_oof": True,
            "gold_used_only_in_training_folds_and_scoring": True,
            "public_feedback_training": False,
            "identity_routing": False,
        },
    }
    if args.reference_oof:
        reference = json.loads(args.reference_oof.read_text())
        reference_by_id = {row["question_id"]: row for row in reference}
        paired = {}
        for row in predictions:
            old = reference_by_id[row["question_id"]]
            paired.setdefault(row["transcript_id"], []).append(
                row["ious"]["ft_raw"] - old["ious"]["ft_raw"]
            )
        summary["paired_raw_vs_standard_locator"] = bootstrap(paired)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "oof_predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
