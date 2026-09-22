#!/usr/bin/env python3
"""Conversation-grouped fine-tuning of a start/end evidence locator.

Each fold trains only on other conversations.  The held-out predictions form a
strict group-wise OOF estimate for this model stage.  The upstream Qwen answers
remain frozen, so this experiment measures evidence localization only.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from extractive_qa_span import AUDIT, CACHE, expand, locate, seconds, tiou, transcript, word_span


class Features(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def gold_character_span(words: list[dict], offsets: list[tuple[int, int]], gold: tuple[float, float]) -> tuple[int, int]:
    selected = [
        index
        for index, word in enumerate(words)
        if float(word["end"]) > gold[0] and float(word["start"]) < gold[1]
    ]
    if not selected:
        first = min(range(len(words)), key=lambda i: abs(float(words[i]["start"]) - gold[0]))
        last = min(range(len(words)), key=lambda i: abs(float(words[i]["end"]) - gold[1]))
        selected = list(range(min(first, last), max(first, last) + 1))
    return offsets[selected[0]][0], offsets[selected[-1]][1]


def training_features(row: dict, document: dict, tokenizer) -> list[dict]:
    words = document["words"]
    context, character_offsets = transcript(words)
    target_start, target_end = gold_character_span(words, character_offsets, tuple(row["gold"]))
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
    for feature in range(len(encoded["input_ids"])):
        sequence_ids = encoded.sequence_ids(feature)
        context_tokens = [i for i, sid in enumerate(sequence_ids) if sid == 1]
        offsets = encoded["offset_mapping"][feature]
        cls = encoded["input_ids"][feature].index(tokenizer.cls_token_id)
        start = end = cls
        if context_tokens:
            window_start = offsets[context_tokens[0]][0]
            window_end = offsets[context_tokens[-1]][1]
            if window_start <= target_start and window_end >= target_end:
                start = next(i for i in context_tokens if offsets[i][1] > target_start)
                end = next(i for i in reversed(context_tokens) if offsets[i][0] < target_end)
        output.append(
            {
                "input_ids": torch.tensor(encoded["input_ids"][feature], dtype=torch.long),
                "attention_mask": torch.tensor(encoded["attention_mask"][feature], dtype=torch.long),
                "start_positions": torch.tensor(start, dtype=torch.long),
                "end_positions": torch.tensor(end, dtype=torch.long),
            }
        )
    return output


def evaluate(
    rows: list[dict],
    documents: dict[str, dict],
    tokenizer,
    model,
    device: str,
    max_answer_tokens: int = 81,
) -> list[dict]:
    predictions: list[dict] = []
    for row in rows:
        document = documents[f"conversation_{row['transcript_id']}.mp3"]
        words = document["words"]
        context, character_offsets = transcript(words)
        answer = locate(
            row["question"], context, tokenizer, model, device, max_answer_tokens
        )
        indices = word_span(answer["char_start"], answer["char_end"], character_offsets)
        raw = seconds(words, indices) if indices else None
        clause = seconds(words, expand(words, *indices, sentence=False)) if indices else None
        sentence = seconds(words, expand(words, *indices, sentence=True)) if indices else None
        baseline = tuple(row["spans"]["clause"]) if row["spans"]["clause"] else None
        gold = tuple(row["gold"])
        active = row["answer"] is True
        spans = {"baseline": baseline, "ft_raw": raw, "ft_clause": clause, "ft_sentence": sentence}
        predictions.append(
            {
                "question_id": row["question_id"],
                "transcript_id": row["transcript_id"],
                "question": row["question"],
                "answer_true": active,
                "gold": gold,
                "spans": spans,
                "ious": {name: tiou(gold, span) if active else 0.0 for name, span in spans.items()},
                "qa": answer,
                "agreement": tiou(baseline, clause),
            }
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepset/roberta-base-squad2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    seed = 260917
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_records = json.loads((AUDIT / "records.json").read_text())
    rows = [row for row in all_records if row["label"]]
    accuracy = sum(bool(row["correct"]) for row in all_records) / len(all_records)
    documents: dict[str, dict] = {}
    for path in CACHE.glob("*.json"):
        document = json.loads(path.read_text())
        documents[document["audio_filename"]] = document

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    groups = np.array([row["transcript_id"] for row in rows])
    splitter = GroupKFold(n_splits=args.folds)
    oof: list[dict] = []
    fold_summaries: list[dict] = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(rows, groups=groups)):
        fold_start = time.monotonic()
        train_rows = [rows[i] for i in train_indices]
        test_rows = [rows[i] for i in test_indices]
        features = [
            feature
            for row in train_rows
            for feature in training_features(
                row, documents[f"conversation_{row['transcript_id']}.mp3"], tokenizer
            )
        ]
        generator = torch.Generator().manual_seed(seed + fold)
        loader = DataLoader(
            Features(features),
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        model.train()
        losses: list[float] = []
        for _epoch in range(args.epochs):
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    result = model(**{key: value.to(device) for key, value in batch.items()})
                    loss = result.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        model.eval()
        predictions = evaluate(test_rows, documents, tokenizer, model, device)
        for prediction in predictions:
            prediction["fold"] = fold
        oof.extend(predictions)
        fold_summaries.append(
            {
                "fold": fold,
                "train_conversations": sorted({row["transcript_id"] for row in train_rows}),
                "test_conversations": sorted({row["transcript_id"] for row in test_rows}),
                "train_features": len(features),
                "last_loss": losses[-1],
                "seconds": time.monotonic() - fold_start,
            }
        )
        del model, optimizer, loader, features
        torch.cuda.empty_cache()
        print(f"fold {fold + 1}/{args.folds} complete", flush=True)

    oof.sort(key=lambda item: item["question_id"])
    names = list(oof[0]["ious"])
    mean_tiou = {name: float(np.mean([row["ious"][name] for row in oof])) for name in names}
    scores = {name: 0.4 * accuracy + 0.6 * value for name, value in mean_tiou.items()}
    summary = {
        "model": args.model,
        "device": device,
        "folds": args.folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "n": len(all_records),
        "positives": len(rows),
        "accuracy_frozen": accuracy,
        "mean_tiou": mean_tiou,
        "scores": scores,
        "oracle_score": 0.4 * accuracy + 0.6 * float(np.mean([max(row["ious"].values()) for row in oof])),
        "fold_summaries": fold_summaries,
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0,
        "seconds": time.monotonic() - started,
        "caveat": "Conversation-grouped OOF for locator training; upstream Qwen prompt was previously developed on this corpus.",
    }
    (output / "oof_predictions.json").write_text(json.dumps(oof, ensure_ascii=False, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
