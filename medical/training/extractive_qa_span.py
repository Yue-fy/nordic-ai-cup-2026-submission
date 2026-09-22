#!/usr/bin/env python3
"""Evaluate a pretrained extractive-QA model as a temporal evidence locator.

The existing Qwen YES/NO decisions are frozen.  This experiment changes only
the evidence interval for questions which Qwen answered yes.  All model inputs
are available at inference time; gold timestamps are used only for scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "strategy_lab/research_20260917/medical_audit"
CACHE = ROOT / "cup/medical/asr_cache/whisperx-large-v3+wav2vec2-base-960h"


def tiou(left: tuple[float, float] | None, right: tuple[float, float] | None) -> float:
    if left is None or right is None:
        return 0.0
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union if union > 0 else 0.0


def transcript(words: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    pieces: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for word in words:
        if pieces:
            cursor += 1
        token = str(word["text"])
        offsets.append((cursor, cursor + len(token)))
        pieces.append(token)
        cursor += len(token)
    return " ".join(pieces), offsets


def word_span(char_start: int, char_end: int, offsets: list[tuple[int, int]]) -> tuple[int, int] | None:
    selected = [i for i, (start, end) in enumerate(offsets) if end > char_start and start < char_end]
    return (selected[0], selected[-1]) if selected else None


def expand(words: list[dict], first: int, last: int, sentence: bool) -> tuple[int, int]:
    punctuation = ".?!" if sentence else ".?!;,:"
    while first > 0 and not str(words[first - 1]["text"]).rstrip().endswith(tuple(punctuation)):
        first -= 1
    while last < len(words) - 1 and not str(words[last]["text"]).rstrip().endswith(tuple(punctuation)):
        last += 1
    return first, last


def seconds(words: list[dict], indices: tuple[int, int], shift: float = -0.1) -> tuple[float, float]:
    first, last = indices
    start = max(0.0, float(words[first]["start"]) + shift)
    end = max(start + 0.05, float(words[last]["end"]) + shift)
    return start, end


@torch.inference_mode()
def locate(
    question: str,
    context: str,
    tokenizer,
    model,
    device: str,
    max_answer_tokens: int = 81,
) -> dict:
    encoded = tokenizer(
        question,
        context,
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    encoded.pop("overflow_to_sample_mapping", None)
    sequence_ids = [encoded.sequence_ids(i) for i in range(encoded["input_ids"].shape[0])]
    output = model(**{key: value.to(device) for key, value in encoded.items()})
    start_logits = output.start_logits.float().cpu()
    end_logits = output.end_logits.float().cpu()
    candidates: list[tuple[float, int, int]] = []
    for feature in range(start_logits.shape[0]):
        context_tokens = [i for i, sid in enumerate(sequence_ids[feature]) if sid == 1]
        if not context_tokens:
            continue
        starts = sorted(context_tokens, key=lambda i: float(start_logits[feature, i]), reverse=True)[:16]
        ends = sorted(context_tokens, key=lambda i: float(end_logits[feature, i]), reverse=True)[:16]
        for first in starts:
            for last in ends:
                if last < first or last - first + 1 > max_answer_tokens:
                    continue
                char_start = int(offsets[feature, first, 0])
                char_end = int(offsets[feature, last, 1])
                if char_end <= char_start:
                    continue
                value = float(start_logits[feature, first] + end_logits[feature, last])
                candidates.append((value, char_start, char_end))
    if not candidates:
        raise RuntimeError("extractive QA produced no context span")
    candidates.sort(reverse=True)
    best = candidates[0]
    runner_up = next((item for item in candidates[1:] if item[1:] != best[1:]), best)
    return {
        "logit": best[0],
        "margin": best[0] - runner_up[0],
        "char_start": best[1],
        "char_end": best[2],
        "text": context[best[1] : best[2]],
    }


def score(rows: list[dict], key: str, accuracy: float) -> float:
    values = [row["ious"][key] for row in rows]
    return 0.4 * accuracy + 0.6 * float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepset/roberta-base-squad2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = json.loads((AUDIT / "records.json").read_text())
    positives = [row for row in records if row["label"]]
    accuracy = sum(bool(row["correct"]) for row in records) / len(records)

    documents: dict[str, dict] = {}
    for path in CACHE.glob("*.json"):
        document = json.loads(path.read_text())
        documents[document["audio_filename"]] = document

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device).eval()
    predictions: list[dict] = []
    for index, row in enumerate(positives, 1):
        document = documents[f"conversation_{row['transcript_id']}.mp3"]
        words = document["words"]
        context, character_offsets = transcript(words)
        answer = locate(row["question"], context, tokenizer, model, device)
        indices = word_span(answer["char_start"], answer["char_end"], character_offsets)
        if indices is None:
            raw = clause = sentence = None
        else:
            raw = seconds(words, indices)
            clause = seconds(words, expand(words, *indices, sentence=False))
            sentence = seconds(words, expand(words, *indices, sentence=True))
        gold = tuple(row["gold"])
        baseline = tuple(row["spans"]["clause"]) if row["spans"]["clause"] else None
        active = row["answer"] is True
        spans = {"baseline": baseline, "qa_raw": raw, "qa_clause": clause, "qa_sentence": sentence}
        ious = {name: tiou(gold, span) if active else 0.0 for name, span in spans.items()}
        predictions.append(
            {
                "question_id": row["question_id"],
                "transcript_id": row["transcript_id"],
                "question": row["question"],
                "answer_true": active,
                "gold": gold,
                "spans": spans,
                "ious": ious,
                "qa": answer,
                "agreement": tiou(baseline, clause),
            }
        )
        if index % 20 == 0 or index == len(positives):
            print(f"{index}/{len(positives)}", flush=True)

    summary = {
        "model": args.model,
        "device": device,
        "n": len(records),
        "positives": len(positives),
        "accuracy_frozen": accuracy,
        "scores": {name: score(predictions, name, accuracy) for name in predictions[0]["ious"]},
        "mean_tiou": {
            name: float(np.mean([row["ious"][name] for row in predictions]))
            for name in predictions[0]["ious"]
        },
        "oracle_score": 0.4 * accuracy
        + 0.6 * float(np.mean([max(row["ious"].values()) for row in predictions])),
        "changed_zero_to_overlap": sum(
            row["ious"]["baseline"] == 0 and row["ious"]["qa_clause"] > 0 for row in predictions
        ),
        "changed_overlap_to_zero": sum(
            row["ious"]["baseline"] > 0 and row["ious"]["qa_clause"] == 0 for row in predictions
        ),
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
        "seconds": time.monotonic() - started,
        "caveat": "Development data only. Gold is used for scoring and oracle, never as model input.",
    }
    (output / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
