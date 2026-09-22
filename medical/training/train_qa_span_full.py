#!/usr/bin/env python3
"""Train the production evidence locator on all supplied positive examples."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from extractive_qa_span import AUDIT, CACHE
from finetune_qa_span_cv import Features, training_features
from model_regularization import freeze_lower_encoder_layers
from structured_tiou_locator_cv import structured_loss, structured_training_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepset/roberta-base-squad2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=260917)
    parser.add_argument("--freeze-lower-layers", type=int, default=0)
    parser.add_argument(
        "--locator-loss", choices=("exact", "structured_tiou"), default="exact"
    )
    parser.add_argument("--structured-weight", type=float, default=0.5)
    parser.add_argument("--structured-temperature", type=float, default=0.05)
    parser.add_argument("--structured-max-answer-tokens", type=int, default=81)
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = json.loads((AUDIT / "records.json").read_text())
    rows = [row for row in records if row["label"]]
    documents = {}
    for path in CACHE.glob("*.json"):
        document = json.loads(path.read_text())
        documents[document["audio_filename"]] = document
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    feature_builder = (
        structured_training_features
        if args.locator_loss == "structured_tiou"
        else training_features
    )
    features = [
        feature
        for row in rows
        for feature in feature_builder(
            row, documents[f"conversation_{row['transcript_id']}.mp3"], tokenizer
        )
    ]
    loader = DataLoader(
        Features(features),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
    parameter_summary = freeze_lower_encoder_layers(model, args.freeze_lower_layers)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    losses = []
    amp_dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    model.train()
    for _epoch in range(args.epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device == "cuda"):
                if args.locator_loss == "structured_tiou":
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
                        structured_weight=args.structured_weight,
                        temperature=args.structured_temperature,
                        max_answer_tokens=args.structured_max_answer_tokens,
                    )
                else:
                    result = model(**{key: value.to(device) for key, value in batch.items()})
                    loss = result.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    model.save_pretrained(output / "model")
    tokenizer.save_pretrained(output / "model")
    manifest = {
        "base_model": args.model,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "train_conversations": len({row["transcript_id"] for row in rows}),
        "train_positives": len(rows),
        "train_features": len(features),
        "last_loss": losses[-1],
        "seed": seed,
        "parameter_summary": parameter_summary,
        "locator_loss": args.locator_loss,
        "structured_weight": args.structured_weight,
        "structured_temperature": args.structured_temperature,
        "structured_max_answer_tokens": args.structured_max_answer_tokens,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
