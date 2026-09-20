"""Inference-only loader for the enriched Medical Ridge selector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medical_codex.runtime.enriched_features import feature


class EnrichedSelector:
    """Choose between the Qwen baseline and one locator's span variants."""

    SOURCE = {
        "median_raw": "ft_raw",
        "median_clause": "ft_clause",
        "median_sentence": "ft_sentence",
    }

    def __init__(self, artifact: str | Path):
        payload = json.loads(Path(artifact).read_text())
        if payload.get("format") != "medical-multiseed-selector-v1" or payload.get("runs") != 1:
            raise ValueError("expected a single-run enriched selector artifact")
        if not str(payload.get("selector_model", "ridge")).startswith("ridge"):
            raise ValueError("only Ridge artifacts are supported")
        self.candidates = tuple(payload["candidate_names"])
        if any(name not in self.SOURCE for name in self.candidates):
            raise ValueError("artifact contains unsupported candidate names")
        self.mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
        self.scale = np.asarray(payload["scaler_scale"], dtype=np.float64)
        self.coef = np.asarray(payload["ridge_coef"], dtype=np.float64)
        self.intercept = float(payload["ridge_intercept"])
        self.threshold = float(payload["switch_threshold"])
        if not (self.mean.shape == self.scale.shape == self.coef.shape):
            raise ValueError("inconsistent selector artifact shapes")
        if len(self.mean) != int(payload["feature_count"]):
            raise ValueError("selector feature count does not match artifact")

    @staticmethod
    def _row(prediction: dict) -> dict:
        spans = prediction["spans"]
        raw = list(spans["ft_raw"])
        return {
            "question_id": prediction.get("question_id", "inference"),
            "transcript_id": prediction.get("transcript_id", "inference"),
            "question": prediction["question"],
            "answer_true": bool(prediction.get("answer_true", True)),
            "spans": {
                "baseline": list(spans["baseline"]),
                "median_raw": raw,
                "median_clause": list(spans["ft_clause"]),
                "median_sentence": list(spans["ft_sentence"]),
            },
            "seed_raw_spans": [raw],
            "seed_qa": [prediction["qa"]],
            "consensus_mean": 1.0,
            "consensus_min": 1.0,
        }

    def expected_gain(self, row: dict, words: list[dict], candidate: str) -> float:
        vector = np.asarray(feature(row, words, candidate), dtype=np.float64)
        if vector.shape != self.mean.shape:
            raise ValueError("selector feature schema changed after artifact export")
        return float(self.intercept + ((vector - self.mean) / self.scale) @ self.coef)

    def choose(self, prediction: dict, words: list[dict]) -> dict:
        row = self._row(prediction)
        gains = {name: self.expected_gain(row, words, name) for name in self.candidates}
        best = max(gains, key=gains.get)
        chosen = best if gains[best] > self.threshold else "baseline"
        source = self.SOURCE.get(chosen, "baseline")
        return {
            "candidate": source,
            "span": prediction["spans"][source],
            "expected_gain": gains[best],
            "switch_threshold": self.threshold,
            "candidate_expected_gains": gains,
        }
