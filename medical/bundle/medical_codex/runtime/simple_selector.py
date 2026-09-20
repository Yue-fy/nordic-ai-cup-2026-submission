"""Inference-only selector for the externally supported two-epoch locator."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from medical_codex.runtime.enriched_features import tiou


STOP = set(
    "a an the is are was were be been being did do does has have had will would could "
    "should can of to in at on for with and or as by it its this that these those patient "
    "doctor clinician he she they their his her you your i my we our".split()
)


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP


def span_text(words: list[dict], span: list[float]) -> str:
    return " ".join(
        str(word["text"])
        for word in words
        if float(word["end"]) > span[0] and float(word["start"]) < span[1]
    )


def feature(prediction: dict, words: list[dict]) -> list[float]:
    """Reproduce the gate used for the independent public-set diagnostic."""
    question = tokens(prediction["question"])
    baseline = prediction["spans"]["baseline"]
    candidate = prediction["spans"]["ft_raw"]
    baseline_tokens = tokens(span_text(words, baseline))
    candidate_tokens = tokens(prediction["qa"]["text"])
    total = max(float(words[-1]["end"]), 1.0)
    numbers = {item for item in question if item.isdigit()}

    def similarity(items: set[str]) -> list[float]:
        item_numbers = {item for item in items if item.isdigit()}
        return [
            len(question & items) / max(1, len(question)),
            len(question & items) / max(1, len(items)),
            float(bool(numbers - item_numbers)),
            float(bool(item_numbers - numbers)),
        ]

    union = baseline_tokens | candidate_tokens
    return [
        1.0,
        tiou(baseline, candidate),
        prediction["qa"]["logit"] / 20.0,
        prediction["qa"]["margin"],
        (baseline[1] - baseline[0]) / 10.0,
        (candidate[1] - candidate[0]) / 10.0,
        abs(sum(baseline) / 2 - sum(candidate) / 2) / total,
        (sum(candidate) / 2) / total,
        len(question) / 10.0,
        len(baseline_tokens) / 20.0,
        len(candidate_tokens) / 20.0,
        *similarity(baseline_tokens),
        *similarity(candidate_tokens),
        len(baseline_tokens & candidate_tokens) / max(1, len(union)),
        float(any(item in question for item in {"no", "not", "without", "free"})),
        float(any(item in candidate_tokens for item in {"no", "not", "without", "none"})),
    ]


class SimpleSelector:
    def __init__(self, artifact: str | Path):
        payload = json.loads(Path(artifact).read_text())
        if payload.get("format") != "medical-simple-ridge-selector-v1":
            raise ValueError("unsupported simple selector artifact")
        self.mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
        self.scale = np.asarray(payload["scaler_scale"], dtype=np.float64)
        self.coef = np.asarray(payload["ridge_coef"], dtype=np.float64)
        self.intercept = float(payload["ridge_intercept"])
        self.threshold = float(payload["switch_threshold"])
        if not (self.mean.shape == self.scale.shape == self.coef.shape):
            raise ValueError("inconsistent simple selector artifact shapes")
        if len(self.mean) != int(payload["feature_count"]):
            raise ValueError("simple selector feature count does not match artifact")

    def choose(self, prediction: dict, words: list[dict]) -> dict:
        vector = np.asarray(feature(prediction, words), dtype=np.float64)
        if vector.shape != self.mean.shape:
            raise ValueError("simple selector feature schema changed after artifact export")
        expected_gain = float(self.intercept + ((vector - self.mean) / self.scale) @ self.coef)
        candidate = "ft_raw" if expected_gain > self.threshold else "baseline"
        return {
            "candidate": candidate,
            "span": prediction["spans"][candidate],
            "expected_gain": expected_gain,
            "switch_threshold": self.threshold,
            "candidate_expected_gains": {"ft_raw": expected_gain},
        }
