"""Version-independent inference for the exported conservative Ridge gate."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


STOP = set(
    "a an the is are was were be been being did do does has have had will would could "
    "should can of to in at on for with and or as by it its this that these those patient "
    "doctor clinician he she they their his her you your i my we our".split()
)
NEGATIONS = {"no", "not", "never", "without", "none", "denies", "denied", "free"}
CANDIDATE_SCHEMA = ("ft_raw", "ft_clause", "ft_sentence")


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
    """Frozen 29-feature schema used by the grouped conservative selector."""
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


class ConservativeSelector:
    def __init__(self, artifact: str | Path):
        payload = json.loads(Path(artifact).read_text())
        if payload.get("format") != "medical-conservative-selector-v1":
            raise ValueError("unsupported selector artifact")
        self.mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
        self.scale = np.asarray(payload["scaler_scale"], dtype=np.float64)
        self.coef = np.asarray(payload["ridge_coef"], dtype=np.float64)
        self.intercept = float(payload["ridge_intercept"])
        self.threshold = float(payload["switch_threshold"])
        self.max_answer_tokens = int(payload["max_answer_tokens"])
        self.candidate_names = tuple(
            payload.get(
                "candidate_names", ("ft_raw", "ft_clause", "ft_sentence")
            )
        )
        if not (self.mean.shape == self.scale.shape == self.coef.shape):
            raise ValueError("inconsistent selector artifact shapes")
        if len(self.mean) != int(payload["feature_count"]):
            raise ValueError("selector feature count does not match artifact")

    def expected_gain(self, prediction: dict, candidate: str, words: list[dict]) -> float:
        vector = np.asarray(feature_vector(prediction, candidate, words), dtype=np.float64)
        if vector.shape != self.mean.shape:
            raise ValueError("selector feature schema changed after artifact export")
        standardized = (vector - self.mean) / self.scale
        return float(self.intercept + standardized @ self.coef)

    def choose(self, prediction: dict, words: list[dict]) -> dict:
        scored = {
            name: self.expected_gain(prediction, name, words)
            for name in self.candidate_names
            if prediction["spans"].get(name) is not None
        }
        best_candidate = max(scored, key=scored.get, default="baseline")
        best_expected_gain = scored.get(best_candidate)
        candidate = best_candidate
        if (
            candidate == "baseline"
            or best_expected_gain is None
            or best_expected_gain <= self.threshold
        ):
            candidate = "baseline"
        return {
            "candidate": candidate,
            "span": prediction["spans"][candidate],
            "expected_gain": best_expected_gain,
            "switch_threshold": self.threshold,
            "candidate_expected_gains": scored,
        }
