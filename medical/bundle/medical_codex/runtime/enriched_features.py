"""Dependency-light feature extraction for the Medical evidence selector.

Keep this module free of scikit-learn so the fitted JSON selector can run in
the 5090 inference environment.  The matching training implementation lives in
``medical_codex.experiments.multiseed_locator_audit``; a regression test checks
that the two implementations remain numerically identical.
"""

from __future__ import annotations

import math
import re

import numpy as np


STOP = set(
    "a an the is are was were be been being did do does has have had will would could "
    "should can of to in at on for with and or as by it its this that these those patient "
    "doctor clinician he she they their his her you your i my we our".split()
)
QUESTION_CATEGORIES = (
    {"mg", "milligram", "dose", "daily", "times", "day", "days", "week", "weeks", "month", "months", "year", "years"},
    {"medicine", "medication", "treatment", "prescribed", "prescription", "taking", "take", "continue", "continued", "renewed"},
    {"normal", "stable", "unchanged", "healthy", "well", "assessment", "considered"},
    {"test", "tests", "blood", "pressure", "examination", "exam", "imaging", "mobility", "status"},
    {"symptom", "symptoms", "pain", "dry", "skin", "lesion", "growth", "sore", "side", "effects"},
    {"plan", "advice", "follow", "checkup", "checkups", "arrange", "referred", "referral"},
    {"no", "not", "none", "without", "absent", "free", "denies"},
)


def tiou(left: list[float], right: list[float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union if union > 0 else 0.0


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP


def span_text(words: list[dict], span: list[float]) -> str:
    return " ".join(
        str(word["text"])
        for word in words
        if float(word["end"]) > span[0] and float(word["start"]) < span[1]
    )


def feature(row: dict, words: list[dict], candidate: str) -> list[float]:
    """Reproduce the fitted selector's 54/57-column inference schema."""
    baseline = row["spans"]["baseline"]
    span = row["spans"][candidate]
    total = max(float(words[-1]["end"]), 1.0)
    bdur = max(baseline[1] - baseline[0], 0.05)
    cdur = max(span[1] - span[0], 0.05)
    starts = np.asarray([item[0] for item in row["seed_raw_spans"]])
    ends = np.asarray([item[1] for item in row["seed_raw_spans"]])
    logits = np.asarray([item["logit"] for item in row["seed_qa"]], dtype=float)
    margins = np.asarray([item["margin"] for item in row["seed_qa"]], dtype=float)
    candidate_match = re.fullmatch(r"seed(\d+)_(raw|clause|sentence)", candidate)
    candidate_seed = int(candidate_match.group(1)) if candidate_match else None
    multi_seed = len(row["seed_raw_spans"]) > 1
    if not multi_seed:
        candidate_logit = 0.0
        candidate_margin = 0.0
        candidate_agreement = 0.0
        candidate_is_max_margin = 0.0
    elif candidate_seed is None:
        candidate_logit = float(np.mean(logits))
        candidate_margin = float(np.mean(margins))
        candidate_agreement = row["consensus_mean"]
        candidate_is_max_margin = 0.0
    else:
        candidate_logit = float(logits[candidate_seed])
        candidate_margin = float(margins[candidate_seed])
        candidate_agreement = float(
            np.mean(
                [
                    tiou(row["seed_raw_spans"][candidate_seed], other)
                    for index, other in enumerate(row["seed_raw_spans"])
                    if index != candidate_seed
                ]
                or [1.0]
            )
        )
        candidate_is_max_margin = float(candidate_seed == int(np.argmax(margins)))
    question = tokens(row["question"])
    btok = tokens(span_text(words, baseline))
    ctok = tokens(span_text(words, span))
    return [
        float(candidate == "median_raw"),
        float(candidate == "medoid_raw"),
        float(candidate == "union_raw"),
        float(candidate == "intersection_raw"),
        float(candidate == "median_clause"),
        float(candidate == "median_sentence"),
        float(candidate == "seed0_raw"),
        float(candidate == "seed1_raw"),
        float(candidate == "seed2_raw"),
        float(candidate == "median_raw_pad025"),
        float(candidate == "median_raw_pad050"),
        float(candidate == "qwen14"),
        float(row["answer_true"]),
        tiou(baseline, span),
        bdur / 10.0,
        cdur / 10.0,
        math.log(cdur / bdur),
        (span[0] - baseline[0]) / 10.0,
        (span[1] - baseline[1]) / 10.0,
        abs(sum(span) / 2.0 - sum(baseline) / 2.0) / total,
        (sum(span) / 2.0) / total,
        row["consensus_mean"],
        row["consensus_min"],
        float(np.std(starts)) / 10.0,
        float(np.std(ends)) / 10.0,
        float(np.mean(logits)) / 20.0,
        float(np.std(logits)) / 20.0,
        float(np.mean(margins)),
        float(np.min(margins)),
        candidate_logit / 20.0,
        candidate_margin,
        candidate_agreement,
        candidate_is_max_margin,
        float(candidate == "qwen14" and bool(row.get("external", {}).get("answer", False))),
        float(row.get("external", {}).get("p_yes", 0.0)) if candidate == "qwen14" else 0.0,
        len(question) / 10.0,
        len(btok) / 20.0,
        len(ctok) / 20.0,
        len(question & btok) / max(1, len(question)),
        len(question & ctok) / max(1, len(question)),
        len(btok & ctok) / max(1, len(btok | ctok)),
        float(span[0] >= baseline[0] and span[1] <= baseline[1]),
        float(baseline[0] >= span[0] and baseline[1] <= span[1]),
        *[float(bool(question & category)) for category in QUESTION_CATEGORIES],
        *[float(bool(ctok & category)) for category in QUESTION_CATEGORIES],
    ]
