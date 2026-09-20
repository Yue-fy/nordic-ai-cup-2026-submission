"""Production evidence refinement for unseen Medical conversations.

The Qwen service remains responsible for the YES/NO decision and its baseline
evidence clause.  This component runs a compact extractive QA locator only for
YES answers, generates raw/clause/sentence spans, and applies the fitted Ridge
gate.  It never uses filenames, audio hashes, question IDs, or public scores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from medical_codex.runtime.enriched_selector import EnrichedSelector
from medical_codex.runtime.conservative_selector import ConservativeSelector
from medical_codex.runtime.simple_selector import SimpleSelector


LOG = logging.getLogger(__name__)


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
    punctuation = ".?!" if sentence else ".?!;, :".replace(" ", "")
    while first > 0 and not str(words[first - 1]["text"]).rstrip().endswith(tuple(punctuation)):
        first -= 1
    while last < len(words) - 1 and not str(words[last]["text"]).rstrip().endswith(tuple(punctuation)):
        last += 1
    return first, last


def seconds(words: list[dict], indices: tuple[int, int], shift: float) -> list[float]:
    first, last = indices
    start = max(0.0, float(words[first]["start"]) + shift)
    end = max(start + 0.05, float(words[last]["end"]) + shift)
    return [start, end]


class EvidenceRefiner:
    """Load the full-data locator and apply an accepted selector artifact."""

    def __init__(
        self,
        model_path: str | Path,
        selector_path: str | Path,
        *,
        device: str = "cuda",
        max_answer_tokens: int = 26,
        timestamp_shift: float = -0.1,
    ):
        self.model_path = Path(model_path)
        selector_format = json.loads(Path(selector_path).read_text()).get("format")
        if selector_format == "medical-multiseed-selector-v1":
            self.selector = EnrichedSelector(selector_path)
        elif selector_format == "medical-conservative-selector-v1":
            self.selector = ConservativeSelector(selector_path)
        elif selector_format == "medical-simple-ridge-selector-v1":
            self.selector = SimpleSelector(selector_path)
        else:
            raise ValueError(f"unsupported evidence selector format: {selector_format}")
        self.device = device
        self.max_answer_tokens = max_answer_tokens
        self.timestamp_shift = timestamp_shift
        self.tokenizer = None
        self.model = None

    def warmup(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_path).to(self.device).eval()

    @torch.inference_mode()
    def _locate(self, question: str, context: str) -> dict:
        self.warmup()
        encoded = self.tokenizer(
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
        output = self.model(**{key: value.to(self.device) for key, value in encoded.items()})
        start_logits = output.start_logits.float().cpu()
        end_logits = output.end_logits.float().cpu()
        candidates: list[tuple[float, int, int]] = []
        for feature_index in range(start_logits.shape[0]):
            context_tokens = [i for i, sid in enumerate(sequence_ids[feature_index]) if sid == 1]
            starts = sorted(context_tokens, key=lambda i: float(start_logits[feature_index, i]), reverse=True)[:16]
            ends = sorted(context_tokens, key=lambda i: float(end_logits[feature_index, i]), reverse=True)[:16]
            for first in starts:
                for last in ends:
                    if last < first or last - first + 1 > self.max_answer_tokens:
                        continue
                    char_start = int(offsets[feature_index, first, 0])
                    char_end = int(offsets[feature_index, last, 1])
                    if char_end <= char_start:
                        continue
                    value = float(start_logits[feature_index, first] + end_logits[feature_index, last])
                    candidates.append((value, char_start, char_end))
        if not candidates:
            raise RuntimeError("extractive locator produced no valid context span")
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

    def refine(self, transcript_document: dict, questions: list[str], response: dict) -> tuple[dict, list[dict]]:
        """Return a copy of ``response`` with conservatively refined YES spans."""
        answers = list(response["answers"])
        starts = list(response["evidence_start"])
        ends = list(response["evidence_end"])
        if not (len(answers) == len(starts) == len(ends) == len(questions)):
            raise ValueError("Medical response arrays do not match the question count")
        words = transcript_document["words"]
        context, character_offsets = transcript(words)
        diagnostics: list[dict] = []
        for index, (question, answer) in enumerate(zip(questions, answers, strict=True)):
            if not answer or starts[index] is None or ends[index] is None:
                diagnostics.append({"candidate": "not_applicable", "question_index": index})
                continue
            try:
                qa = self._locate(question, context)
                indices = word_span(qa["char_start"], qa["char_end"], character_offsets)
                if indices is None:
                    raise ValueError("locator character span does not overlap an ASR word")
                raw = seconds(words, indices, self.timestamp_shift)
                clause = seconds(words, expand(words, *indices, sentence=False), self.timestamp_shift)
                sentence = seconds(words, expand(words, *indices, sentence=True), self.timestamp_shift)
                prediction = {
                    "question": question,
                    "answer_true": True,
                    "spans": {
                        "baseline": [float(starts[index]), float(ends[index])],
                        "ft_raw": raw,
                        "ft_clause": clause,
                        "ft_sentence": sentence,
                    },
                    "qa": qa,
                }
                choice = self.selector.choose(prediction, words)
                starts[index], ends[index] = map(lambda value: round(float(value), 3), choice["span"])
                diagnostics.append({"question_index": index, **choice, "locator_text": qa["text"]})
            except Exception as exc:
                # The accepted Qwen clause is the safety policy.  A locator or
                # selector failure must never turn a valid request into the
                # service's coarse all-YES emergency response.
                LOG.exception("Evidence refinement failed for question %s", index)
                diagnostics.append(
                    {
                        "candidate": "baseline",
                        "question_index": index,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        return {
            "answers": answers,
            "evidence_start": starts,
            "evidence_end": ends,
        }, diagnostics
