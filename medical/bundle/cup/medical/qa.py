"""Evidence grounded ten question answering with Qwen3-8B."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path

from cup.common import ROOT
from cup.medical.triton.localize import evidence_from_answer

LOG = logging.getLogger(__name__)
MODEL_PATH = Path(os.environ.get("MEDICAL_QA_MODEL", ROOT / "models" / "medical" / "qwen3-8b"))
CONFIG_PATH = ROOT / "cup" / "medical" / "config.json"
PROMPT_VERSION = "triton-v1b-20260917"
SYSTEM = ("You are a meticulous clinical documentation auditor. You answer yes/no questions about a doctor-patient "
          "conversation strictly from its transcript. A question is 'yes' only if the transcript explicitly states it; "
          "check drug names, doses, units, durations, dates, who said it, negations and the final agreed plan. Near-miss "
          "statements (a different dose, a different drug, a plan that was discussed but rejected) are 'no'.")
SYSTEM_V2 = ("You are a meticulous clinical documentation auditor. You answer yes/no questions about a doctor-patient "
             "conversation strictly from its transcript. A question is 'yes' only if the transcript explicitly states the SAME "
             "fact: same drug, same dose and unit, same duration, same date, same person, same polarity. If the transcript says "
             "the opposite, a different value, or does not mention it, the answer is 'no'. Most questions are traps built from "
             "near-miss changes, so compare the question against the quote word by word before answering.")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"a", "an", "and", "are", "be", "did", "do", "does", "for", "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "will", "with"}


def yes_threshold(expected_tiou: float) -> float:
    if not 0 <= expected_tiou <= 1:
        raise ValueError("expected_tiou must be in [0, 1]")
    return 0.4 / (0.8 + 1.2 * expected_tiou)


def build_prompt(words: list[dict], questions: list[str]) -> str:
    """Use Triton's v1b prompt, which asks for a complete evidence clause."""
    text = " ".join(f"[{word['i']}]{word['text']}" for word in words)
    queries = "\n".join(f"Q{index + 1}: {question}" for index, question in enumerate(questions))
    return (
        f"Transcript with word indices:\n{text}\n\nQuestions:\n{queries}\n\n"
        "For EVERY question return one JSON object with keys: q (1-based question number), answer (true/false), "
        "confidence (0-1, how confident you are in your answer), start (index of the first word of the shortest "
        "passage that supports or best addresses the question), end (index of its last word, end >= start), "
        "quote (copy verbatim the COMPLETE clause that states the fact, from the previous punctuation mark to the next, "
        "typically 5-15 words; not a 2-word fragment). "
        "Answer with a JSON array of exactly "
        f"{len(questions)} objects and nothing else."
    )


def build_prompt_v2(words: list[dict], questions: list[str]) -> str:
    """Ask for an explicit fact comparison and a tight verbatim evidence span."""
    text = " ".join(f"[{word['i']}]{word['text']}" for word in words)
    queries = "\n".join(f"Q{index + 1}: {question}" for index, question in enumerate(questions))
    return (
        f"Transcript with word indices:\n{text}\n\nQuestions:\n{queries}\n\n"
        "For EVERY question return one JSON object with these keys IN THIS ORDER: q (1-based number); "
        "quote (the exact transcript words, 3-15 words, of the single passage most relevant to the question, copied verbatim); "
        "start and end (word indices of the first and last word of that quote); "
        "question_claims (the specific facts the question asserts); transcript_says (what the quote says about them); "
        "verdict (SAME, DIFFERENT_VALUE, OPPOSITE, NOT_MENTIONED); answer (true only if verdict is SAME); "
        f"confidence (0-1 probability that the answer is yes). Output a JSON array of exactly {len(questions)} objects and nothing else."
    )


def _extract_json(text: str) -> list[dict] | dict:
    start = text.find("[")
    if start < 0:
        start = text.find("{")
    if start < 0:
        raise ValueError("model output contains no JSON")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, (dict, list)):
        raise ValueError("model output is not a JSON object or array")
    return value


def _fallback_span(question: str, words: list[dict], segments: list[dict], duration: float) -> tuple[float, float]:
    tokens = set(_WORD_RE.findall(question.lower())) - _STOPWORDS
    if segments:
        segment = max(
            segments,
            key=lambda s: len(tokens & set(_WORD_RE.findall(str(s.get("text", "")).lower()))),
        )
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start + 1.0))
    elif words:
        start = float(words[0]["start"])
        end = float(words[min(12, len(words) - 1)]["end"])
    else:
        start, end = 0.0, min(1.0, duration)
    start = max(0.0, min(start, duration - 0.01))
    end = min(duration, max(start + 0.01, end))
    return round(start, 3), round(end, 3)


def _interpret(
    raw: list | dict,
    questions: list[str],
    transcript: dict,
    threshold: float,
    confidence_is_p_yes: bool = False,
) -> tuple[dict, list[dict]]:
    items = raw if isinstance(raw, list) else raw.get("answers")
    if not isinstance(items, list) or len(items) != len(questions):
        raise ValueError(f"expected {len(questions)} answer objects, got {len(items) if isinstance(items, list) else type(items).__name__}")
    if all(isinstance(item, dict) and "q" in item for item in items):
        numbered = {int(item["q"]): item for item in items}
        if set(numbered) != set(range(1, len(questions) + 1)):
            raise ValueError("question numbers are missing or duplicated")
        items = [numbered[i] for i in range(1, len(questions) + 1)]
    words = transcript["words"]
    duration = float(transcript["duration_s"])
    answers, starts, ends, diagnostics = [], [], [], []
    for question, item in zip(questions, items):
        if not isinstance(item, dict):
            raise ValueError("answer item is not an object")
        model_answer = item.get("answer")
        raw_confidence = item.get("confidence")
        confidence_flipped = (
            not confidence_is_p_yes
            and raw_confidence is not None
            and type(model_answer) is bool
            and not model_answer
        )
        if raw_confidence is not None:
            probability = float(raw_confidence)
            if confidence_flipped:
                probability = 1 - probability
        elif item.get("p_yes") is not None:
            probability = float(item["p_yes"])
        elif type(model_answer) is bool:
            probability = 0.95 if model_answer else 0.05
        else:
            raise ValueError("answer has neither confidence, p_yes nor a JSON boolean")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("p_yes is outside [0, 1]")
        # v1b asks for confidence in the chosen answer; the boolean is primary.
        yes = model_answer if type(model_answer) is bool else probability > threshold
        start = end = None
        fallback = False
        if yes:
            try:
                span = evidence_from_answer(item, words, mode="clause")
                if span is None:
                    raise ValueError("no quote or valid word indices")
                start, end = map(float, span)
                if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + 0.1):
                    raise ValueError("word timestamps invalid")
                start, end = round(max(0.0, start), 3), round(min(duration, end), 3)
            except (KeyError, TypeError, ValueError):
                start, end = _fallback_span(question, words, transcript.get("segments", []), duration)
                fallback = True
        answers.append(yes)
        starts.append(start)
        ends.append(end)
        diagnostics.append({"p_yes": probability, "raw_confidence": raw_confidence,
                            "confidence_flipped": confidence_flipped,
                            "fallback_span": fallback,
                            "start_word": item.get("start_word", item.get("start")),
                            "end_word": item.get("end_word", item.get("end")),
                            "model_answer": model_answer, "quote": str(item.get("quote", ""))[:200]})
    return {"answers": answers, "evidence_start": starts, "evidence_end": ends}, diagnostics


class QwenEvidenceQA:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        expected_tiou: float | None = None,
        prompt_version: str = "v1b",
    ):
        self.model_path = model_path
        if prompt_version not in {"v1b", "v2"}:
            raise ValueError(f"unsupported prompt version: {prompt_version}")
        self.prompt_version = prompt_version
        if expected_tiou is None:
            configured = json.loads(CONFIG_PATH.read_text()).get("expected_tiou", 0.5) if CONFIG_PATH.exists() else 0.5
            expected_tiou = float(os.environ.get("MEDICAL_EXPECTED_TIOU", configured))
        self.expected_tiou = expected_tiou
        self.yes_threshold = yes_threshold(self.expected_tiou)
        self.tokenizer = None
        self.model = None

    def warmup(self) -> None:
        if self.model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Qwen3-8B weights not found at {self.model_path}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path), dtype=torch.bfloat16, device_map={"": 0},
            low_cpu_mem_usage=True, attn_implementation="sdpa",
        )
        self.model.eval()

    def _generate(self, messages: list[dict]) -> str:
        import torch

        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt", return_dict=True,
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=2000, do_sample=False, max_time=45,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    def generate_raw(self, transcript: dict, questions: list[str]) -> list | dict:
        self.warmup()
        prompt = (
            build_prompt_v2(transcript["words"], questions)
            if self.prompt_version == "v2"
            else build_prompt(transcript["words"], questions)
        )
        messages = [
            {
                "role": "system",
                "content": SYSTEM_V2 if self.prompt_version == "v2" else SYSTEM,
            },
            {"role": "user", "content": prompt},
        ]
        for attempt in range(2):
            output = self._generate(messages)
            try:
                parsed = _extract_json(output)
                _interpret(
                    parsed,
                    questions,
                    transcript,
                    self.yes_threshold,
                    confidence_is_p_yes=self.prompt_version == "v2",
                )
                return parsed
            except (ValueError, TypeError, KeyError) as exc:
                LOG.warning("Qwen response invalid (attempt %s): %s", attempt + 1, exc)
                messages.append({"role": "assistant", "content": output})
                messages.append({"role": "user", "content": f"Your JSON was invalid: {exc}. Return corrected JSON only."})
        raise ValueError("Qwen returned invalid answers twice")

    def answer(self, transcript: dict, questions: list[str]) -> tuple[dict, list[dict]]:
        return _interpret(
            self.generate_raw(transcript, questions),
            questions,
            transcript,
            self.yes_threshold,
            confidence_is_p_yes=self.prompt_version == "v2",
        )
