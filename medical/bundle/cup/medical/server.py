"""Medical appointment HTTP service. Start with ``uvicorn cup.medical.server:app``."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI

from cup.common import ROOT, protocol, source_commit
from cup.medical.asr import WhisperXTranscriber
from cup.medical.qa import PROMPT_VERSION, QwenEvidenceQA, _fallback_span
from medical_codex.runtime.evidence_refiner import EvidenceRefiner

DTO = protocol("medical-appointment")
LOG = logging.getLogger(__name__)
QA = QwenEvidenceQA()
ASR = WhisperXTranscriber()
REFINER = EvidenceRefiner(
    os.environ.get(
        "MEDICAL_REFINER_MODEL",
        ROOT / "models" / "medical" / "locator-2ep-external-holdout" / "model",
    ),
    os.environ.get(
        "MEDICAL_REFINER_SELECTOR",
        ROOT / "medical_codex" / "artifacts" / "simple_selector_2ep_external_holdout.json",
    ),
    device=os.environ.get("MEDICAL_REFINER_DEVICE", "cuda"),
    max_answer_tokens=int(os.environ.get("MEDICAL_REFINER_MAX_ANSWER_TOKENS", "81")),
    timestamp_shift=-0.1,
)
LOCK = threading.Lock()
COMMIT = source_commit()
CANDIDATE = os.environ.get("MEDICAL_RELEASE_CANDIDATE", "medical-development-service")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    QA.warmup()
    REFINER.warmup()
    if os.environ.get("MEDICAL_PRELOAD_ASR") == "1":
        ASR.warmup()
    yield


app = FastAPI(lifespan=lifespan)


def _record(entry: dict) -> None:
    path = os.environ.get("MEDICAL_PREDICTIONS_JSONL")
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


@app.get("/")
def index():
    return {
        "service": "medical-appointment",
        "ready": QA.model is not None and REFINER.model is not None,
        "candidate": CANDIDATE,
    }


@app.post("/predict", response_model=DTO.ASRQuestionResponseDto)
def predict(request: DTO.ASRQuestionRequestDto):
    started = time.perf_counter()
    audio = base64.b64decode(request.audio_base64, validate=True)
    sha256 = hashlib.sha256(audio).hexdigest()
    with LOCK:
        torch.cuda.reset_peak_memory_stats()
        transcript = None
        transcript_model = None
        try:
            transcript = ASR.transcribe(audio, request.audio_filename)
            transcript_model = transcript.get("model")
            asr_done = time.perf_counter()
            response, details = QA.answer(transcript, request.questions)
            response, refinement = REFINER.refine(transcript, request.questions, response)
            details = [
                {**item, "refinement": refinement[index]}
                for index, item in enumerate(details)
            ]
            fallback = False
        except Exception:
            LOG.exception("Medical prediction failed for sha256=%s", sha256)
            asr_done = time.perf_counter()
            spans = [
                _fallback_span(question, transcript["words"], transcript.get("segments", []),
                               float(transcript["duration_s"]))
                if transcript is not None else (0.0, 1.0)
                for question in request.questions
            ]
            response = {
                "answers": [True] * len(request.questions),
                "evidence_start": [start for start, _ in spans],
                "evidence_end": [end for _, end in spans],
            }
            details = []
            fallback = True
    result = DTO.ASRQuestionResponseDto.model_validate(response)
    finished = time.perf_counter()
    _record({
        "audio_filename": Path(request.audio_filename).name,
        "sha256": sha256,
        "questions": request.questions,
        "response": result.model_dump(),
        "details": details,
        "asr_model": transcript_model,
        "qa_model": "Qwen/Qwen3-8B bf16",
        "evidence_refiner": CANDIDATE,
        "prompt_version": PROMPT_VERSION,
        "expected_tiou": QA.expected_tiou,
        "yes_threshold": round(QA.yes_threshold, 5),
        "fallback_all_yes": fallback,
        "timing_s": {"asr": round(asr_done - started, 3),
                     "qa": round(finished - asr_done, 3),
                     "total": round(finished - started, 3)},
        "gpu_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "machine": os.uname().nodename,
        "git_commit": COMMIT,
        "date": time.strftime("%Y-%m-%d"),
    })
    return result
