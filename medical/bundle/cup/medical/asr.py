"""WhisperX transcription with a content keyed cache.

The shared training cache is read from ``cup/medical/asr_cache``. New audio is
cached under ``models/medical/live_asr_cache`` so validation transcripts are not
accidentally committed to git.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from cup.common import ROOT, source_commit

MODEL_TAG = "whisperx-large-v3+wav2vec2-base-960h"
WHISPER_REPOSITORY = "Systran/faster-whisper-large-v3"
WHISPER_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
SHARED_CACHE = ROOT / "cup" / "medical" / "asr_cache" / MODEL_TAG
LIVE_CACHE = ROOT / "models" / "medical" / "live_asr_cache" / MODEL_TAG
SILERO_CACHE_DIRECTORY = "snakers4_silero-vad_master"
COMMIT = source_commit()


def _cache_path(directory: Path, sha256: str) -> Path:
    return directory / f"{sha256[:16]}.json"


def _read_cache(path: Path, sha256: str) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("sha256") != sha256:
        return None
    words = data.get("words")
    if not isinstance(words, list) or not words:
        raise ValueError(f"ASR cache has no words: {path}")
    if any(word.get("i") != i for i, word in enumerate(words)):
        raise ValueError(f"ASR cache has nonconsecutive word indices: {path}")
    return data


def cached_transcript(audio: bytes) -> dict | None:
    sha256 = hashlib.sha256(audio).hexdigest()
    for directory in (SHARED_CACHE, LIVE_CACHE):
        cached = _read_cache(_cache_path(directory, sha256), sha256)
        if cached is not None:
            return cached
    return None


def whisper_model_reference() -> str:
    """Prefer the pinned local snapshot so offline inference never resolves `main`."""
    explicit = os.environ.get("MEDICAL_WHISPER_MODEL")
    if explicit:
        return explicit
    hf_home = Path(os.environ.get("HF_HOME", ROOT / "models" / "hf"))
    snapshot = (
        hf_home
        / "hub"
        / f"models--{WHISPER_REPOSITORY.replace('/', '--')}"
        / "snapshots"
        / os.environ.get("MEDICAL_WHISPER_REVISION", WHISPER_REVISION)
    )
    if (snapshot / "config.json").is_file():
        return str(snapshot)
    return "large-v3"


def silero_cache_path() -> Path:
    torch_home = Path(os.environ.get("TORCH_HOME", ROOT / "models" / "torch"))
    return torch_home / "hub" / SILERO_CACHE_DIRECTORY


def require_offline_silero() -> Path:
    """Fail before WhisperX can make an implicit torch.hub network request."""
    cache = silero_cache_path()
    required = (
        cache / "hubconf.py",
        cache / "src" / "silero_vad" / "data" / "silero_vad.jit",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "offline Silero VAD cache is incomplete; stage the pinned cache before "
            f"starting the service. Missing: {missing}"
        )
    return cache


def _words_from_alignment(segments: list[dict], duration: float) -> list[dict]:
    words = []
    for segment in segments:
        for item in segment.get("words", []):
            word = {"i": len(words), "text": str(item.get("word", "")).strip()}
            start, end = item.get("start"), item.get("end")
            if start is None or end is None or float(end) <= float(start):
                word["interp"] = True
                word["start"] = None
                word["end"] = None
            else:
                word["start"] = round(max(0.0, float(start)), 3)
                word["end"] = round(min(duration, float(end)), 3)
            words.append(word)

    i = 0
    while i < len(words):
        if words[i]["start"] is not None:
            i += 1
            continue
        j = i
        while j < len(words) and words[j]["start"] is None:
            j += 1
        left = words[i - 1]["end"] if i else 0.0
        right = words[j]["start"] if j < len(words) else duration
        gap = max(0.01 * (j - i), right - left)
        width = gap / (j - i)
        for k in range(i, j):
            start = min(duration - 0.01, left + width * (k - i))
            words[k]["start"] = round(max(0.0, start), 3)
            words[k]["end"] = round(min(duration, start + max(0.01, width)), 3)
        i = j
    return words


class WhisperXTranscriber:
    def __init__(self, device: str = "cuda", batch_size: int = 8):
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.align_model = None
        self.align_metadata = None

    def warmup(self) -> None:
        if self.model is not None:
            return
        os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
        os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch"))
        require_offline_silero()
        import whisperx

        self.model = whisperx.load_model(
            whisper_model_reference(), self.device, compute_type="float16", language="en",
            vad_method="silero", download_root=str(ROOT / "models" / "whisperx"),
        )
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="en", device=self.device,
        )

    def transcribe(self, audio: bytes, audio_filename: str) -> dict:
        if os.environ.get("MEDICAL_BYPASS_ASR_CACHE") != "1":
            cached = cached_transcript(audio)
            if cached is not None:
                return cached
        self.warmup()
        import whisperx

        sha256 = hashlib.sha256(audio).hexdigest()
        started = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio)
            path = Path(tmp.name)
        try:
            waveform = whisperx.load_audio(str(path))
        finally:
            path.unlink(missing_ok=True)
        duration = len(waveform) / 16000
        raw = self.model.transcribe(waveform, batch_size=self.batch_size)
        aligned = whisperx.align(
            raw["segments"], self.align_model, self.align_metadata, waveform,
            self.device, return_char_alignments=False,
        )
        words = _words_from_alignment(aligned["segments"], duration)
        if not words:
            raise RuntimeError("WhisperX produced no aligned words")
        document = {
            "audio_filename": Path(audio_filename).name,
            "sha256": sha256,
            "model": MODEL_TAG,
            "revision": getattr(whisperx, "__version__", "unknown"),
            "duration_s": round(duration, 3),
            "words": words,
            "segments": [
                {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
                for s in aligned["segments"]
            ],
            "machine": os.uname().nodename,
            "git_commit": COMMIT,
            "date": time.strftime("%Y-%m-%d"),
            "timing_s": {"transcribe_and_align": round(time.perf_counter() - started, 3)},
        }
        LIVE_CACHE.mkdir(parents=True, exist_ok=True)
        target = _cache_path(LIVE_CACHE, sha256)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False))
        temporary.replace(target)
        return document
