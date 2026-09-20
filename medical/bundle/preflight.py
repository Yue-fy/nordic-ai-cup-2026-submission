#!/usr/bin/env python3
"""Fail fast before the one-shot Medical evaluation begins."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parent
CANDIDATE = json.loads((ROOT / "candidate.json").read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(root: Path) -> tuple[str, int, int]:
    """Hash stable source files in a torch.hub repository cache."""
    entries = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        relative = path.relative_to(root).as_posix()
        entries.append((relative, path.stat().st_size, sha256(path)))
    digest = hashlib.sha256()
    for relative, size, file_digest in entries:
        digest.update(f"{relative}\0{size}\0{file_digest}\n".encode())
    return digest.hexdigest(), len(entries), sum(item[1] for item in entries)


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"preflight: missing {label}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="verify files and hashes without requiring a CUDA device",
    )
    args = parser.parse_args()
    locator = Path(
        os.environ.get(
            "MEDICAL_REFINER_MODEL",
            ROOT / "models/medical/active-locator/model",
        )
    )
    selector = Path(
        os.environ.get(
            "MEDICAL_REFINER_SELECTOR",
            ROOT / "medical_codex/artifacts/active_selector.json",
        )
    )
    qwen = Path(os.environ.get("MEDICAL_QA_MODEL", ROOT / "models/medical/qwen3-8b"))
    require(locator / "model.safetensors", "trained locator weights")
    require(locator / "config.json", "locator config")
    require(selector, "selector artifact")
    require(qwen / "config.json", "Qwen3-8B model")
    require(ROOT / "upstream/medical-appointment/dtos.py", "official DTO")
    if sha256(locator / "model.safetensors") != CANDIDATE["trained_weight_sha256"]:
        raise SystemExit("preflight: locator SHA-256 mismatch")
    if sha256(selector) != CANDIDATE["selector_sha256"]:
        raise SystemExit("preflight: selector SHA-256 mismatch")

    from medical_codex.runtime.evidence_refiner import EvidenceRefiner

    EvidenceRefiner(
        locator,
        selector,
        device="cpu",
        max_answer_tokens=int(CANDIDATE["pipeline"]["max_answer_tokens"]),
    )
    if args.structural_only:
        print(json.dumps({"status": "PASS", "mode": "structural-only"}, indent=2))
        return

    hf_home = Path(os.environ.get("HF_HOME", ROOT / "models/hf"))
    torch_home = Path(os.environ.get("TORCH_HOME", ROOT / "models/torch"))
    whisper_revision = CANDIDATE["pretrained_models"]["whisper"]["revision"]
    whisper = (
        hf_home
        / "hub/models--Systran--faster-whisper-large-v3/snapshots"
        / whisper_revision
    )
    aligner = (
        torch_home
        / "hub/checkpoints"
        / CANDIDATE["pretrained_models"]["aligner"]["name"]
    )
    silero_spec = CANDIDATE["pretrained_models"]["silero_vad"]
    silero = torch_home / "hub" / silero_spec["torch_hub_cache_directory"]
    require(whisper / "config.json", "offline Whisper large-v3 model")
    require(aligner, "offline wav2vec2 aligner")
    require(silero / "hubconf.py", "offline Silero VAD torch.hub cache")
    require(
        silero / "src/silero_vad/data/silero_vad.jit",
        "offline Silero VAD weights",
    )
    if sha256(aligner) != CANDIDATE["pretrained_models"]["aligner"]["sha256"]:
        raise SystemExit("preflight: wav2vec2 aligner SHA-256 mismatch")
    silero_digest, silero_files, silero_bytes = tree_digest(silero)
    if (
        silero_digest != silero_spec["tree_sha256"]
        or silero_files != silero_spec["file_count"]
        or silero_bytes != silero_spec["bytes"]
    ):
        raise SystemExit(
            "preflight: Silero VAD cache does not match the pinned tree: "
            f"sha256={silero_digest}, files={silero_files}, bytes={silero_bytes}"
        )

    try:
        import accelerate
    except ImportError as exc:
        raise SystemExit(
            "preflight: accelerate is required by the Qwen device_map loader; "
            "install requirements.txt"
        ) from exc

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("preflight: CUDA is unavailable")
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gib < 30:
        raise SystemExit(f"preflight: GPU has only {total_gib:.1f} GiB; at least 30 GiB required")

    print(
        json.dumps(
            {
                "status": "PASS",
                "candidate": CANDIDATE["candidate"],
                "python": sys.version.split()[0],
                "accelerate": accelerate.__version__,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_gib": round(total_gib, 1),
                "silero_tree_sha256": silero_digest,
                "offline_assets": "PASS",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
