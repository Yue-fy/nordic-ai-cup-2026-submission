#!/usr/bin/env bash
set -euo pipefail

readonly RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RELEASE_ROOT"

export PYTHONPATH="$RELEASE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$RELEASE_ROOT/models/hf}"
export TORCH_HOME="${TORCH_HOME:-$RELEASE_ROOT/models/torch}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export MEDICAL_QA_MODEL="${MEDICAL_QA_MODEL:-$RELEASE_ROOT/models/medical/qwen3-8b}"
export MEDICAL_REFINER_MODEL="${MEDICAL_REFINER_MODEL:-$RELEASE_ROOT/models/medical/active-locator/model}"
export MEDICAL_REFINER_SELECTOR="${MEDICAL_REFINER_SELECTOR:-$RELEASE_ROOT/medical_codex/artifacts/active_selector.json}"
export MEDICAL_REFINER_DEVICE="${MEDICAL_REFINER_DEVICE:-cuda}"
export MEDICAL_REFINER_MAX_ANSWER_TOKENS="${MEDICAL_REFINER_MAX_ANSWER_TOKENS:-$(python -c 'import json; print(json.load(open("candidate.json"))["pipeline"]["max_answer_tokens"])')}"
export MEDICAL_PRELOAD_ASR="${MEDICAL_PRELOAD_ASR:-1}"
export MEDICAL_RELEASE_COMMIT="${MEDICAL_RELEASE_COMMIT:-$(python -c 'import json; print(json.load(open("BUILD.json"))["source_commit"])')}"
export MEDICAL_RELEASE_CANDIDATE="${MEDICAL_RELEASE_CANDIDATE:-$(python -c 'import json; print(json.load(open("candidate.json"))["candidate"])')}"

python preflight.py
exec python -m uvicorn cup.medical.server:app --host "${MEDICAL_HOST:-0.0.0.0}" --port "${MEDICAL_PORT:-9054}" --workers 1
