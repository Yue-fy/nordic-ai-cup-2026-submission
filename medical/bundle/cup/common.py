"""Project paths and explicit, case-local imports of official protocol code."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'upstream'


def source_commit() -> str:
    """Return a build identifier even when the release is outside a Git checkout."""
    override = os.environ.get("MEDICAL_RELEASE_COMMIT")
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"

def protocol(case):
    name = 'official_' + case.replace('-', '_') + '_dtos'
    if name not in sys.modules:
        path = UPSTREAM / case / 'dtos.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
