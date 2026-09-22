"""Project paths and explicit, case-local imports of official protocol code."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'upstream'

def protocol(case):
    name = 'official_' + case.replace('-', '_') + '_dtos'
    if name not in sys.modules:
        path = UPSTREAM / case / 'dtos.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
