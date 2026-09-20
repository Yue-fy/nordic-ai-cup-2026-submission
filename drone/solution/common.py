"""Workspace paths and authoritative protocol imports."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "official/drone-flyby"
sys.path.insert(0, str(OFFICIAL))
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".cache/ultralytics")
(ROOT / ".cache/ultralytics/Ultralytics").mkdir(parents=True, exist_ok=True)

def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()
