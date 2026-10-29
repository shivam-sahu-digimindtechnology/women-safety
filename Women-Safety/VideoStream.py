"""
Forwarding shim for VideoStream module.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
for sub in ["detection", "tracking", "incident_detection", "alerts", "utilities"]:
    p = str(SRC_DIR / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from VideoStream import VideoStream

__all__ = ["VideoStream"]
