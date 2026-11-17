"""Explicit V1 entry point; the historical demo service remains separate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dice_viewer.recognition.service import DiceRecognitionService  # noqa: E402,F401
