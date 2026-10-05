"""BentoML import entry point for the src-layout package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dice_viewer.service import DiceService  # noqa: E402,F401
