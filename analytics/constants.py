"""constants.py — Shared paths for the analytics module."""
from pathlib import Path

MODULE_DIR  = Path(__file__).parent
TITANIC_CSV = MODULE_DIR / "titanic.csv"
OUTPUTS_DIR = MODULE_DIR / "outputs"
CHARTS_DIR  = OUTPUTS_DIR / "charts"
MODELS_DIR  = MODULE_DIR / "models"
