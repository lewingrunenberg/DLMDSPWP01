"""Project-wide configuration constants and assignment-specific schemas."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DB_DIR = ARTIFACTS_DIR / "db"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

DATABASE_PATH = DB_DIR / "assignment.db"

TRAIN_COLUMNS = ("x", "y1", "y2", "y3", "y4")
IDEAL_COLUMNS = ("x",) + tuple(f"y{index}" for index in range(1, 51))
TEST_COLUMNS = ("x", "y")

TRAIN_FUNCTION_COLUMNS = TRAIN_COLUMNS[1:]
IDEAL_FUNCTION_COLUMNS = IDEAL_COLUMNS[1:]

FLOAT_TOLERANCE = 1e-12
