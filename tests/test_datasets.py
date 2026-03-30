"""Tests for dataset loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlmdspwp01_project.config import DATASET_DIR
from dlmdspwp01_project.datasets import (
    IdealDataset,
    TestDataset,
    TrainingDataset,
    validate_matching_x_grid,
)
from dlmdspwp01_project.exceptions import CsvSchemaError, DataValidationError, XGridValidationError


def write_csv(path: Path, content: str) -> Path:
    """Write a small CSV fixture and return its path."""

    path.write_text(content, encoding="utf-8")
    return path


def test_assignment_datasets_load_successfully() -> None:
    """The provided assignment datasets should load and share the same train/ideal x-grid."""

    training = TrainingDataset.from_csv(DATASET_DIR / "train.csv")
    ideal = IdealDataset.from_csv(DATASET_DIR / "ideal.csv")
    test = TestDataset.from_csv(DATASET_DIR / "test.csv")

    validate_matching_x_grid(training, ideal)

    assert training.row_count == 400
    assert ideal.row_count == 400
    assert test.row_count == 100


def test_training_schema_mismatch_raises_error(tmp_path: Path) -> None:
    """Unexpected columns should fail fast."""

    csv_path = write_csv(
        tmp_path / "train.csv",
        "x,y1,y2,y3\n0.0,1.0,2.0,3.0\n",
    )

    with pytest.raises(CsvSchemaError):
        TrainingDataset.from_csv(csv_path)


def test_non_numeric_values_raise_validation_error(tmp_path: Path) -> None:
    """Non-numeric content should not be silently coerced."""

    csv_path = write_csv(
        tmp_path / "train.csv",
        "x,y1,y2,y3,y4\n0.0,1.0,not-a-number,3.0,4.0\n",
    )

    with pytest.raises(DataValidationError):
        TrainingDataset.from_csv(csv_path)


def test_duplicate_training_x_values_raise_error(tmp_path: Path) -> None:
    """Training and ideal datasets require unique x-values for valid SSE comparison."""

    csv_path = write_csv(
        tmp_path / "train.csv",
        "x,y1,y2,y3,y4\n0.0,1.0,2.0,3.0,4.0\n0.0,1.5,2.5,3.5,4.5\n",
    )

    with pytest.raises(XGridValidationError):
        TrainingDataset.from_csv(csv_path)


def test_test_dataset_allows_duplicate_and_unsorted_x(tmp_path: Path) -> None:
    """Test rows may repeat x-values and do not need to be ordered."""

    csv_path = write_csv(
        tmp_path / "test.csv",
        "x,y\n1.0,2.0\n0.0,1.0\n1.0,2.5\n",
    )

    dataset = TestDataset.from_csv(csv_path)

    assert dataset.row_count == 3


def test_validate_matching_x_grid_rejects_mismatch(tmp_path: Path) -> None:
    """Training and ideal datasets must use the same ordered x-grid."""

    training_csv = write_csv(
        tmp_path / "train.csv",
        "x,y1,y2,y3,y4\n0.0,1.0,2.0,3.0,4.0\n1.0,1.5,2.5,3.5,4.5\n",
    )
    ideal_csv = write_csv(
        tmp_path / "ideal.csv",
        "x,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20,"
        "y21,y22,y23,y24,y25,y26,y27,y28,y29,y30,y31,y32,y33,y34,y35,y36,y37,y38,y39,"
        "y40,y41,y42,y43,y44,y45,y46,y47,y48,y49,y50\n"
        "0.0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\n"
        "2.0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2\n",
    )

    training = TrainingDataset.from_csv(training_csv)
    ideal = IdealDataset.from_csv(ideal_csv)

    with pytest.raises(XGridValidationError):
        validate_matching_x_grid(training, ideal)
