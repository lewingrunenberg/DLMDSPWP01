"""Dataset loading and validation utilities for the assignment CSV files."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd

from dlmdspwp01_project.config import IDEAL_COLUMNS, TEST_COLUMNS, TRAIN_COLUMNS
from dlmdspwp01_project.exceptions import CsvSchemaError, DataValidationError, XGridValidationError


@dataclass
class TabularDataset(ABC):
    """Base class for validated CSV-backed datasets."""

    dataframe: pd.DataFrame
    source_path: Path

    dataset_name: ClassVar[str] = "dataset"
    required_columns: ClassVar[tuple[str, ...]] = tuple()
    allow_duplicate_x: ClassVar[bool] = False
    require_strictly_increasing_x: ClassVar[bool] = True

    @classmethod
    def from_csv(cls, path: str | Path) -> "TabularDataset":
        """Load a dataset from CSV and validate its schema and numeric content."""

        source_path = Path(path)
        try:
            dataframe = pd.read_csv(source_path)
        except FileNotFoundError as error:
            raise DataValidationError(
                f"{cls.dataset_name} file does not exist: {source_path}"
            ) from error
        except Exception as error:  # pragma: no cover - pandas exception types vary.
            raise DataValidationError(
                f"Failed to read {cls.dataset_name} file: {source_path}"
            ) from error

        validated = cls._validate_frame(dataframe)
        return cls(dataframe=validated, source_path=source_path)

    @classmethod
    def _validate_frame(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Validate the raw dataframe and return a canonical numeric copy."""

        cls._validate_schema(dataframe)
        validated = dataframe.loc[:, list(cls.required_columns)].copy()

        try:
            validated = validated.apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as error:
            raise DataValidationError(
                f"{cls.dataset_name} contains non-numeric values."
            ) from error

        if validated.isnull().any().any():
            raise DataValidationError(f"{cls.dataset_name} contains missing values.")

        cls._validate_x_values(validated)
        return validated

    @classmethod
    def _validate_schema(cls, dataframe: pd.DataFrame) -> None:
        actual_columns = tuple(dataframe.columns)
        if actual_columns != cls.required_columns:
            raise CsvSchemaError(
                f"{cls.dataset_name} schema mismatch. "
                f"Expected columns {cls.required_columns} but received {actual_columns}."
            )

    @classmethod
    def _validate_x_values(cls, dataframe: pd.DataFrame) -> None:
        x_values = dataframe["x"]

        if not cls.allow_duplicate_x and x_values.duplicated().any():
            raise XGridValidationError(f"{cls.dataset_name} contains duplicate x-values.")

        if cls.require_strictly_increasing_x and not x_values.is_monotonic_increasing:
            raise XGridValidationError(
                f"{cls.dataset_name} x-values must be ordered increasingly."
            )

    @property
    def x_values(self) -> pd.Series:
        """Return the validated x-axis values."""

        return self.dataframe["x"].copy()

    @property
    def row_count(self) -> int:
        """Return the number of validated rows."""

        return len(self.dataframe.index)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the validated dataframe."""

        return self.dataframe.copy()


@dataclass
class TrainingDataset(TabularDataset):
    """Validated representation of the assignment training dataset."""

    dataset_name: ClassVar[str] = "training dataset"
    required_columns: ClassVar[tuple[str, ...]] = TRAIN_COLUMNS

    @property
    def function_columns(self) -> tuple[str, ...]:
        """Return the names of the four training functions."""

        return self.required_columns[1:]


@dataclass
class IdealDataset(TabularDataset):
    """Validated representation of the assignment ideal function dataset."""

    dataset_name: ClassVar[str] = "ideal dataset"
    required_columns: ClassVar[tuple[str, ...]] = IDEAL_COLUMNS

    @property
    def function_columns(self) -> tuple[str, ...]:
        """Return the names of the fifty ideal functions."""

        return self.required_columns[1:]


@dataclass
class TestDataset(TabularDataset):
    """Validated representation of the assignment test dataset."""

    __test__ = False
    dataset_name: ClassVar[str] = "test dataset"
    required_columns: ClassVar[tuple[str, ...]] = TEST_COLUMNS
    allow_duplicate_x: ClassVar[bool] = True
    require_strictly_increasing_x: ClassVar[bool] = False


def validate_matching_x_grid(
    training_dataset: TrainingDataset, ideal_dataset: IdealDataset
) -> None:
    """Ensure the training and ideal datasets share the same x-grid."""

    train_x = training_dataset.dataframe["x"].reset_index(drop=True)
    ideal_x = ideal_dataset.dataframe["x"].reset_index(drop=True)

    if len(train_x) != len(ideal_x):
        raise XGridValidationError(
            "Training and ideal datasets must contain the same number of x-values."
        )

    if not train_x.equals(ideal_x):
        raise XGridValidationError(
            "Training and ideal datasets must share the same ordered x-grid."
        )
