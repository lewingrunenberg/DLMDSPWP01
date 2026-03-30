"""Domain models used by the analytical services and artifact generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class SelectedFunctionPair:
    """Represents the chosen ideal function for one training function."""

    training_function_name: str
    ideal_function_name: str
    ideal_function_no: int
    sse: float
    max_abs_deviation: float
    mapping_threshold: float


@dataclass(frozen=True)
class SelectionSummary:
    """Contains the selected function pairs and the full SSE comparison table."""

    selected_pairs: Tuple[SelectedFunctionPair, ...]
    sse_table: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        """Return the selected function pairs as a dataframe."""

        return pd.DataFrame(
            [
                {
                    "training_function_name": pair.training_function_name,
                    "ideal_function_name": pair.ideal_function_name,
                    "ideal_function_no": pair.ideal_function_no,
                    "sse": pair.sse,
                    "max_abs_deviation": pair.max_abs_deviation,
                    "mapping_threshold": pair.mapping_threshold,
                }
                for pair in self.selected_pairs
            ]
        )


@dataclass(frozen=True)
class MappedTestPoint:
    """Represents one successfully mapped test point."""

    test_row_number: int
    x: float
    y: float
    delta_y: float
    ideal_function_no: int
    ideal_function_name: str
    ideal_y: float


@dataclass(frozen=True)
class MappingSummary:
    """Contains successfully mapped points and aggregate mapping counts."""

    total_test_points: int
    mapped_points: Tuple[MappedTestPoint, ...]
    unmapped_count: int
    mapped_count_by_ideal_function: Dict[int, int]

    @property
    def mapped_count(self) -> int:
        """Return the number of successfully mapped points."""

        return len(self.mapped_points)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the mapped test points in assignment-aligned table form."""

        return pd.DataFrame(
            [
                {
                    "x": point.x,
                    "y": point.y,
                    "delta_y": point.delta_y,
                    "ideal_function_no": point.ideal_function_no,
                }
                for point in self.mapped_points
            ]
        )

    def counts_dataframe(self) -> pd.DataFrame:
        """Return a compact per-function mapping count table."""

        return pd.DataFrame(
            [
                {
                    "ideal_function_no": ideal_function_no,
                    "mapped_count": mapped_count,
                }
                for ideal_function_no, mapped_count in sorted(
                    self.mapped_count_by_ideal_function.items()
                )
            ],
            columns=["ideal_function_no", "mapped_count"],
        )


@dataclass(frozen=True)
class RunSummary:
    """Summarizes a deterministic pipeline run and its generated artifacts."""

    total_test_points: int
    mapped_count: int
    unmapped_count: int
    mapped_count_by_ideal_function: Dict[int, int]
    database_path: Path
    plot_paths: Tuple[Path, ...]
    report_paths: Tuple[Path, ...]

    def to_dict(self) -> dict:
        """Return the run summary in JSON-serializable form."""

        return {
            "total_test_points": self.total_test_points,
            "mapped_count": self.mapped_count,
            "unmapped_count": self.unmapped_count,
            "mapped_count_by_ideal_function": self.mapped_count_by_ideal_function,
            "database_path": str(self.database_path),
            "plot_paths": [str(path) for path in self.plot_paths],
            "report_paths": [str(path) for path in self.report_paths],
        }
