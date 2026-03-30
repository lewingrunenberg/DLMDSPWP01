"""Tests for threshold-based test-point mapping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dlmdspwp01_project.datasets import IdealDataset, TestDataset
from dlmdspwp01_project.exceptions import MappingError
from dlmdspwp01_project.mapping import TestPointMapper
from dlmdspwp01_project.models import SelectedFunctionPair, SelectionSummary


def write_csv(path: Path, lines: list[str]) -> Path:
    """Write a CSV fixture from individual lines."""

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_ideal_csv(path: Path, rows: list[dict[str, float]]) -> Path:
    """Create an ideal-function CSV with all fifty required columns."""

    columns = ["x"] + [f"y{index}" for index in range(1, 51)]
    lines = [",".join(columns)]

    for row in rows:
        values = [str(row.get("x"))]
        for index in range(1, 51):
            values.append(str(row.get(f"y{index}", 0.0)))
        lines.append(",".join(values))

    return write_csv(path, lines)


def build_selection_summary(*pairs: SelectedFunctionPair) -> SelectionSummary:
    """Create a minimal selection summary for mapping tests."""

    return SelectionSummary(selected_pairs=tuple(pairs), sse_table=pd.DataFrame())


def test_mapper_stores_only_successfully_mapped_points(tmp_path: Path) -> None:
    """Only rows satisfying the threshold rule should appear in the mapping output."""

    ideal_csv = build_ideal_csv(
        tmp_path / "ideal.csv",
        [
            {"x": 0.0, "y7": 10.0},
            {"x": 1.0, "y7": 12.0},
        ],
    )
    test_csv = write_csv(
        tmp_path / "test.csv",
        [
            "x,y",
            "0.0,10.3",
            "1.0,13.0",
        ],
    )
    summary = build_selection_summary(
        SelectedFunctionPair(
            training_function_name="y1",
            ideal_function_name="y7",
            ideal_function_no=7,
            sse=0.0,
            max_abs_deviation=0.3,
            mapping_threshold=0.5,
        )
    )

    mapping_summary = TestPointMapper().map_points(
        TestDataset.from_csv(test_csv),
        IdealDataset.from_csv(ideal_csv),
        summary,
    )

    assert mapping_summary.mapped_count == 1
    assert mapping_summary.unmapped_count == 1
    assert mapping_summary.to_dataframe().to_dict(orient="records") == [
        {"x": 0.0, "y": 10.3, "delta_y": pytest.approx(0.3), "ideal_function_no": 7}
    ]


def test_mapper_resolves_multiple_valid_candidates_by_smallest_delta(tmp_path: Path) -> None:
    """If several selected ideals qualify, choose the closest one deterministically."""

    ideal_csv = build_ideal_csv(
        tmp_path / "ideal.csv",
        [
            {"x": 0.0, "y3": 2.2, "y5": 2.1},
        ],
    )
    test_csv = write_csv(
        tmp_path / "test.csv",
        [
            "x,y",
            "0.0,2.0",
        ],
    )
    summary = build_selection_summary(
        SelectedFunctionPair(
            training_function_name="y1",
            ideal_function_name="y3",
            ideal_function_no=3,
            sse=0.0,
            max_abs_deviation=0.2,
            mapping_threshold=0.5,
        ),
        SelectedFunctionPair(
            training_function_name="y2",
            ideal_function_name="y5",
            ideal_function_no=5,
            sse=0.0,
            max_abs_deviation=0.1,
            mapping_threshold=0.5,
        ),
    )

    mapping_summary = TestPointMapper().map_points(
        TestDataset.from_csv(test_csv),
        IdealDataset.from_csv(ideal_csv),
        summary,
    )

    mapped_point = mapping_summary.mapped_points[0]
    assert mapped_point.ideal_function_no == 5
    assert mapped_point.delta_y == pytest.approx(0.1)


def test_mapper_raises_error_for_missing_ideal_x_value(tmp_path: Path) -> None:
    """Exact x-grid lookup should fail fast instead of interpolating."""

    ideal_csv = build_ideal_csv(
        tmp_path / "ideal.csv",
        [
            {"x": 0.0, "y2": 1.0},
        ],
    )
    test_csv = write_csv(
        tmp_path / "test.csv",
        [
            "x,y",
            "1.0,1.0",
        ],
    )
    summary = build_selection_summary(
        SelectedFunctionPair(
            training_function_name="y1",
            ideal_function_name="y2",
            ideal_function_no=2,
            sse=0.0,
            max_abs_deviation=0.0,
            mapping_threshold=0.1,
        )
    )

    with pytest.raises(MappingError):
        TestPointMapper().map_points(
            TestDataset.from_csv(test_csv),
            IdealDataset.from_csv(ideal_csv),
            summary,
        )
