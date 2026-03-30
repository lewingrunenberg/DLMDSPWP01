"""Tests for SSE-based ideal function selection."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from dlmdspwp01_project.datasets import IdealDataset, TrainingDataset
from dlmdspwp01_project.selection import IdealFunctionSelector


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


def test_selector_chooses_lowest_sse_per_training_function(tmp_path: Path) -> None:
    """Each training function should be matched independently to its SSE minimum."""

    training_csv = write_csv(
        tmp_path / "train.csv",
        [
            "x,y1,y2,y3,y4",
            "0.0,0.0,5.0,2.0,-1.0",
            "1.0,1.0,5.0,2.0,-2.0",
        ],
    )
    ideal_csv = build_ideal_csv(
        tmp_path / "ideal.csv",
        [
            {"x": 0.0, "y1": 2.0, "y2": 0.0, "y4": -1.0, "y5": 5.0},
            {"x": 1.0, "y1": 2.0, "y2": 1.0, "y4": -2.0, "y5": 5.0},
        ],
    )

    selector = IdealFunctionSelector()
    summary = selector.select(
        TrainingDataset.from_csv(training_csv),
        IdealDataset.from_csv(ideal_csv),
    )

    selected = {
        pair.training_function_name: pair.ideal_function_name
        for pair in summary.selected_pairs
    }
    assert selected == {"y1": "y2", "y2": "y5", "y3": "y1", "y4": "y4"}


def test_selector_uses_smallest_function_number_as_tie_break(tmp_path: Path) -> None:
    """Equal SSE candidates should be resolved deterministically."""

    training_csv = write_csv(
        tmp_path / "train.csv",
        [
            "x,y1,y2,y3,y4",
            "0.0,0.0,1.0,2.0,3.0",
            "1.0,0.0,1.0,2.0,3.0",
        ],
    )
    ideal_csv = build_ideal_csv(
        tmp_path / "ideal.csv",
        [
            {"x": 0.0, "y1": 0.0, "y2": 0.0, "y3": 1.0, "y4": 2.0, "y5": 3.0},
            {"x": 1.0, "y1": 0.0, "y2": 0.0, "y3": 1.0, "y4": 2.0, "y5": 3.0},
        ],
    )

    summary = IdealFunctionSelector().select(
        TrainingDataset.from_csv(training_csv),
        IdealDataset.from_csv(ideal_csv),
    )

    y1_pair = next(pair for pair in summary.selected_pairs if pair.training_function_name == "y1")
    assert y1_pair.ideal_function_name == "y1"


def test_threshold_is_derived_from_max_abs_deviation(tmp_path: Path) -> None:
    """The mapping threshold must equal max training deviation times sqrt(2)."""

    training_csv = write_csv(
        tmp_path / "train.csv",
        [
            "x,y1,y2,y3,y4",
            "0.0,1.0,0.0,0.0,0.0",
            "1.0,2.0,0.0,0.0,0.0",
        ],
    )
    ideal_csv = build_ideal_csv(
        tmp_path / "ideal.csv",
        [
            {"x": 0.0, "y3": 1.2},
            {"x": 1.0, "y3": 1.7},
        ],
    )

    summary = IdealFunctionSelector().select(
        TrainingDataset.from_csv(training_csv),
        IdealDataset.from_csv(ideal_csv),
    )

    y1_pair = next(pair for pair in summary.selected_pairs if pair.training_function_name == "y1")
    assert y1_pair.ideal_function_name == "y3"
    assert y1_pair.max_abs_deviation == pytest.approx(0.3)
    assert y1_pair.mapping_threshold == pytest.approx(0.3 * math.sqrt(2.0))
