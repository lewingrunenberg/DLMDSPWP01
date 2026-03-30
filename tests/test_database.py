"""Tests for assignment-aligned SQLite persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from dlmdspwp01_project.database import DatabaseManager
from dlmdspwp01_project.datasets import IdealDataset, TestDataset, TrainingDataset
from dlmdspwp01_project.mapping import TestPointMapper
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


def test_database_manager_writes_required_tables(tmp_path: Path) -> None:
    """The SQLite database should contain the required assignment tables and rows."""

    training_csv = write_csv(
        tmp_path / "train.csv",
        [
            "x,y1,y2,y3,y4",
            "0.0,1.0,4.0,7.0,10.0",
            "1.0,2.0,5.0,8.0,11.0",
        ],
    )
    ideal_csv = build_ideal_csv(
        tmp_path / "ideal.csv",
        [
            {"x": 0.0, "y1": 1.0, "y2": 4.0, "y3": 7.0, "y4": 10.0},
            {"x": 1.0, "y1": 2.0, "y2": 5.0, "y3": 8.0, "y4": 11.0},
        ],
    )
    test_csv = write_csv(
        tmp_path / "test.csv",
        [
            "x,y",
            "0.0,1.0",
            "1.0,20.0",
        ],
    )

    training_dataset = TrainingDataset.from_csv(training_csv)
    ideal_dataset = IdealDataset.from_csv(ideal_csv)
    selection_summary = IdealFunctionSelector().select(training_dataset, ideal_dataset)
    mapping_summary = TestPointMapper().map_points(
        TestDataset.from_csv(test_csv),
        ideal_dataset,
        selection_summary,
    )

    database_path = tmp_path / "assignment.db"
    manager = DatabaseManager(database_path)
    manager.persist_all(training_dataset, ideal_dataset, selection_summary, mapping_summary)

    connection = sqlite3.connect(database_path)
    try:
        table_names = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"training_data", "ideal_functions", "test_mapping", "selected_function_pairs"} <= table_names

        training_columns = [
            row[1] for row in connection.execute("PRAGMA table_info(training_data)")
        ]
        ideal_columns = [
            row[1] for row in connection.execute("PRAGMA table_info(ideal_functions)")
        ]
        mapping_columns = [
            row[1] for row in connection.execute("PRAGMA table_info(test_mapping)")
        ]

        assert training_columns == ["x", "y1", "y2", "y3", "y4"]
        assert ideal_columns == ["x"] + [f"y{index}" for index in range(1, 51)]
        assert mapping_columns == ["x", "y", "delta_y", "ideal_function_no"]

        training_count = connection.execute("SELECT COUNT(*) FROM training_data").fetchone()[0]
        ideal_count = connection.execute("SELECT COUNT(*) FROM ideal_functions").fetchone()[0]
        mapping_count = connection.execute("SELECT COUNT(*) FROM test_mapping").fetchone()[0]
        pair_count = connection.execute("SELECT COUNT(*) FROM selected_function_pairs").fetchone()[0]

        assert training_count == 2
        assert ideal_count == 2
        assert mapping_count == 1
        assert pair_count == 4
    finally:
        connection.close()
