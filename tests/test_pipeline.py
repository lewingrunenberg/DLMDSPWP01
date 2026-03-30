"""Concise integration test for the end-to-end assignment pipeline."""

from __future__ import annotations

from pathlib import Path

from dlmdspwp01_project.pipeline import AssignmentPipeline


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


def test_pipeline_creates_database_reports_and_plots(tmp_path: Path) -> None:
    """The pipeline should create deterministic database, report, and plot artifacts."""

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
            "1.0,50.0",
        ],
    )

    plots_dir = tmp_path / "plots"
    reports_dir = tmp_path / "reports"
    database_path = tmp_path / "assignment.db"

    summary = AssignmentPipeline(
        train_path=training_csv,
        ideal_path=ideal_csv,
        test_path=test_csv,
        database_path=database_path,
        plots_directory=plots_dir,
        reports_directory=reports_dir,
    ).run()

    assert summary.database_path.exists()
    assert summary.mapped_count == 1
    assert summary.unmapped_count == 1
    assert len(summary.plot_paths) == 3
    assert all(path.exists() for path in summary.plot_paths)
    assert (reports_dir / "selection_summary.csv").exists()
    assert (reports_dir / "mapped_test_points.csv").exists()
    assert (reports_dir / "mapping_counts_by_function.csv").exists()
    assert (reports_dir / "mapping_summary.csv").exists()
    assert (reports_dir / "run_summary.json").exists()
