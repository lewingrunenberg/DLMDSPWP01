"""Command-line entrypoint for the assignment pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from dlmdspwp01_project.config import DATABASE_PATH, DATASET_DIR, PLOTS_DIR, REPORTS_DIR
from dlmdspwp01_project.pipeline import AssignmentPipeline


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Run the DLMDSPWP01 dataset assignment pipeline."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DATASET_DIR / "train.csv",
        help="Path to train.csv",
    )
    parser.add_argument(
        "--ideal",
        type=Path,
        default=DATASET_DIR / "ideal.csv",
        help="Path to ideal.csv",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=DATASET_DIR / "test.csv",
        help="Path to test.csv",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=PLOTS_DIR,
        help="Directory for generated Bokeh plots.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory for generated CSV and JSON reports.",
    )
    return parser


def main() -> int:
    """Run the pipeline and print a concise execution summary."""

    arguments = build_argument_parser().parse_args()
    summary = AssignmentPipeline(
        train_path=arguments.train,
        ideal_path=arguments.ideal,
        test_path=arguments.test,
        database_path=arguments.database,
        plots_directory=arguments.plots_dir,
        reports_directory=arguments.reports_dir,
    ).run()

    print(f"Database: {summary.database_path}")
    print(f"Mapped test points: {summary.mapped_count}/{summary.total_test_points}")
    print(f"Unmapped test points: {summary.unmapped_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
