"""End-to-end orchestration for the assignment workflow and artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dlmdspwp01_project.config import (
    DATABASE_PATH,
    DATASET_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
)
from dlmdspwp01_project.database import DatabaseManager
from dlmdspwp01_project.datasets import IdealDataset, TestDataset, TrainingDataset
from dlmdspwp01_project.models import RunSummary
from dlmdspwp01_project.mapping import TestPointMapper
from dlmdspwp01_project.selection import IdealFunctionSelector
from dlmdspwp01_project.visualization import VisualizationBuilder


class AssignmentPipeline:
    """Run the full assignment workflow and write deterministic artifacts."""

    def __init__(
        self,
        train_path: str | Path = DATASET_DIR / "train.csv",
        ideal_path: str | Path = DATASET_DIR / "ideal.csv",
        test_path: str | Path = DATASET_DIR / "test.csv",
        database_path: str | Path = DATABASE_PATH,
        plots_directory: str | Path = PLOTS_DIR,
        reports_directory: str | Path = REPORTS_DIR,
    ) -> None:
        self.train_path = Path(train_path)
        self.ideal_path = Path(ideal_path)
        self.test_path = Path(test_path)
        self.database_path = Path(database_path)
        self.plots_directory = Path(plots_directory)
        self.reports_directory = Path(reports_directory)

    def run(self) -> RunSummary:
        """Execute the full analytical workflow and return a summary."""

        training_dataset = TrainingDataset.from_csv(self.train_path)
        ideal_dataset = IdealDataset.from_csv(self.ideal_path)
        test_dataset = TestDataset.from_csv(self.test_path)

        selection_summary = IdealFunctionSelector().select(training_dataset, ideal_dataset)
        mapping_summary = TestPointMapper().map_points(
            test_dataset,
            ideal_dataset,
            selection_summary,
        )

        DatabaseManager(self.database_path).persist_all(
            training_dataset,
            ideal_dataset,
            selection_summary,
            mapping_summary,
        )
        plot_paths = tuple(
            VisualizationBuilder(self.plots_directory).build_all(
                training_dataset,
                ideal_dataset,
                selection_summary,
                mapping_summary,
            )
        )
        report_paths = self._write_reports(selection_summary, mapping_summary, plot_paths)

        return RunSummary(
            total_test_points=mapping_summary.total_test_points,
            mapped_count=mapping_summary.mapped_count,
            unmapped_count=mapping_summary.unmapped_count,
            mapped_count_by_ideal_function=mapping_summary.mapped_count_by_ideal_function,
            database_path=self.database_path,
            plot_paths=plot_paths,
            report_paths=report_paths,
        )

    def _write_reports(
        self,
        selection_summary,
        mapping_summary,
        plot_paths: tuple[Path, ...],
    ) -> tuple[Path, ...]:
        """Write deterministic CSV and JSON report artifacts."""

        self.reports_directory.mkdir(parents=True, exist_ok=True)

        selection_path = self.reports_directory / "selection_summary.csv"
        selection_summary.to_dataframe().to_csv(selection_path, index=False)

        mapped_points_path = self.reports_directory / "mapped_test_points.csv"
        mapping_summary.to_dataframe().to_csv(mapped_points_path, index=False)

        mapping_counts_path = self.reports_directory / "mapping_counts_by_function.csv"
        mapping_summary.counts_dataframe().to_csv(mapping_counts_path, index=False)

        mapping_summary_path = self.reports_directory / "mapping_summary.csv"
        pd.DataFrame(
            [
                {
                    "total_test_points": mapping_summary.total_test_points,
                    "mapped_count": mapping_summary.mapped_count,
                    "unmapped_count": mapping_summary.unmapped_count,
                }
            ]
        ).to_csv(mapping_summary_path, index=False)

        run_summary = RunSummary(
            total_test_points=mapping_summary.total_test_points,
            mapped_count=mapping_summary.mapped_count,
            unmapped_count=mapping_summary.unmapped_count,
            mapped_count_by_ideal_function=mapping_summary.mapped_count_by_ideal_function,
            database_path=self.database_path,
            plot_paths=plot_paths,
            report_paths=(
                selection_path,
                mapped_points_path,
                mapping_counts_path,
                mapping_summary_path,
            ),
        )
        run_summary_path = self.reports_directory / "run_summary.json"
        run_summary_path.write_text(
            json.dumps(run_summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return (
            selection_path,
            mapped_points_path,
            mapping_counts_path,
            mapping_summary_path,
            run_summary_path,
        )
