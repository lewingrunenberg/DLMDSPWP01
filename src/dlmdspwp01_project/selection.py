"""Least-squares selection of ideal functions for the training data."""

from __future__ import annotations

import math

import pandas as pd

from dlmdspwp01_project.config import FLOAT_TOLERANCE
from dlmdspwp01_project.datasets import IdealDataset, TrainingDataset, validate_matching_x_grid
from dlmdspwp01_project.models import SelectedFunctionPair, SelectionSummary


def function_number(function_name: str) -> int:
    """Extract the numeric suffix from a function column such as ``y13``."""

    return int(function_name.removeprefix("y"))


class IdealFunctionSelector:
    """Select the best ideal function for each training function via SSE."""

    def __init__(self, tolerance: float = FLOAT_TOLERANCE) -> None:
        self.tolerance = tolerance

    def select(
        self, training_dataset: TrainingDataset, ideal_dataset: IdealDataset
    ) -> SelectionSummary:
        """Return independent SSE-based selections for the four training functions."""

        validate_matching_x_grid(training_dataset, ideal_dataset)

        sse_table = self._build_sse_table(training_dataset, ideal_dataset)
        selected_pairs = []

        for training_function_name in training_dataset.function_columns:
            sse_series = sse_table.loc[training_function_name]
            ideal_function_name = self._select_best_ideal_function(sse_series)

            absolute_differences = (
                training_dataset.dataframe[training_function_name]
                - ideal_dataset.dataframe[ideal_function_name]
            ).abs()
            max_abs_deviation = float(absolute_differences.max())

            selected_pairs.append(
                SelectedFunctionPair(
                    training_function_name=training_function_name,
                    ideal_function_name=ideal_function_name,
                    ideal_function_no=function_number(ideal_function_name),
                    sse=float(sse_series[ideal_function_name]),
                    max_abs_deviation=max_abs_deviation,
                    mapping_threshold=max_abs_deviation * math.sqrt(2.0),
                )
            )

        return SelectionSummary(
            selected_pairs=tuple(selected_pairs),
            sse_table=sse_table,
        )

    def _build_sse_table(
        self, training_dataset: TrainingDataset, ideal_dataset: IdealDataset
    ) -> pd.DataFrame:
        """Compute the complete SSE table for all train/ideal combinations."""

        sse_rows = {}
        for training_function_name in training_dataset.function_columns:
            sse_rows[training_function_name] = {}
            training_series = training_dataset.dataframe[training_function_name]

            for ideal_function_name in ideal_dataset.function_columns:
                residuals = training_series - ideal_dataset.dataframe[ideal_function_name]
                sse_rows[training_function_name][ideal_function_name] = float(
                    (residuals.pow(2)).sum()
                )

        return pd.DataFrame.from_dict(sse_rows, orient="index").sort_index(axis=1)

    def _select_best_ideal_function(self, sse_series: pd.Series) -> str:
        """Choose the smallest SSE, breaking ties by the smaller function number."""

        minimum_sse = float(sse_series.min())
        candidate_names = [
            function_name
            for function_name, sse_value in sse_series.items()
            if math.isclose(float(sse_value), minimum_sse, abs_tol=self.tolerance)
        ]
        return min(candidate_names, key=function_number)
