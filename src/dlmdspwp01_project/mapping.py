"""Threshold-based mapping of test points onto the selected ideal functions."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict

from dlmdspwp01_project.config import FLOAT_TOLERANCE
from dlmdspwp01_project.datasets import IdealDataset, TestDataset
from dlmdspwp01_project.exceptions import MappingError
from dlmdspwp01_project.models import MappedTestPoint, MappingSummary, SelectionSummary


class TestPointMapper:
    """Map test points to selected ideal functions under the assignment threshold."""

    __test__ = False

    def __init__(self, tolerance: float = FLOAT_TOLERANCE) -> None:
        self.tolerance = tolerance

    def map_points(
        self,
        test_dataset: TestDataset,
        ideal_dataset: IdealDataset,
        selection_summary: SelectionSummary,
    ) -> MappingSummary:
        """Map test rows to ideal functions and keep only successful mappings."""

        ideal_rows_by_x = self._build_ideal_lookup(ideal_dataset)
        mapped_points = []
        per_function_counter: Counter[int] = Counter()

        for row_number, row in enumerate(test_dataset.dataframe.itertuples(index=False), start=1):
            if row.x not in ideal_rows_by_x:
                raise MappingError(
                    f"Test x-value {row.x} is not present in the ideal function grid."
                )

            ideal_row = ideal_rows_by_x[row.x]
            candidate_mappings = []

            for pair in selection_summary.selected_pairs:
                ideal_y = float(ideal_row[pair.ideal_function_name])
                delta_y = abs(float(row.y) - ideal_y)

                if delta_y <= pair.mapping_threshold + self.tolerance:
                    candidate_mappings.append(
                        MappedTestPoint(
                            test_row_number=row_number,
                            x=float(row.x),
                            y=float(row.y),
                            delta_y=delta_y,
                            ideal_function_no=pair.ideal_function_no,
                            ideal_function_name=pair.ideal_function_name,
                            ideal_y=ideal_y,
                        )
                    )

            if not candidate_mappings:
                continue

            selected_mapping = min(
                candidate_mappings,
                key=lambda point: (round(point.delta_y, 12), point.ideal_function_no),
            )
            mapped_points.append(selected_mapping)
            per_function_counter[selected_mapping.ideal_function_no] += 1

        total_test_points = test_dataset.row_count
        return MappingSummary(
            total_test_points=total_test_points,
            mapped_points=tuple(mapped_points),
            unmapped_count=total_test_points - len(mapped_points),
            mapped_count_by_ideal_function=dict(sorted(per_function_counter.items())),
        )

    def _build_ideal_lookup(self, ideal_dataset: IdealDataset) -> Dict[float, Dict[str, float]]:
        """Create a dictionary-based exact lookup from x-values to ideal rows."""

        return {
            float(row["x"]): row.to_dict()
            for _, row in ideal_dataset.dataframe.iterrows()
        }
