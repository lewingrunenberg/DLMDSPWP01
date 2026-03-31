"""Focused Bokeh visualizations for assignment analysis and reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category10
from bokeh.plotting import figure, output_file, save

from dlmdspwp01_project.config import PLOTS_DIR
from dlmdspwp01_project.datasets import IdealDataset, TrainingDataset
from dlmdspwp01_project.models import MappingSummary, SelectionSummary


class VisualizationBuilder:
    """Build a compact set of analytically justified Bokeh plots."""

    def __init__(self, output_directory: str | Path = PLOTS_DIR) -> None:
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def build_all(
        self,
        training_dataset: TrainingDataset,
        ideal_dataset: IdealDataset,
        selection_summary: SelectionSummary,
        mapping_summary: MappingSummary,
    ) -> list[Path]:
        """Generate the standard plot set and return the saved file paths."""

        return [
            self.build_training_vs_selected_plot(
                training_dataset, ideal_dataset, selection_summary
            ),
            self.build_mapped_test_points_plot(ideal_dataset, selection_summary, mapping_summary),
            self.build_accepted_deviation_by_ideal_plot(
                selection_summary, mapping_summary
            ),
        ]

    def build_training_vs_selected_plot(
        self,
        training_dataset: TrainingDataset,
        ideal_dataset: IdealDataset,
        selection_summary: SelectionSummary,
        file_name: str = "training_vs_selected_ideal.html",
    ) -> Path:
        """Overlay each training function with its selected ideal function."""

        figures = []
        x_values = training_dataset.dataframe["x"]

        for pair in selection_summary.selected_pairs:
            plot = figure(
                title=f"{pair.training_function_name} vs {pair.ideal_function_name}",
                x_axis_label="x",
                y_axis_label="y",
                width=500,
                height=300,
            )
            plot.line(
                x_values,
                training_dataset.dataframe[pair.training_function_name],
                line_width=2,
                color="#1f77b4",
                legend_label=pair.training_function_name,
            )
            plot.line(
                x_values,
                ideal_dataset.dataframe[pair.ideal_function_name],
                line_width=2,
                color="#ff7f0e",
                legend_label=pair.ideal_function_name,
            )
            plot.legend.location = "top_left"
            figures.append(plot)

        output_path = self.output_directory / file_name
        output_file(output_path)
        save(gridplot([figures[:2], figures[2:]]))
        return output_path

    def build_mapped_test_points_plot(
        self,
        ideal_dataset: IdealDataset,
        selection_summary: SelectionSummary,
        mapping_summary: MappingSummary,
        file_name: str = "mapped_test_points.html",
    ) -> Path:
        """Show mapped test points together with the selected ideal functions."""

        plot = figure(
            title="Mapped test points against selected ideal functions",
            x_axis_label="x",
            y_axis_label="y",
            width=900,
            height=500,
        )

        palette = Category10[10]
        for index, pair in enumerate(selection_summary.selected_pairs):
            plot.line(
                ideal_dataset.dataframe["x"],
                ideal_dataset.dataframe[pair.ideal_function_name],
                line_width=2,
                color=palette[index],
                legend_label=f"Ideal {pair.ideal_function_no}",
            )

        if mapping_summary.mapped_points:
            source = ColumnDataSource(
                {
                    "x": [point.x for point in mapping_summary.mapped_points],
                    "y": [point.y for point in mapping_summary.mapped_points],
                    "delta_y": [point.delta_y for point in mapping_summary.mapped_points],
                    "ideal_function_no": [
                        point.ideal_function_no for point in mapping_summary.mapped_points
                    ],
                }
            )
            renderer = plot.scatter(
                "x",
                "y",
                size=7,
                color="black",
                alpha=0.75,
                source=source,
                legend_label="Mapped test points",
            )
            plot.add_tools(
                HoverTool(
                    renderers=[renderer],
                    tooltips=[
                        ("x", "@x"),
                        ("y", "@y"),
                        ("delta_y", "@delta_y"),
                        ("ideal_function_no", "@ideal_function_no"),
                    ],
                )
            )

        plot.legend.location = "top_left"
        output_path = self.output_directory / file_name
        output_file(output_path)
        save(plot)
        return output_path

    def build_accepted_deviation_by_ideal_plot(
        self,
        selection_summary: SelectionSummary,
        mapping_summary: MappingSummary,
        file_name: str = "accepted_deviation_by_ideal_function.html",
    ) -> Path:
        threshold_by_ideal = {
            pair.ideal_function_no: pair.mapping_threshold
            for pair in selection_summary.selected_pairs
        }

        points = mapping_summary.mapped_points
        if not points:
            plot = figure(
                title="Distribution of accepted absolute deviations by selected ideal function",
                x_axis_label="Ideal function number",
                y_axis_label="Absolute deviation",
                width=700,
                height=400,
                y_range=(0, 1),
            )
            output_path = self.output_directory / file_name
            output_file(output_path)
            save(plot)
            return output_path

        df = pd.DataFrame(
            {
                "ideal_function_no": [p.ideal_function_no for p in points],
                "abs_deviation": [p.delta_y for p in points],
            }
        )
        grouped = df.groupby("ideal_function_no", sort=True)["abs_deviation"]
        stats = grouped.agg(
            lower="min",
            q1=lambda s: float(s.quantile(0.25)),
            med="median",
            q3=lambda s: float(s.quantile(0.75)),
            upper="max",
        ).reset_index()
        n = len(stats)
        x = list(range(n))
        stats["x"] = x
        stats["threshold"] = stats["ideal_function_no"].map(threshold_by_ideal).astype(float)

        ymax_data = float(stats[["upper", "threshold"]].max().max())
        ymax = ymax_data * 1.05 if ymax_data > 0 else 1.0

        plot = figure(
            title="Distribution of accepted absolute deviations by selected ideal function",
            x_axis_label="Ideal function number",
            y_axis_label="Absolute deviation",
            x_range=(-0.5, n - 0.5),
            y_range=(0, ymax),
            width=700,
            height=400,
        )
        plot.xaxis.ticker = x
        ideal_labels = stats["ideal_function_no"].astype(int).tolist()
        plot.xaxis.major_label_overrides = {i: str(ideal_labels[i]) for i in range(n)}

        plot.segment(
            x0=x,
            y0=stats["lower"],
            x1=x,
            y1=stats["q1"],
            line_color="black",
            line_width=1,
        )
        plot.segment(
            x0=x,
            y0=stats["q3"],
            x1=x,
            y1=stats["upper"],
            line_color="black",
            line_width=1,
        )
        plot.vbar(
            x=x,
            width=0.45,
            bottom=stats["q1"],
            top=stats["q3"],
            fill_color="white",
            line_color="black",
        )
        xm = 0.22
        plot.segment(
            x0=[i - xm for i in x],
            y0=stats["med"],
            x1=[i + xm for i in x],
            y1=stats["med"],
            line_color="black",
            line_width=2,
        )
        plot.segment(
            x0=[i - xm for i in x],
            y0=stats["threshold"],
            x1=[i + xm for i in x],
            y1=stats["threshold"],
            line_color="#000000",
            line_width=1.5,
            line_dash="dashed",
            legend_label="Mapping threshold",
        )
        plot.legend.location = "top_right"

        output_path = self.output_directory / file_name
        output_file(output_path)
        save(plot)
        return output_path
