"""Focused Bokeh visualizations for assignment analysis and reporting."""

from __future__ import annotations

from pathlib import Path

from bokeh.layouts import column, gridplot
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
            self.build_mapping_distribution_plot(mapping_summary),
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

    def build_mapping_distribution_plot(
        self,
        mapping_summary: MappingSummary,
        file_name: str = "mapping_distribution.html",
    ) -> Path:
        """Create a compact bar chart of mapped-point counts per ideal function."""

        count_table = mapping_summary.counts_dataframe()
        x_values = [str(value) for value in count_table["ideal_function_no"].tolist()]
        y_values = count_table["mapped_count"].tolist()

        plot = figure(
            title="Mapped test points per ideal function",
            x_axis_label="Ideal function number",
            y_axis_label="Mapped count",
            x_range=x_values,
            width=700,
            height=400,
        )
        plot.vbar(x=x_values, top=y_values, width=0.8, color="#4c78a8")

        output_path = self.output_directory / file_name
        output_file(output_path)
        save(column(plot))
        return output_path
