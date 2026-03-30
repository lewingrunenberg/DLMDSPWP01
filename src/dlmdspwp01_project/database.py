"""SQLAlchemy-backed persistence for assignment-aligned database outputs."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from dlmdspwp01_project.config import DATABASE_PATH, IDEAL_FUNCTION_COLUMNS, TRAIN_FUNCTION_COLUMNS
from dlmdspwp01_project.datasets import IdealDataset, TrainingDataset
from dlmdspwp01_project.exceptions import PersistenceError
from dlmdspwp01_project.models import MappingSummary, SelectionSummary


class DatabaseManager:
    """Create and populate the SQLite database required by the assignment."""

    def __init__(self, database_path: str | Path = DATABASE_PATH) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine: Engine = create_engine(f"sqlite:///{self.database_path}")
        self.metadata = MetaData()

        self.training_data = Table(
            "training_data",
            self.metadata,
            Column("x", Float, primary_key=True),
            *(Column(column_name, Float, nullable=False) for column_name in TRAIN_FUNCTION_COLUMNS),
        )
        self.ideal_functions = Table(
            "ideal_functions",
            self.metadata,
            Column("x", Float, primary_key=True),
            *(Column(column_name, Float, nullable=False) for column_name in IDEAL_FUNCTION_COLUMNS),
        )
        self.test_mapping = Table(
            "test_mapping",
            self.metadata,
            Column("x", Float, nullable=False),
            Column("y", Float, nullable=False),
            Column("delta_y", Float, nullable=False),
            Column("ideal_function_no", Integer, nullable=False),
        )
        self.selected_function_pairs = Table(
            "selected_function_pairs",
            self.metadata,
            Column("training_function_name", String, primary_key=True),
            Column("ideal_function_name", String, nullable=False),
            Column("ideal_function_no", Integer, nullable=False),
            Column("sse", Float, nullable=False),
            Column("max_abs_deviation", Float, nullable=False),
            Column("mapping_threshold", Float, nullable=False),
        )

    def initialize_database(self) -> None:
        """Create all configured tables if they do not already exist."""

        try:
            self.metadata.create_all(self.engine)
        except SQLAlchemyError as error:
            raise PersistenceError("Failed to initialize the SQLite database.") from error

    def persist_all(
        self,
        training_dataset: TrainingDataset,
        ideal_dataset: IdealDataset,
        selection_summary: SelectionSummary,
        mapping_summary: MappingSummary,
    ) -> None:
        """Persist raw inputs and derived outputs into the database."""

        self.initialize_database()

        try:
            with self.engine.begin() as connection:
                connection.execute(self.training_data.delete())
                connection.execute(self.ideal_functions.delete())
                connection.execute(self.test_mapping.delete())
                connection.execute(self.selected_function_pairs.delete())

                connection.execute(
                    self.training_data.insert(),
                    self._records_from_dataframe(training_dataset.to_dataframe()),
                )
                connection.execute(
                    self.ideal_functions.insert(),
                    self._records_from_dataframe(ideal_dataset.to_dataframe()),
                )

                mapped_records = self._records_from_dataframe(mapping_summary.to_dataframe())
                if mapped_records:
                    connection.execute(self.test_mapping.insert(), mapped_records)

                selected_records = self._records_from_dataframe(selection_summary.to_dataframe())
                if selected_records:
                    connection.execute(self.selected_function_pairs.insert(), selected_records)
        except SQLAlchemyError as error:
            raise PersistenceError("Failed to persist assignment results to SQLite.") from error

    @staticmethod
    def _records_from_dataframe(dataframe) -> list[dict]:
        """Convert a dataframe to plain SQLAlchemy-ready records."""

        return list(dataframe.to_dict(orient="records"))
