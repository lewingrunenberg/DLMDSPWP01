"""Custom exceptions for assignment-specific validation and pipeline failures."""

from __future__ import annotations


class AssignmentProjectError(Exception):
    """Base exception for all domain-specific project errors."""


class DataValidationError(AssignmentProjectError):
    """Raised when dataset contents violate required validation rules."""


class CsvSchemaError(DataValidationError):
    """Raised when a CSV file does not match the expected assignment schema."""


class XGridValidationError(DataValidationError):
    """Raised when x-values are invalid for comparison or lookup."""


class MappingError(AssignmentProjectError):
    """Raised when test-point mapping cannot be performed safely."""


class PersistenceError(AssignmentProjectError):
    """Raised when database persistence or artifact writing fails."""
