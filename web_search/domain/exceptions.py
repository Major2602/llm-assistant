# web_search/domain/exceptions.py

from __future__ import annotations



# ==========================================================
# Base exceptions
# ==========================================================


class WebSearchError(Exception):
    """
    Base domain exception.
    """



# ==========================================================
# Validation errors
# ==========================================================


class ValidationError(
    WebSearchError
):
    """
    Invalid input data.
    """



class InvalidQueryError(
    ValidationError
):
    """
    Query validation failed.
    """



# ==========================================================
# Provider errors
# ==========================================================


class ProviderError(
    WebSearchError
):
    """
    External provider failure.
    """



class ProviderUnavailableError(
    ProviderError
):
    """
    Temporary provider outage.
    """



class InvalidProviderResponseError(
    ProviderError
):
    """
    Provider returned invalid data.
    """



# ==========================================================
# Storage errors
# ==========================================================


class StorageError(
    WebSearchError
):
    """
    Vector storage failure.
    """



class StorageUnavailableError(
    StorageError
):
    """
    Temporary storage unavailable.
    """



# ==========================================================
# Pipeline errors
# ==========================================================


class PipelineError(
    WebSearchError
):
    """
    Pipeline execution error.
    """



class StageExecutionError(
    PipelineError
):
    """
    Pipeline stage failed.
    """



# ==========================================================
# Retry classification
# ==========================================================


TRANSIENT_ERRORS = (
    ProviderUnavailableError,
    StorageUnavailableError,
)


PERMANENT_ERRORS = (
    ValidationError,
    InvalidProviderResponseError,
)
