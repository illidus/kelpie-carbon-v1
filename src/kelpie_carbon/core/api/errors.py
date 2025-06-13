"""Standardized error handling for Kelpie Carbon API.

This module provides comprehensive error handling with standardized HTTP responses,
detailed error messages, and proper exception chaining for debugging.
"""

from __future__ import annotations

import traceback
from enum import Enum

from fastapi import HTTPException
from pydantic import BaseModel

from ..logging_config import get_logger

logger = get_logger(__name__)


class ErrorCode(str, Enum):
    """Standardized error codes for the API."""

    # Client errors (4xx)
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_COORDINATES = "INVALID_COORDINATES"
    INVALID_DATE_RANGE = "INVALID_DATE_RANGE"
    INVALID_ANALYSIS_ID = "INVALID_ANALYSIS_ID"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_PARAMETER = "INVALID_PARAMETER"

    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    DATA_PROCESSING_ERROR = "DATA_PROCESSING_ERROR"
    SATELLITE_DATA_ERROR = "SATELLITE_DATA_ERROR"
    MODEL_PREDICTION_ERROR = "MODEL_PREDICTION_ERROR"
    IMAGERY_GENERATION_ERROR = "IMAGERY_GENERATION_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"


class ErrorDetail(BaseModel):
    """Detailed error information."""

    code: ErrorCode
    message: str
    details: str | None = None
    field: str | None = None
    suggestions: list[str] | None = None


class StandardizedError(HTTPException):
    """Standardized error class with consistent format."""

    def __init__(
        self,
        status_code: int,
        error_code: ErrorCode,
        message: str,
        details: str | None = None,
        field: str | None = None,
        suggestions: list[str] | None = None,
        log_error: bool = True,
    ):
        """Initialize standardized error.

        Args:
            status_code: HTTP status code
            error_code: Standardized error code
            message: Error message
            details: Optional error details
            field: Optional field name for validation errors
            suggestions: Optional list of suggestions
            log_error: Whether to log the error

        """
        self.error_detail = ErrorDetail(
            code=error_code,
            message=message,
            details=details,
            field=field,
            suggestions=suggestions,
        )

        # Create FastAPI-compatible detail
        detail = {
            "error": {
                "code": error_code.value,
                "message": message,
                "details": details,
                "field": field,
                "suggestions": suggestions,
            }
        }

        super().__init__(status_code=status_code, detail=detail)

        if log_error:
            if status_code >= 500:
                logger.error(
                    f"Server error {status_code}: {error_code.value} - {message}"
                )
                if details:
                    logger.error(f"Error details: {details}")
            else:
                logger.warning(
                    f"Client error {status_code}: {error_code.value} - {message}"
                )


def create_validation_error(
    message: str, field: str | None = None, details: str | None = None
) -> StandardizedError:
    """Create a standardized validation error (400)."""
    return StandardizedError(
        status_code=400,
        error_code=ErrorCode.INVALID_REQUEST,
        message=message,
        field=field,
        details=details,
        suggestions=["Check the request format and try again"],
    )


def create_not_found_error(
    resource: str, identifier: str | None = None
) -> StandardizedError:
    """Create a standardized not found error (404)."""
    message = f"{resource} not found"
    if identifier:
        message += f": {identifier}"

    return StandardizedError(
        status_code=404,
        error_code=ErrorCode.RESOURCE_NOT_FOUND,
        message=message,
        suggestions=[f"Verify the {resource.lower()} identifier and try again"],
    )


def create_coordinate_error(
    message: str, lat: float | None = None, lng: float | None = None
) -> StandardizedError:
    """Create a standardized coordinate validation error (400)."""
    details = None
    if lat is not None and lng is not None:
        details = f"Provided coordinates: lat={lat}, lng={lng}"

    return StandardizedError(
        status_code=400,
        error_code=ErrorCode.INVALID_COORDINATES,
        message=message,
        details=details,
        suggestions=[
            "Latitude must be between -90 and 90",
            "Longitude must be between -180 and 180",
            "Ensure coordinates are over water areas for kelp analysis",
        ],
    )


def create_date_range_error(
    message: str, start_date: str | None = None, end_date: str | None = None
) -> StandardizedError:
    """Create a standardized date range validation error (400)."""
    details = None
    if start_date and end_date:
        details = f"Provided range: {start_date} to {end_date}"

    return StandardizedError(
        status_code=400,
        error_code=ErrorCode.INVALID_DATE_RANGE,
        message=message,
        details=details,
        suggestions=[
            "Use ISO format: YYYY-MM-DD",
            "End date must be after start date",
            "Date range should not exceed 1 year",
        ],
    )


def create_processing_error(
    operation: str,
    original_error: Exception,
    analysis_id: str | None = None,
) -> StandardizedError:
    """Create a standardized processing error (500)."""
    message = f"{operation} failed"
    if analysis_id:
        message += f" for analysis {analysis_id}"

    # Get error details from original exception
    error_details = str(original_error)

    # Log the full traceback for debugging
    logger.error(f"Processing error in {operation}: {error_details}")
    logger.error(traceback.format_exc())

    return StandardizedError(
        status_code=500,
        error_code=ErrorCode.DATA_PROCESSING_ERROR,
        message=message,
        details=error_details[:200] if error_details else None,
        suggestions=[
            "Try again in a few moments",
            "Check if the area has recent satellite coverage",
            "Contact support if the problem persists",
        ],
    )


def create_satellite_data_error(
    message: str,
    coordinates: dict[str, float] | None = None,
) -> StandardizedError:
    """Create a standardized satellite data error (422)."""
    details = None
    if coordinates:
        details = (
            f"Location: lat={coordinates.get('lat')}, lng={coordinates.get('lng')}"
        )

    return StandardizedError(
        status_code=422,
        error_code=ErrorCode.SATELLITE_DATA_ERROR,
        message=message,
        details=details,
        suggestions=[
            "Try a different date range",
            "Check if the location has satellite coverage",
            "Ensure coordinates are over water areas",
        ],
    )


def create_imagery_error(
    operation: str,
    analysis_id: str,
    original_error: Exception | None = None,
) -> StandardizedError:
    """Create a standardized imagery generation error (500)."""
    message = f"{operation} failed for analysis {analysis_id}"

    details = None
    if original_error:
        details = str(original_error)[:200]
        logger.error(f"Imagery error: {details}")

    return StandardizedError(
        status_code=500,
        error_code=ErrorCode.IMAGERY_GENERATION_ERROR,
        message=message,
        details=details,
        suggestions=[
            "Try refreshing the analysis",
            "Check if the analysis completed successfully",
            "Contact support if the problem persists",
        ],
    )


def create_service_unavailable_error(
    service: str,
    details: str | None = None,
) -> StandardizedError:
    """Create a standardized service unavailable error (503)."""
    return StandardizedError(
        status_code=503,
        error_code=ErrorCode.SERVICE_UNAVAILABLE,
        message=f"{service} is temporarily unavailable",
        details=details,
        suggestions=[
            "Try again in a few minutes",
            "Check system status",
            "Contact support if the problem persists",
        ],
    )


def handle_unexpected_error(
    operation: str,
    error: Exception,
    analysis_id: str | None = None,
) -> StandardizedError:
    """Handle unexpected errors with proper logging and standardized response."""
    error_details = str(error)

    # Log full traceback for debugging
    logger.error(f"Unexpected error in {operation}: {error_details}")
    logger.error(traceback.format_exc())

    message = f"Unexpected error during {operation}"
    if analysis_id:
        message += f" (analysis {analysis_id})"

    return StandardizedError(
        status_code=500,
        error_code=ErrorCode.INTERNAL_ERROR,
        message=message,
        details=error_details[:200] if error_details else None,
        suggestions=[
            "Try again in a few moments",
            "Contact support with the analysis ID if the problem persists",
        ],
    )
