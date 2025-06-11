"""Tests for standardized error handling."""

import pytest
from fastapi import HTTPException
from kelpie_carbon_v1.api.errors import (
    ErrorCode,
    StandardizedError,
    create_validation_error,
    create_not_found_error,
    create_coordinate_error,
    create_date_range_error,
    create_processing_error,
    create_satellite_data_error,
    create_imagery_error,
    create_service_unavailable_error,
    handle_unexpected_error,
)


class TestStandardizedErrors:
    """Test standardized error handling."""

    def test_standardized_error_structure(self):
        """Test that standardized errors have consistent structure."""
        error = StandardizedError(
            status_code=400,
            error_code=ErrorCode.INVALID_REQUEST,
            message="Test error message",
            details="Test details",
            field="test_field",
            suggestions=["Try this", "Or this"],
        )
        
        assert error.status_code == 400
        assert error.error_detail.code == ErrorCode.INVALID_REQUEST
        assert error.error_detail.message == "Test error message"
        assert error.error_detail.details == "Test details"
        assert error.error_detail.field == "test_field"
        assert error.error_detail.suggestions == ["Try this", "Or this"]
        
        # Check FastAPI-compatible detail format
        expected_detail = {
            "error": {
                "code": "INVALID_REQUEST",
                "message": "Test error message",
                "details": "Test details",
                "field": "test_field",
                "suggestions": ["Try this", "Or this"],
            }
        }
        assert error.detail == expected_detail

    def test_validation_error(self):
        """Test validation error creation."""
        error = create_validation_error(
            "Invalid input format",
            field="email",
            details="Must be valid email address"
        )
        
        assert error.status_code == 400
        assert error.error_detail.code == ErrorCode.INVALID_REQUEST
        assert error.error_detail.message == "Invalid input format"
        assert error.error_detail.field == "email"
        assert error.error_detail.details == "Must be valid email address"
        assert "Check the request format and try again" in error.error_detail.suggestions

    def test_not_found_error(self):
        """Test not found error creation."""
        error = create_not_found_error("Analysis", "abc123")
        
        assert error.status_code == 404
        assert error.error_detail.code == ErrorCode.RESOURCE_NOT_FOUND
        assert error.error_detail.message == "Analysis not found: abc123"
        assert "Verify the analysis identifier and try again" in error.error_detail.suggestions

    def test_coordinate_error(self):
        """Test coordinate validation error creation."""
        error = create_coordinate_error(
            "Invalid latitude value",
            lat=95.0,
            lng=-120.0
        )
        
        assert error.status_code == 400
        assert error.error_detail.code == ErrorCode.INVALID_COORDINATES
        assert error.error_detail.message == "Invalid latitude value"
        assert "lat=95.0, lng=-120.0" in error.error_detail.details
        assert "Latitude must be between -90 and 90" in error.error_detail.suggestions

    def test_date_range_error(self):
        """Test date range validation error creation."""
        error = create_date_range_error(
            "End date must be after start date",
            start_date="2023-12-01",
            end_date="2023-11-01"
        )
        
        assert error.status_code == 400
        assert error.error_detail.code == ErrorCode.INVALID_DATE_RANGE
        assert error.error_detail.message == "End date must be after start date"
        assert "2023-12-01 to 2023-11-01" in error.error_detail.details
        assert "Use ISO format: YYYY-MM-DD" in error.error_detail.suggestions

    def test_processing_error(self):
        """Test processing error creation."""
        original_error = ValueError("Missing required data")
        error = create_processing_error(
            "Data processing",
            original_error,
            analysis_id="abc123"
        )
        
        assert error.status_code == 500
        assert error.error_detail.code == ErrorCode.DATA_PROCESSING_ERROR
        assert error.error_detail.message == "Data processing failed for analysis abc123"
        assert "Missing required data" in error.error_detail.details
        assert "Try again in a few moments" in error.error_detail.suggestions

    def test_satellite_data_error(self):
        """Test satellite data error creation."""
        error = create_satellite_data_error(
            "No satellite data available for this location",
            coordinates={"lat": 45.0, "lng": -123.0}
        )
        
        assert error.status_code == 422
        assert error.error_detail.code == ErrorCode.SATELLITE_DATA_ERROR
        assert error.error_detail.message == "No satellite data available for this location"
        assert "lat=45.0, lng=-123.0" in error.error_detail.details
        assert "Try a different date range" in error.error_detail.suggestions

    def test_imagery_error(self):
        """Test imagery generation error creation."""
        original_error = RuntimeError("Memory allocation failed")
        error = create_imagery_error(
            "RGB composite generation",
            "abc123",
            original_error=original_error
        )
        
        assert error.status_code == 500
        assert error.error_detail.code == ErrorCode.IMAGERY_GENERATION_ERROR
        assert error.error_detail.message == "RGB composite generation failed for analysis abc123"
        assert "Memory allocation failed" in error.error_detail.details
        assert "Try refreshing the analysis" in error.error_detail.suggestions

    def test_service_unavailable_error(self):
        """Test service unavailable error creation."""
        error = create_service_unavailable_error(
            "Satellite data service",
            details="Scheduled maintenance until 2023-12-01 10:00 UTC"
        )
        
        assert error.status_code == 503
        assert error.error_detail.code == ErrorCode.SERVICE_UNAVAILABLE
        assert error.error_detail.message == "Satellite data service is temporarily unavailable"
        assert "Scheduled maintenance" in error.error_detail.details
        assert "Try again in a few minutes" in error.error_detail.suggestions

    def test_unexpected_error_handling(self):
        """Test handling of unexpected errors."""
        original_error = KeyError("Unexpected key missing")
        error = handle_unexpected_error(
            "data analysis",
            original_error,
            analysis_id="abc123"
        )
        
        assert error.status_code == 500
        assert error.error_detail.code == ErrorCode.INTERNAL_ERROR
        assert error.error_detail.message == "Unexpected error during data analysis (analysis abc123)"
        assert "Unexpected key missing" in error.error_detail.details
        assert "Contact support with the analysis ID if the problem persists" in error.error_detail.suggestions

    def test_error_without_optional_fields(self):
        """Test error creation without optional fields."""
        error = create_not_found_error("Resource")
        
        assert error.status_code == 404
        assert error.error_detail.code == ErrorCode.RESOURCE_NOT_FOUND
        assert error.error_detail.message == "Resource not found"
        assert error.error_detail.details is None
        assert error.error_detail.field is None
        assert error.error_detail.suggestions is not None

    def test_error_inheritance(self):
        """Test that StandardizedError inherits from HTTPException."""
        error = StandardizedError(
            status_code=400,
            error_code=ErrorCode.INVALID_REQUEST,
            message="Test message"
        )
        
        assert isinstance(error, HTTPException)
        assert isinstance(error, StandardizedError)


class TestErrorCodes:
    """Test error code enumeration."""

    def test_error_code_values(self):
        """Test that error codes have expected string values."""
        assert ErrorCode.INVALID_REQUEST.value == "INVALID_REQUEST"
        assert ErrorCode.INVALID_COORDINATES.value == "INVALID_COORDINATES"
        assert ErrorCode.RESOURCE_NOT_FOUND.value == "RESOURCE_NOT_FOUND"
        assert ErrorCode.INTERNAL_ERROR.value == "INTERNAL_ERROR"
        assert ErrorCode.DATA_PROCESSING_ERROR.value == "DATA_PROCESSING_ERROR"

    def test_error_code_coverage(self):
        """Test that we have error codes for different HTTP status ranges."""
        # Client errors (4xx)
        client_error_codes = [
            ErrorCode.INVALID_REQUEST,
            ErrorCode.INVALID_COORDINATES,
            ErrorCode.INVALID_DATE_RANGE,
            ErrorCode.RESOURCE_NOT_FOUND,
        ]
        
        # Server errors (5xx)
        server_error_codes = [
            ErrorCode.INTERNAL_ERROR,
            ErrorCode.DATA_PROCESSING_ERROR,
            ErrorCode.IMAGERY_GENERATION_ERROR,
            ErrorCode.SERVICE_UNAVAILABLE,
        ]
        
        assert len(client_error_codes) >= 4
        assert len(server_error_codes) >= 4 