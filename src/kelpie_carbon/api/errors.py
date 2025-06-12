"""Errors module for backwards compatibility.

This module provides backwards compatibility by importing from core.api.errors.
"""

from ..core.api.errors import *

__all__ = [
    "ErrorCode",
    "StandardizedError",
    "create_not_found_error",
    "create_validation_error", 
    "handle_unexpected_error",
    "create_imagery_error",
    "create_coordinate_error",
    "create_date_range_error",
    "create_processing_error",
    "create_service_unavailable_error",
] 