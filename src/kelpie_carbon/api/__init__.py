"""API module for backwards compatibility.

This module provides backwards compatibility by importing from core.api.
"""

# Import everything from core.api for backwards compatibility
from ..core.api.errors import *
from ..core.api.imagery import *
from ..core.api.main import *
from ..core.api.models import *

__all__ = [
    # From errors
    "ErrorCode",
    "StandardizedError", 
    "create_not_found_error",
    "create_validation_error",
    "handle_unexpected_error",
    # From imagery
    "_analysis_cache",
    # From main
    "app",
    # From models
    "CoordinateModel",
] 