"""Utility modules for Kelpie Carbon v1.

This package contains organized utility functions for common operations:
- Array and data manipulation utilities
- Validation and error handling utilities  
- Performance and monitoring utilities
- Math and statistical utilities
"""

from .array_utils import (
    normalize_array,
    clip_array_percentiles,
    calculate_statistics,
    safe_divide,
    interpolate_missing_values,
)

from .validation_utils import (
    validate_coordinates,
    validate_date_range,
    validate_dataset_bands,
    validate_config_structure,
    ValidationError,
)

from .performance_utils import (
    timing_context,
    memory_usage,
    profile_function,
    PerformanceMonitor,
)

from .math_utils import (
    calculate_area_from_pixels,
    convert_coordinates,
    calculate_distance,
    gaussian_kernel,
)

__all__ = [
    # Array utilities
    "normalize_array",
    "clip_array_percentiles", 
    "calculate_statistics",
    "safe_divide",
    "interpolate_missing_values",
    
    # Validation utilities
    "validate_coordinates",
    "validate_date_range",
    "validate_dataset_bands",
    "validate_config_structure",
    "ValidationError",
    
    # Performance utilities
    "timing_context",
    "memory_usage",
    "profile_function",
    "PerformanceMonitor",
    
    # Math utilities
    "calculate_area_from_pixels",
    "convert_coordinates",
    "calculate_distance",
    "gaussian_kernel",
] 