"""Real world validation module for backwards compatibility.

This module provides backwards compatibility by importing from validation.core.real_world_validation.
"""

from .core.real_world_validation import (
    RealWorldValidator,
    ValidationResult,
    ValidationSite,
)

__all__ = [
    "RealWorldValidator",
    "ValidationResult",
    "ValidationSite",
]
