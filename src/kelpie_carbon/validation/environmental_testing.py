"""Environmental testing module for backwards compatibility.

This module provides backwards compatibility by importing from validation.core.environmental_testing.
"""

from .core.environmental_testing import *

__all__ = [
    "EnvironmentalCondition",
    "EnvironmentalRobustnessValidator",
    "EnvironmentalValidationError",
] 