"""Main module for backwards compatibility.

This module provides backwards compatibility by importing from core.api.main.
"""

from ..core.api.main import *

__all__ = [
    "app",
    "root",
    "health",
    "readiness",
    "run_analysis",
]
