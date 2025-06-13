"""Historical baseline analysis module for backwards compatibility.

This module provides backwards compatibility by importing from
validation.core.historical_baseline_analysis.
"""

from .core.historical_baseline_analysis import (
    ChangeDetectionAnalyzer,
    HistoricalDataset,
    HistoricalSite,
)

__all__: list[str] = [
    "ChangeDetectionAnalyzer",
    "HistoricalDataset",
    "HistoricalSite",
]
