"""Processing module for backwards compatibility.

This module provides backwards compatibility by importing from core.processing.
"""

from ..core.processing import *

__all__ = [
    "KelpSpecies",
    "SpeciesClassifier",
    "SpeciesClassificationError",
    "DerivativeFeatures",
    "WaterAnomalyFilter",
    "MorphologyDetector",
]
