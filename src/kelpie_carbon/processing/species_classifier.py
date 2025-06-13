"""Species classifier module for backwards compatibility.

This module provides backwards compatibility by importing from core.processing.species_classifier.
"""

from ..core.processing.species_classifier import (
    KelpSpecies,
    SpeciesClassifier,
)

__all__ = [
    "KelpSpecies",
    "SpeciesClassifier",
]
