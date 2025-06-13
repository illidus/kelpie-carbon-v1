"""Processing module for backwards compatibility.

This module provides backwards compatibility by importing from core.processing.
"""

from ..core.processing import (
    BiomassEstimate,
    BladeFromdDetector,
    KelpSpecies,
    MorphologicalFeature,
    MorphologyDetectionResult,
    MorphologyDetector,
    MorphologyType,
    PneumatocystDetector,
    SpeciesClassificationResult,
    SpeciesClassifier,
    create_morphology_detector,
    create_species_classifier,
)

__all__ = [
    "BiomassEstimate",
    "BladeFromdDetector",
    "KelpSpecies",
    "MorphologicalFeature",
    "MorphologyDetectionResult",
    "MorphologyDetector",
    "MorphologyType",
    "PneumatocystDetector",
    "SpeciesClassificationResult",
    "SpeciesClassifier",
    "create_morphology_detector",
    "create_species_classifier",
]
