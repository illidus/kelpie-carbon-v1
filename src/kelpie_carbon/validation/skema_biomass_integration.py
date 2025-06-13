"""SKEMA biomass integration module for backwards compatibility.

This module provides backwards compatibility by importing from validation.core.skema_biomass_integration.
"""

from .core.skema_biomass_integration import (
    BiomassGroundTruth,
    BiomassValidationSite,
    SKEMABiomassDatasetIntegrator,
    SKEMAIntegrationConfig,
)

__all__ = [
    "BiomassGroundTruth",
    "BiomassValidationSite",
    "SKEMABiomassDatasetIntegrator",
    "SKEMAIntegrationConfig",
]
