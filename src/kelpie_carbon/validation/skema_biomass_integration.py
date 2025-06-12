"""SKEMA biomass integration module for backwards compatibility.

This module provides backwards compatibility by importing from validation.core.skema_biomass_integration.
"""

from .core.skema_biomass_integration import *

__all__ = [
    "SKEMAIntegrationConfig",
    "SiteBiomassData",
]
