"""Kelpie-Carbon v1 package.

A sophisticated satellite imagery analysis application for assessing 
kelp forest carbon sequestration using Sentinel-2 data.
"""

__version__ = "0.1.0"
__author__ = "Ryan Eyre"
__email__ = "ryaneyre1337@gmail.com"
__description__ = "Kelp Forest Carbon Sequestration Assessment"

# Main components
from .config import get_settings, SimpleConfig
from .logging_config import setup_logging, get_logger

# Core functionality
from .core import (
    fetch_sentinel_tiles,
    calculate_indices_from_dataset,
    apply_mask,
    get_mask_statistics,
    KelpBiomassModel,
    predict_biomass,
)

# Validation framework (Task 2 - SKEMA Integration)
from .validation import (
    ValidationDataManager,
    ValidationMetrics,
    MockValidationGenerator,
    FieldCampaignProtocols,
)

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__description__",
    "get_settings",
    "SimpleConfig",
    "setup_logging",
    "get_logger",
    "fetch_sentinel_tiles",
    "calculate_indices_from_dataset",
    "apply_mask",
    "get_mask_statistics", 
    "KelpBiomassModel",
    "predict_biomass",
    # Validation framework
    "ValidationDataManager",
    "ValidationMetrics",
    "MockValidationGenerator",
    "FieldCampaignProtocols",
]
