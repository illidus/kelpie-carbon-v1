"""Kelpie-Carbon package.

A sophisticated satellite imagery analysis application for assessing
kelp forest carbon sequestration using Sentinel-2 data.

Refactored into four main packages:
- core: Core functionality, configuration, utilities
- data: Data handling, imagery processing, detection
- validation: Validation framework and metrics
- reporting: Visualization, analytics, and reporting
"""

import contextlib

__version__ = "0.1.0"
__author__ = "Ryan Eyre"
__email__ = "ryaneyre1337@gmail.com"
__description__ = "Kelp Forest Carbon Sequestration Assessment"

# Import from core package
from .core.cli import app as cli_app
from .core.config import Config, get_settings, load
from .core.logging_config import get_logger, setup_logging

# Import core functionality
with contextlib.suppress(ImportError):
    from .core import (
        KelpBiomassModel,
        apply_mask,
        calculate_indices_from_dataset,
        fetch_sentinel_tiles,
        get_mask_statistics,
        predict_biomass,
    )

# Import validation framework
with contextlib.suppress(ImportError):
    from .validation.core import (
        RealWorldValidator,
        ValidationResult,
        ValidationSite,
        validate_primary_sites,
        validate_with_controls,
    )

# Import analytics framework
with contextlib.suppress(ImportError):
    from .reporting.analytics import (
        AnalyticsFramework,
        FirstNationsReport,
        ManagementReport,
        ScientificReport,
        create_analytics_framework,
    )

# Import API framework
with contextlib.suppress(ImportError):
    from .core import api

    # Make API available at package level for backwards compatibility
    # This allows imports like: from kelpie_carbon.core.api.models import CoordinateModel

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "get_settings",
    "Config",
    "load",
    "setup_logging",
    "get_logger",
    "cli_app",
    # Core functionality (when available)
    "fetch_sentinel_tiles",
    "calculate_indices_from_dataset",
    "apply_mask",
    "get_mask_statistics",
    "KelpBiomassModel",
    "predict_biomass",
    # Analytics framework (when available)
    "AnalyticsFramework",
    "create_analytics_framework",
    "FirstNationsReport",
    "ScientificReport",
    "ManagementReport",
    # Validation framework (when available)
    "RealWorldValidator",
    "ValidationSite",
    "ValidationResult",
    "validate_primary_sites",
    "validate_with_controls",
]
