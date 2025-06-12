"""Core functionality for Kelpie-Carbon.

This package contains:
- Configuration management
- Logging utilities
- Core processing functions
- Machine learning models
- API endpoints
- CLI interface
- Spectral analysis
- Deep learning components
- Optimization algorithms
"""

# Configuration and logging
# CLI interface
from .cli import app
from .config import Config, get_settings, load
from .logging_config import get_logger, setup_logging

# Core functionality (import what's available)
try:
    # These imports may fail if the modules are still being organized
    from .fetch import fetch_sentinel_tiles
    from .indices import calculate_indices_from_dataset
    from .mask import apply_mask, get_mask_statistics
    from .model import KelpBiomassModel, predict_biomass
except ImportError as e:
    # Modules may not be available yet during refactoring
    print(f"Note: Some core modules not yet available: {e}")

__all__ = [
    "get_settings",
    "Config",
    "load",
    "setup_logging",
    "get_logger",
    "app",
    # Core functions (when available)
    "calculate_indices_from_dataset",
    "fetch_sentinel_tiles",
    "apply_mask",
    "get_mask_statistics",
    "KelpBiomassModel",
    "predict_biomass",
]
