"""Validation framework for Kelpie-Carbon.

This package contains:
- Real-world validation tools
- Validation sites and results
- Validation metrics and methods
- Cross-validation frameworks
- T2-001: Standardized ValidationResult & metric helpers
- CLI interface for validation tasks
"""

# Import validation functionality
try:
    from .core import (
        RealWorldValidator,
        ValidationSite,
        validate_primary_sites,
        validate_with_controls,
    )

    # T2-001: Import new standardized validation classes
    from .core.metrics import (
        MetricHelpers,
        ValidationMetrics,
        ValidationResult,
    )
    
    # CLI interface
    from .cli import app as cli_app
except ImportError:
    # Modules may not be fully organized yet
    pass

__all__ = [
    "RealWorldValidator",
    "ValidationSite",
    "ValidationResult",
    "MetricHelpers",
    "ValidationMetrics",
    "validate_primary_sites",
    "validate_with_controls",
    "cli_app",
]
