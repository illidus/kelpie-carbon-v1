"""Reporting and visualization for Kelpie-Carbon.

This package contains:
- Visualization tools and plotting
- Analytics framework
- Web-based reporting interfaces
- Report generation utilities
"""

# Import reporting functionality
import contextlib

with contextlib.suppress(ImportError):
    from .analytics import (
        AnalyticsFramework,
        FirstNationsReport,
        ManagementReport,
        ScientificReport,
        create_analytics_framework,
    )

with contextlib.suppress(ImportError):
    from .visualization import *

with contextlib.suppress(ImportError):
    from .web import *

__all__ = [
    "AnalyticsFramework",
    "create_analytics_framework",
    "FirstNationsReport",
    "ScientificReport",
    "ManagementReport",
    # Additional exports will be added as modules are organized
]
