"""Reporting and visualization for Kelpie-Carbon.

This package contains:
- Visualization tools and plotting
- Analytics framework
- Web-based reporting interfaces
- Report generation utilities
"""

# Import reporting functionality
try:
    from .analytics import (
        AnalyticsFramework,
        FirstNationsReport,
        ManagementReport,
        ScientificReport,
        create_analytics_framework,
    )
except ImportError:
    # Modules may not be fully organized yet
    pass

try:
    from .visualization import *
except ImportError:
    pass

try:
    from .web import *
except ImportError:
    pass

__all__ = [
    "AnalyticsFramework",
    "create_analytics_framework",
    "FirstNationsReport",
    "ScientificReport",
    "ManagementReport",
    # Additional exports will be added as modules are organized
]
