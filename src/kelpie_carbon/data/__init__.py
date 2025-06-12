"""Data handling and processing for Kelpie-Carbon.

This package contains:
- Core data structures and management
- Imagery processing and handling
- Detection algorithms
- Data loading and preprocessing
"""

# Import what's available from data modules
try:
    from .core import *
except ImportError:
    pass

try:
    from .imagery import *
except ImportError:
    pass

try:
    from .detection import *
except ImportError:
    pass

__all__ = [
    # Data handling functions will be added as modules are organized
] 
