"""
Validation module for SKEMA integration - Task 2

This module provides comprehensive validation capabilities for the enhanced NDRE
kelp detection system, including:
- Ground truth data management
- Field campaign coordination
- Validation metrics calculation
- Comparative analysis tools
"""

__version__ = "1.0.0"
__author__ = "Kelpie Carbon Team"

from .data_manager import ValidationDataManager
from .metrics import ValidationMetrics
from .mock_data import MockValidationGenerator
from .field_protocols import FieldCampaignProtocols

__all__ = [
    "ValidationDataManager",
    "ValidationMetrics", 
    "MockValidationGenerator",
    "FieldCampaignProtocols"
] 