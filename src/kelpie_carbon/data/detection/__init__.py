"""
Detection module for Kelpie Carbon v1.

This module provides comprehensive kelp detection capabilities including:
- Traditional surface kelp detection using NDVI/NDRE
- Submerged kelp detection using red-edge methodology  
- Species-level classification and detection
- Depth-sensitive detection algorithms
- Water column modeling for depth estimation
"""

from .submerged_kelp_detection import (
    DepthDetectionResult,
    SubmergedKelpConfig,
    SubmergedKelpDetector,
    WaterColumnModel,
    analyze_depth_distribution,
    create_submerged_kelp_detector,
    detect_submerged_kelp,
)

__all__ = [
    # Submerged kelp detection
    "SubmergedKelpDetector",
    "SubmergedKelpConfig", 
    "WaterColumnModel",
    "DepthDetectionResult",
    "create_submerged_kelp_detector",
    "detect_submerged_kelp",
    "analyze_depth_distribution",
] 
