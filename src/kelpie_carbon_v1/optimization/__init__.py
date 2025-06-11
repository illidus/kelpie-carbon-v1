"""Optimization Module for SKEMA Detection Pipeline.

This module provides optimization tools for tuning detection thresholds,
performance parameters, and environmental adaptations.

Task A2.7: Optimize detection pipeline - Performance Optimization
"""

from .threshold_optimizer import (
    ThresholdOptimizer,
    optimize_detection_pipeline,
    get_optimized_config_for_site
)

__all__ = [
    'ThresholdOptimizer',
    'optimize_detection_pipeline', 
    'get_optimized_config_for_site'
] 