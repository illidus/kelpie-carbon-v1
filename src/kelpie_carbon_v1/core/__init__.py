"""Core processing modules for Kelpie Carbon v1."""

from .fetch import fetch_sentinel_tiles
from .indices import calculate_indices_from_dataset
from .mask import apply_mask, get_mask_statistics, create_skema_kelp_detection_mask
from .model import KelpBiomassModel, predict_biomass

__all__ = [
    "fetch_sentinel_tiles",
    "calculate_indices_from_dataset",
    "apply_mask",
    "get_mask_statistics",
    "create_skema_kelp_detection_mask",
    "KelpBiomassModel",
    "predict_biomass",
]
