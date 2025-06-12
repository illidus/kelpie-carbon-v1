"""Satellite imagery visualization and processing module."""

from .generators import (
    generate_false_color_composite,
    generate_rgb_composite,
    generate_spectral_visualization,
)
from .overlays import (
    create_colored_mask,
    generate_biomass_heatmap,
    generate_cloud_mask_overlay,
    generate_kelp_mask_overlay,
    generate_mask_overlay,
    generate_water_mask_overlay,
)
from .utils import (
    apply_colormap,
    array_to_image,
    get_image_bounds,
    normalize_band,
)

__all__ = [
    "generate_rgb_composite",
    "generate_false_color_composite",
    "generate_spectral_visualization",
    "generate_mask_overlay",
    "generate_biomass_heatmap",
    "create_colored_mask",
    "generate_kelp_mask_overlay",
    "generate_water_mask_overlay",
    "generate_cloud_mask_overlay",
    "normalize_band",
    "apply_colormap",
    "array_to_image",
    "get_image_bounds",
]
