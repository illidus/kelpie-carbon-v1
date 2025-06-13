"""Data handling and processing for Kelpie-Carbon.

This package contains:
- Core data structures and management
- Imagery processing and handling
- Detection algorithms
- Data loading and preprocessing
"""

import contextlib

# Import what's available from data modules
with contextlib.suppress(ImportError):
    from .core import (
        SKEMADataIntegrator,
        SKEMAValidationPoint,
        get_skema_validation_data,
    )

with contextlib.suppress(ImportError):
    from .imagery import (
        apply_colormap,
        array_to_image,
        create_colored_mask,
        generate_biomass_heatmap,
        generate_cloud_mask_overlay,
        generate_false_color_composite,
        generate_kelp_mask_overlay,
        generate_mask_overlay,
        generate_rgb_composite,
        generate_spectral_visualization,
        generate_water_mask_overlay,
        get_image_bounds,
        normalize_band,
    )

with contextlib.suppress(ImportError):
    from .detection import (
        DepthDetectionResult,
        SubmergedKelpConfig,
        SubmergedKelpDetector,
        WaterColumnModel,
        analyze_depth_distribution,
        create_submerged_kelp_detector,
        detect_submerged_kelp,
    )

__all__: list[str] = [
    # Core data integration
    "SKEMADataIntegrator",
    "SKEMAValidationPoint",
    "get_skema_validation_data",
    # Imagery functions
    "apply_colormap",
    "array_to_image",
    "create_colored_mask",
    "generate_biomass_heatmap",
    "generate_cloud_mask_overlay",
    "generate_false_color_composite",
    "generate_kelp_mask_overlay",
    "generate_mask_overlay",
    "generate_rgb_composite",
    "generate_spectral_visualization",
    "generate_water_mask_overlay",
    "get_image_bounds",
    "normalize_band",
    # Detection classes
    "DepthDetectionResult",
    "SubmergedKelpConfig",
    "SubmergedKelpDetector",
    "WaterColumnModel",
    "analyze_depth_distribution",
    "create_submerged_kelp_detector",
    "detect_submerged_kelp",
]
