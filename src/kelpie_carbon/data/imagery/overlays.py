"""Overlay generators for analysis result visualization."""


import numpy as np
import xarray as xr
from PIL import Image

from .utils import apply_colormap, array_to_image, create_rgba_overlay, normalize_to_0_1


def generate_mask_overlay(
    mask: xr.DataArray, color: tuple[int, int, int] = (0, 255, 0), alpha: float = 0.6
) -> Image.Image:
    """Generate semi-transparent colored overlay from boolean mask.

    Args:
        mask: Boolean mask data array
        color: RGB color tuple (0-255)
        alpha: Alpha transparency (0-1)

    Returns:
        RGBA PIL Image
    """
    # Convert to boolean numpy array
    mask_array: np.ndarray = mask.values.astype(bool)

    # Create RGBA overlay
    rgba_array = create_rgba_overlay(mask_array, color, alpha)

    return array_to_image(rgba_array)


def generate_kelp_mask_overlay(dataset: xr.Dataset, alpha: float = 0.6) -> Image.Image:
    """Generate kelp detection mask overlay in green.

    Args:
        dataset: Dataset containing kelp mask
        alpha: Alpha transparency (0-1)

    Returns:
        Green kelp mask overlay
    """
    if "kelp_mask" not in dataset:
        raise ValueError("Dataset does not contain kelp_mask")

    return generate_mask_overlay(
        dataset["kelp_mask"], color=(0, 255, 0), alpha=alpha  # Green
    )


def generate_water_mask_overlay(dataset: xr.Dataset, alpha: float = 0.4) -> Image.Image:
    """Generate water mask overlay in blue.

    Args:
        dataset: Dataset containing water mask
        alpha: Alpha transparency (0-1)

    Returns:
        Blue water mask overlay
    """
    if "water_mask" not in dataset:
        raise ValueError("Dataset does not contain water_mask")

    return generate_mask_overlay(
        dataset["water_mask"], color=(0, 100, 255), alpha=alpha  # Light blue
    )


def generate_cloud_mask_overlay(dataset: xr.Dataset, alpha: float = 0.5) -> Image.Image:
    """Generate cloud mask overlay in gray.

    Args:
        dataset: Dataset containing cloud mask
        alpha: Alpha transparency (0-1)

    Returns:
        Gray cloud mask overlay
    """
    if "cloud_mask" not in dataset:
        raise ValueError("Dataset does not contain cloud_mask")

    return generate_mask_overlay(
        dataset["cloud_mask"], color=(128, 128, 128), alpha=alpha  # Gray
    )


def generate_biomass_heatmap(
    biomass_data: xr.DataArray,
    colormap: str = "hot",
    min_biomass: float | None = None,
    max_biomass: float | None = None,
) -> Image.Image:
    """Generate biomass density heatmap.

    Args:
        biomass_data: Biomass values (kg/ha)
        colormap: Matplotlib colormap for heatmap
        min_biomass: Minimum biomass for color scaling
        max_biomass: Maximum biomass for color scaling

    Returns:
        Biomass heatmap image
    """
    values = biomass_data.values

    # Handle zero/negative values
    valid_mask = (values > 0) & (~np.isnan(values))

    if not np.any(valid_mask):
        # No valid biomass data, return transparent image
        height, width = values.shape
        return array_to_image(np.zeros((height, width, 4), dtype=np.uint8))

    # Set custom scaling if provided
    if min_biomass is not None and max_biomass is not None:
        normalized = np.clip((values - min_biomass) / (max_biomass - min_biomass), 0, 1)
    else:
        # Auto-scale based on valid data
        valid_values = values[valid_mask]
        min_val: float = float(np.min(valid_values))
        max_val: float = float(np.max(valid_values))

        if max_val == min_val:
            normalized = np.ones_like(values) * 0.5
        else:
            normalized = np.zeros_like(values)
            normalized[valid_mask] = (valid_values - min_val) / (max_val - min_val)

    # Apply colormap
    colored_array = apply_colormap(normalized, colormap)

    # Set alpha to 0 for invalid areas
    colored_array[~valid_mask, 3] = 0

    return array_to_image(colored_array)


def create_colored_mask(
    mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.6
) -> np.ndarray:
    """Create colored RGBA array from boolean mask.

    Args:
        mask: Boolean mask array
        color: RGB color tuple (0-255)
        alpha: Alpha transparency (0-1)

    Returns:
        RGBA array
    """
    height, width = mask.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Set RGB channels where mask is True
    rgba[mask, 0] = color[0]  # Red
    rgba[mask, 1] = color[1]  # Green
    rgba[mask, 2] = color[2]  # Blue
    rgba[mask, 3] = int(alpha * 255)  # Alpha

    return rgba


def generate_confidence_overlay(
    confidence_data: xr.DataArray,
    threshold: float = 0.5,
    low_color: tuple[int, int, int] = (255, 0, 0),  # Red for low confidence
    high_color: tuple[int, int, int] = (0, 255, 0),  # Green for high confidence
) -> Image.Image:
    """Generate confidence level overlay.

    Args:
        confidence_data: Confidence values (0-1)
        threshold: Confidence threshold for color change
        low_color: RGB color for low confidence areas
        high_color: RGB color for high confidence areas

    Returns:
        Confidence overlay image
    """
    values = confidence_data.values
    height, width = values.shape

    # Create RGBA array
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Define masks
    valid_mask = ~np.isnan(values)
    low_conf_mask = valid_mask & (values < threshold)
    high_conf_mask = valid_mask & (values >= threshold)

    # Set colors based on confidence
    rgba[low_conf_mask, :3] = low_color
    rgba[high_conf_mask, :3] = high_color

    # Set alpha based on confidence level (more transparent = less confident)
    alpha_values: np.ndarray = (values * 255).astype(np.uint8)
    rgba[valid_mask, 3] = alpha_values[valid_mask]

    return array_to_image(rgba)


def generate_change_detection_overlay(
    before_data: xr.DataArray,
    after_data: xr.DataArray,
    threshold: float = 0.1,
    decrease_color: tuple[int, int, int] = (255, 0, 0),  # Red for decrease
    increase_color: tuple[int, int, int] = (0, 255, 0),  # Green for increase
    alpha: float = 0.7,
) -> Image.Image:
    """Generate change detection overlay showing biomass/kelp changes.

    Args:
        before_data: Baseline data
        after_data: Comparison data
        threshold: Minimum change threshold
        decrease_color: Color for significant decreases
        increase_color: Color for significant increases
        alpha: Overlay transparency

    Returns:
        Change detection overlay
    """
    # Calculate difference
    diff = after_data.values - before_data.values

    height, width = diff.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Define change masks
    significant_decrease = diff < -threshold
    significant_increase = diff > threshold

    # Set colors
    rgba[significant_decrease, :3] = decrease_color
    rgba[significant_increase, :3] = increase_color

    # Set alpha for areas with significant change
    change_mask = significant_decrease | significant_increase
    rgba[change_mask, 3] = int(alpha * 255)

    return array_to_image(rgba)


def generate_multi_class_overlay(
    classification_data: xr.DataArray, class_colors: dict, alpha: float = 0.6
) -> Image.Image:
    """Generate multi-class classification overlay.

    Args:
        classification_data: Classification data with integer class labels
        class_colors: Dict mapping class labels to RGB colors
        alpha: Overlay transparency

    Returns:
        Multi-class overlay image
    """
    values = classification_data.values
    height, width = values.shape

    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Apply colors for each class
    for class_label, color in class_colors.items():
        class_mask = values == class_label
        rgba[class_mask, :3] = color
        rgba[class_mask, 3] = int(alpha * 255)

    return array_to_image(rgba)


def generate_gradient_overlay(
    data: xr.DataArray,
    start_color: tuple[int, int, int] = (0, 0, 255),  # Blue
    end_color: tuple[int, int, int] = (255, 0, 0),  # Red
    alpha_gradient: bool = True,
) -> Image.Image:
    """Generate gradient overlay with custom color interpolation.

    Args:
        data: Input data for gradient mapping
        start_color: RGB color for minimum values
        end_color: RGB color for maximum values
        alpha_gradient: Whether to apply alpha gradient based on values

    Returns:
        Gradient overlay image
    """
    # Normalize data to 0-1
    normalized = normalize_to_0_1(data)

    height, width = normalized.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Interpolate colors
    for i in range(3):  # RGB channels
        rgba[:, :, i] = (
            start_color[i] * (1 - normalized) + end_color[i] * normalized
        ).astype(np.uint8)

    # Set alpha
    if alpha_gradient:
        # Alpha based on data values (higher values = more opaque)
        rgba[:, :, 3] = (normalized * 255).astype(np.uint8)
    else:
        # Uniform alpha where data is valid
        valid_mask = ~np.isnan(data.values)
        rgba[valid_mask, 3] = 180  # 70% opacity

    return array_to_image(rgba)
