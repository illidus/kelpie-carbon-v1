"""Utility functions for satellite imagery processing."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image


def normalize_band(
    band: xr.DataArray, percentile_clip: tuple[float, float] = (2, 98)
) -> np.ndarray:
    """Normalize satellite band to 0-255 range with percentile clipping.

    Args:
        band: Input satellite band data
        percentile_clip: Lower and upper percentiles for clipping

    Returns:
        Normalized array in 0-255 range
    """
    # Remove NaN values
    valid_data = band.values[~np.isnan(band.values)]

    if len(valid_data) == 0:
        return np.zeros_like(band.values, dtype=np.uint8)

    # Calculate percentile clipping values
    p_low: float = float(np.percentile(valid_data, percentile_clip[0]))
    p_high: float = float(np.percentile(valid_data, percentile_clip[1]))

    # Clip and normalize
    clipped = np.clip(band.values, p_low, p_high)

    # Handle edge case where p_high == p_low
    if p_high == p_low:
        normalized = np.zeros_like(clipped)
    else:
        normalized = (clipped - p_low) / (p_high - p_low)

    # Handle any remaining NaN or inf values before casting
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)

    # Convert to 0-255 range
    return (normalized * 255).astype(np.uint8)


def normalize_to_0_1(data: xr.DataArray) -> np.ndarray:
    """Normalize data to 0-1 range for colormap application.

    Args:
        data: Input data array

    Returns:
        Normalized array in 0-1 range
    """
    values = data.values
    valid_mask = ~np.isnan(values)

    if not np.any(valid_mask):
        return np.zeros_like(values)

    valid_data = values[valid_mask]
    min_val: float = float(np.min(valid_data))
    max_val: float = float(np.max(valid_data))

    if max_val == min_val:
        return np.zeros_like(values)

    normalized = np.zeros_like(values)
    normalized[valid_mask] = (valid_data - min_val) / (max_val - min_val)

    return normalized


def apply_colormap(normalized_data: np.ndarray, colormap: str = "RdYlGn") -> np.ndarray:
    """Apply matplotlib colormap to normalized data.

    Args:
        normalized_data: Data normalized to 0-1 range
        colormap: Matplotlib colormap name

    Returns:
        RGBA array with applied colormap
    """
    # Get colormap
    cmap = plt.get_cmap(colormap)

    # Apply colormap
    colored = cmap(normalized_data)

    # Convert to 0-255 range
    return (colored * 255).astype(np.uint8)


def array_to_image(array: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image.

    Args:
        array: Input array (H, W) or (H, W, C)

    Returns:
        PIL Image object
    """
    if array.ndim == 2:
        # Grayscale
        return Image.fromarray(array, mode="L")
    elif array.ndim == 3:
        if array.shape[2] == 3:
            # RGB
            return Image.fromarray(array, mode="RGB")
        elif array.shape[2] == 4:
            # RGBA
            return Image.fromarray(array, mode="RGBA")

    raise ValueError(f"Unsupported array shape: {array.shape}")


def get_image_bounds(dataset: xr.Dataset) -> tuple[float, float, float, float]:
    """Get geographical bounds of the dataset in WGS84 coordinates.

    Args:
        dataset: Input xarray dataset

    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat) in WGS84
    """
    # First check if bbox is stored in dataset attributes
    if hasattr(dataset, "attrs") and "bbox" in dataset.attrs:
        bbox = dataset.attrs["bbox"]
        return tuple(bbox)  # bbox is already [min_lon, min_lat, max_lon, max_lat]

    # Try different coordinate names, prioritizing geographic coordinates
    x_coords: np.ndarray = np.array([])  # Initialize to prevent Never type
    y_coords: np.ndarray = np.array([])  # Initialize to prevent Never type

    if "longitude" in dataset.coords and "latitude" in dataset.coords:
        x_coords = dataset.coords["longitude"].values
        y_coords = dataset.coords["latitude"].values
    elif "lon" in dataset.coords and "lat" in dataset.coords:
        x_coords = dataset.coords["lon"].values
        y_coords = dataset.coords["lat"].values
    elif "x" in dataset.coords and "y" in dataset.coords:
        x_coords = dataset.coords["x"].values
        y_coords = dataset.coords["y"].values

        # Check if coordinates look like geographic (lat/lon) or projected (UTM)
        x_range: float = float(np.max(x_coords)) - float(np.min(x_coords))
        y_range: float = float(np.max(y_coords)) - float(np.min(y_coords))

        # If coordinates are in a very large range, they're likely UTM
        if x_range > 1000 or y_range > 1000:
            # These are likely UTM coordinates - convert to degrees roughly
            # This is a simple approximation for display purposes
            center_x: float = float(np.mean(x_coords))
            center_y: float = float(np.mean(y_coords))

            # Rough conversion from UTM to lat/lon (approximate)
            # UTM Zone calculation (assuming Northern Hemisphere)
            if center_x > 100000:  # Typical UTM easting
                zone = int((center_x / 1000000) + 31)  # Rough zone estimation
                central_meridian = (
                    (zone - 1) * 6 - 180 + 3
                )  # Central meridian of UTM zone

                # Convert UTM to lat/lon (very rough approximation)
                lon_offset = (center_x - 500000) / 111320  # Rough conversion
                lat_offset = center_y / 110540  # Rough conversion

                center_lon = central_meridian + lon_offset
                center_lat = lat_offset / 1000  # Scale down large UTM northing

                # Create bounds around center point
                lon_range = x_range / 111320 / 2  # Half range
                lat_range = y_range / 110540 / 1000 / 2  # Half range, scaled

                min_lon = center_lon - lon_range
                max_lon = center_lon + lon_range
                min_lat = center_lat - lat_range
                max_lat = center_lat + lat_range

                return (min_lon, min_lat, max_lon, max_lat)
    else:
        # Fallback - return world bounds
        return (-180, -90, 180, 90)

    # Standard case - coordinates are already in geographic format
    min_lon, max_lon = float(np.min(x_coords)), float(np.max(x_coords))
    min_lat, max_lat = float(np.min(y_coords)), float(np.max(y_coords))

    # Validate bounds are reasonable for lat/lon
    if min_lon < -180 or max_lon > 180 or min_lat < -90 or max_lat > 90:
        # Invalid geographic coordinates - return world bounds as fallback
        return (-180, -90, 180, 90)

    return (min_lon, min_lat, max_lon, max_lat)


def create_rgba_overlay(
    mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.6
) -> np.ndarray:
    """Create RGBA overlay from boolean mask.

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


def enhance_contrast(array: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction for contrast enhancement.

    Args:
        array: Input array (0-255)
        gamma: Gamma value (< 1 darkens, > 1 lightens)

    Returns:
        Enhanced array
    """
    normalized = array / 255.0
    enhanced = np.power(normalized, gamma)
    return (enhanced * 255).astype(np.uint8)


def calculate_histogram_stretch(
    data: np.ndarray, percent: float = 2.0
) -> tuple[float, float]:
    """Calculate histogram stretch values for enhanced visualization.

    Args:
        data: Input data array
        percent: Percentage for min/max clipping

    Returns:
        Tuple of (min_val, max_val) for stretching
    """
    valid_data = data[~np.isnan(data)]

    if len(valid_data) == 0:
        return (0.0, 1.0)

    min_val = np.percentile(valid_data, percent)
    max_val = np.percentile(valid_data, 100 - percent)

    return (float(min_val), float(max_val))
