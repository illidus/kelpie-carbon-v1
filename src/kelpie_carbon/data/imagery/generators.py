"""Image generators for satellite data visualization."""

import numpy as np
import xarray as xr
from PIL import Image

from .utils import apply_colormap, array_to_image, normalize_band, normalize_to_0_1


def generate_rgb_composite(
    dataset: xr.Dataset,
    red_band: str = "red",
    green_band: str | None = None,
    blue_band: str | None = None,
    contrast_enhance: bool = True,
) -> Image.Image:
    """Generate true-color RGB composite from Sentinel-2 bands.

    Args:
        dataset: Input satellite dataset
        red_band: Name of red band
        green_band: Name of green band (will simulate if None)
        blue_band: Name of blue band (will simulate if None)
        contrast_enhance: Whether to apply contrast enhancement

    Returns:
        PIL Image in RGB format
    """
    # Get red band
    if red_band not in dataset:
        raise ValueError(f"Red band '{red_band}' not found in dataset")

    red = normalize_band(dataset[red_band])

    # Get or simulate green band
    if green_band and green_band in dataset:
        green = normalize_band(dataset[green_band])
    else:
        # Simulate green using red and NIR if available
        if "nir" in dataset:
            green = normalize_band((dataset[red_band] + dataset["nir"]) / 2)
        else:
            green = red.copy()

    # Get or simulate blue band
    if blue_band and blue_band in dataset:
        blue = normalize_band(dataset[blue_band])
    else:
        # Simulate blue using red band (darker version)
        blue = (red * 0.7).astype(np.uint8)

    # Stack into RGB
    rgb_array = np.stack([red, green, blue], axis=-1)

    # Apply contrast enhancement if requested
    if contrast_enhance:
        from .utils import enhance_contrast

        rgb_array = enhance_contrast(rgb_array, gamma=0.8)

    return array_to_image(rgb_array)


def generate_false_color_composite(
    dataset: xr.Dataset,
    nir_band: str = "nir",
    red_band: str = "red",
    green_band: str = "red_edge",
) -> Image.Image:
    """Generate false-color composite (NIR-Red-Green) for vegetation enhancement.

    Args:
        dataset: Input satellite dataset
        nir_band: Name of NIR band (displayed as red)
        red_band: Name of red band (displayed as green)
        green_band: Name of band for blue channel

    Returns:
        PIL Image in RGB format
    """
    # Validate bands exist
    required_bands = [nir_band, red_band, green_band]
    missing_bands = [band for band in required_bands if band not in dataset]

    if missing_bands:
        raise ValueError(f"Missing bands for false-color composite: {missing_bands}")

    # Normalize bands
    nir_norm = normalize_band(dataset[nir_band])  # Display as red
    red_norm = normalize_band(dataset[red_band])  # Display as green
    green_norm = normalize_band(dataset[green_band])  # Display as blue

    # Stack into RGB (NIR, Red, Green -> Red, Green, Blue display)
    false_color_array = np.stack([nir_norm, red_norm, green_norm], axis=-1)

    return array_to_image(false_color_array)


def generate_spectral_visualization(
    index_data: xr.DataArray, colormap: str = "RdYlGn", title: str | None = None
) -> Image.Image:
    """Generate false-color visualization of spectral index.

    Args:
        index_data: Spectral index data (e.g., NDVI, FAI)
        colormap: Matplotlib colormap name
        title: Optional title for the visualization

    Returns:
        PIL Image with applied colormap
    """
    # Normalize to 0-1 range
    normalized = normalize_to_0_1(index_data)

    # Apply colormap
    colored_array = apply_colormap(normalized, colormap)

    # Convert to RGB (remove alpha channel)
    rgb_array = colored_array[:, :, :3]

    return array_to_image(rgb_array)


def generate_ndvi_visualization(dataset: xr.Dataset) -> Image.Image:
    """Generate NDVI visualization with standard green-to-red colormap.

    Args:
        dataset: Dataset containing NDVI data

    Returns:
        NDVI visualization image
    """
    if "ndvi" not in dataset:
        raise ValueError("Dataset does not contain NDVI data")

    return generate_spectral_visualization(
        dataset["ndvi"], colormap="RdYlGn", title="NDVI"
    )


def generate_fai_visualization(dataset: xr.Dataset) -> Image.Image:
    """Generate FAI (Floating Algae Index) visualization.

    Args:
        dataset: Dataset containing FAI data

    Returns:
        FAI visualization image
    """
    if "fai" not in dataset:
        raise ValueError("Dataset does not contain FAI data")

    return generate_spectral_visualization(
        dataset["fai"],
        colormap="plasma",  # Blue to yellow colormap for algae
        title="FAI",
    )


def generate_red_edge_ndvi_visualization(dataset: xr.Dataset) -> Image.Image:
    """Generate Red Edge NDVI visualization.

    Args:
        dataset: Dataset containing Red Edge NDVI data

    Returns:
        Red Edge NDVI visualization image
    """
    if "red_edge_ndvi" not in dataset:
        raise ValueError("Dataset does not contain Red Edge NDVI data")

    return generate_spectral_visualization(
        dataset["red_edge_ndvi"],
        colormap="viridis",  # Green-blue colormap
        title="Red Edge NDVI",
    )


def generate_ndre_visualization(dataset: xr.Dataset) -> Image.Image:
    """Generate NDRE (Normalized Difference Red Edge) visualization.

    Based on SKEMA research findings for enhanced submerged kelp detection.
    NDRE outperforms traditional NDVI by 18% for kelp area detection.

    Args:
        dataset: Dataset containing NDRE data

    Returns:
        NDRE visualization image with research-optimized colormap
    """
    if "ndre" not in dataset:
        raise ValueError("Dataset does not contain NDRE data")

    return generate_spectral_visualization(
        dataset["ndre"],
        colormap="RdYlGn",  # Red-Yellow-Green colormap (same as NDVI for comparison)
        title="NDRE (Enhanced Kelp Detection)",
    )


def generate_comparative_ndvi_ndre_visualization(dataset: xr.Dataset) -> Image.Image:
    """Generate side-by-side comparison of NDVI vs NDRE for research validation.

    Shows the improvement in kelp detection capability using red-edge bands.

    Args:
        dataset: Dataset containing both NDVI and NDRE data

    Returns:
        Composite image showing NDVI and NDRE side by side
    """
    if "ndvi" not in dataset or "ndre" not in dataset:
        raise ValueError("Dataset must contain both NDVI and NDRE data for comparison")

    import matplotlib.pyplot as plt

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot NDVI
    ndvi_data = normalize_to_0_1(dataset["ndvi"])
    im1 = ax1.imshow(ndvi_data, cmap="RdYlGn", vmin=0, vmax=1)
    ax1.set_title("Traditional NDVI\n(NIR-Red)/(NIR+Red)")
    ax1.axis("off")

    # Plot NDRE
    ndre_data = normalize_to_0_1(dataset["ndre"])
    im2 = ax2.imshow(ndre_data, cmap="RdYlGn", vmin=0, vmax=1)
    ax2.set_title("Enhanced NDRE\n(RedEdge-Red)/(RedEdge+Red)\n+18% kelp detection")
    ax2.axis("off")

    # Add colorbar
    plt.colorbar(
        im2,
        ax=[ax1, ax2],
        orientation="horizontal",
        pad=0.1,
        label="Vegetation Index Value",
    )

    plt.tight_layout()

    # Convert matplotlib figure to PIL Image
    fig.canvas.draw()
    # Use type casting to handle matplotlib canvas method compatibility
    from typing import Any, cast

    canvas_any = cast(Any, fig.canvas)
    buf = np.frombuffer(canvas_any.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return array_to_image(buf)


def generate_multi_band_composite(
    dataset: xr.Dataset, band_mapping: dict, normalize_bands: bool = True
) -> Image.Image:
    """Generate custom multi-band composite.

    Args:
        dataset: Input satellite dataset
        band_mapping: Dict mapping 'red', 'green', 'blue' to band names
        normalize_bands: Whether to normalize each band

    Returns:
        RGB composite image
    """
    required_channels = ["red", "green", "blue"]

    # Validate band mapping
    for channel in required_channels:
        if channel not in band_mapping:
            raise ValueError(f"Missing '{channel}' channel in band_mapping")

        band_name = band_mapping[channel]
        if band_name not in dataset:
            raise ValueError(f"Band '{band_name}' not found in dataset")

    # Extract and normalize bands
    bands = []
    for channel in required_channels:
        band_name = band_mapping[channel]

        if normalize_bands:
            band_data = normalize_band(dataset[band_name])
        else:
            # Assume data is already in 0-255 range
            band_data = dataset[band_name].values.astype(np.uint8)

        bands.append(band_data)

    # Stack into RGB
    rgb_array = np.stack(bands, axis=-1)

    return array_to_image(rgb_array)


def generate_stretched_composite(
    dataset: xr.Dataset, bands: list, stretch_percent: float = 2.0
) -> Image.Image:
    """Generate histogram-stretched composite for enhanced visualization.

    Args:
        dataset: Input satellite dataset
        bands: List of band names to use [red_band, green_band, blue_band]
        stretch_percent: Percentage for histogram stretching

    Returns:
        Histogram-stretched RGB image
    """
    if len(bands) != 3:
        raise ValueError("Must provide exactly 3 bands for RGB composite")

    # Validate bands exist
    missing_bands = [band for band in bands if band not in dataset]
    if missing_bands:
        raise ValueError(f"Missing bands: {missing_bands}")

    from .utils import calculate_histogram_stretch

    rgb_bands = []

    for band_name in bands:
        band_data = dataset[band_name].values

        # Calculate stretch values
        min_val, max_val = calculate_histogram_stretch(band_data, stretch_percent)

        # Apply stretch
        stretched = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
        stretched_uint8 = (stretched * 255).astype(np.uint8)

        rgb_bands.append(stretched_uint8)

    # Stack into RGB
    rgb_array = np.stack(rgb_bands, axis=-1)

    return array_to_image(rgb_array)
