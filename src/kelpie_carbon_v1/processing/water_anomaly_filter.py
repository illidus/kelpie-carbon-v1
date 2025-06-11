"""
Water Anomaly Filter (WAF) Implementation

Based on SKEMA research from Uhl et al. (2016) - "Submerged Kelp Detection with Hyperspectral Data"
Removes sunglint and surface artifacts from satellite imagery to improve submerged kelp detection.

The WAF algorithm identifies and filters out water surface anomalies that can interfere with
kelp detection algorithms, particularly important for submerged kelp analysis.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from scipy import ndimage


class WaterAnomalyFilter:
    """Water Anomaly Filter (WAF) class for kelp detection.

    A wrapper class providing object-oriented access to WAF functionality
    based on SKEMA research from Uhl et al. (2016).
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize WAF with configuration parameters.

        Args:
            config: Configuration parameters for WAF algorithm
        """
        self.config = config or {
            "sunglint_threshold": 0.15,
            "kernel_size": 5,
            "spectral_smoothing": True,
            "artifact_fill_method": "interpolation",
        }

    def apply_filter(self, dataset: xr.Dataset) -> xr.Dataset:
        """Apply Water Anomaly Filter to satellite imagery dataset.

        Args:
            dataset: xarray Dataset with satellite bands

        Returns:
            Filtered xarray Dataset with reduced surface artifacts
        """
        return apply_water_anomaly_filter(dataset, self.config)

    def detect_sunglint(self, dataset: xr.Dataset) -> np.ndarray:
        """Detect sunglint areas using spectral characteristics.

        Args:
            dataset: Satellite imagery dataset

        Returns:
            Boolean mask where True indicates sunglint
        """
        return _detect_sunglint(dataset, self.config["sunglint_threshold"])


def apply_water_anomaly_filter(
    dataset: xr.Dataset, waf_config: Optional[Dict] = None
) -> xr.Dataset:
    """Apply Water Anomaly Filter to satellite imagery dataset.

    Removes sunglint and surface artifacts that can interfere with kelp detection.
    Based on Uhl et al. (2016) methodology for improved submerged kelp detection.

    Args:
        dataset: xarray Dataset with satellite bands
        waf_config: Configuration parameters for WAF algorithm

    Returns:
        Filtered xarray Dataset with reduced surface artifacts
    """
    if waf_config is None:
        waf_config = {
            "sunglint_threshold": 0.15,  # Threshold for sunglint detection
            "kernel_size": 5,  # Morphological operation kernel size
            "spectral_smoothing": True,  # Apply spectral smoothing
            "artifact_fill_method": "interpolation",  # How to fill filtered areas
        }

    # Create a copy to avoid modifying original data
    filtered_dataset = dataset.copy()

    # Step 1: Detect sunglint and surface artifacts
    sunglint_mask = _detect_sunglint(dataset, waf_config["sunglint_threshold"])

    # Step 2: Identify other surface anomalies
    surface_anomalies = _detect_surface_anomalies(dataset, waf_config["kernel_size"])

    # Step 3: Combine masks for comprehensive filtering
    artifact_mask = sunglint_mask | surface_anomalies

    # Step 4: Apply filtering to spectral bands
    for band in ["red", "red_edge", "nir", "swir1"]:
        if band in filtered_dataset:
            filtered_dataset[band] = _apply_artifact_filter(
                filtered_dataset[band],
                artifact_mask,
                waf_config["artifact_fill_method"],
            )

    # Step 5: Optional spectral smoothing
    if waf_config.get("spectral_smoothing", False):
        filtered_dataset = _apply_spectral_smoothing(filtered_dataset)

    # Add WAF quality mask to dataset
    filtered_dataset["waf_mask"] = (["y", "x"], (~artifact_mask).astype(np.uint8))

    return filtered_dataset


def _detect_sunglint(dataset: xr.Dataset, threshold: float) -> np.ndarray:
    """Detect sunglint areas using spectral characteristics.

    Sunglint appears as high reflectance across all visible bands.

    Args:
        dataset: Satellite imagery dataset
        threshold: Reflectance threshold for sunglint detection

    Returns:
        Boolean mask where True indicates sunglint
    """
    # Sunglint shows high reflectance in visible bands
    red = dataset["red"].values
    red_edge = dataset["red_edge"].values
    nir = dataset["nir"].values

    # High reflectance across multiple bands indicates sunglint
    high_visible = red > threshold
    high_red_edge = red_edge > threshold
    high_nir = nir > threshold

    # Sunglint typically affects all bands similarly
    sunglint = high_visible & high_red_edge & high_nir

    # Morphological cleanup to remove isolated pixels
    kernel = np.ones((3, 3))
    sunglint = ndimage.binary_opening(sunglint, kernel)

    return sunglint


def _detect_surface_anomalies(dataset: xr.Dataset, kernel_size: int) -> np.ndarray:
    """Detect other surface anomalies like foam, debris, or glare.

    Uses spatial variance analysis to identify unusual surface patterns.

    Args:
        dataset: Satellite imagery dataset
        kernel_size: Size of analysis kernel

    Returns:
        Boolean mask where True indicates surface anomalies
    """
    # Use NIR band for surface anomaly detection (sensitive to surface conditions)
    nir: np.ndarray = dataset["nir"].values.astype(np.float32)

    # Calculate local variance to detect unusual patterns
    kernel = np.ones((kernel_size, kernel_size))
    local_mean = ndimage.uniform_filter(nir, size=kernel_size)
    local_variance = (
        ndimage.uniform_filter(nir**2, size=kernel_size) - local_mean**2
    )

    # High variance indicates surface disruption
    variance_threshold = np.percentile(local_variance[local_variance > 0], 95)
    high_variance = local_variance > variance_threshold

    # Also check for unusually high NIR values (foam, whitecaps)
    nir_threshold = np.percentile(nir, 98)
    high_nir = nir > nir_threshold

    # Combine variance and high reflectance criteria
    surface_anomalies = high_variance | high_nir

    # Clean up with morphological operations
    kernel = np.ones((3, 3))
    surface_anomalies = ndimage.binary_opening(surface_anomalies, kernel)
    surface_anomalies = ndimage.binary_closing(surface_anomalies, kernel)

    return surface_anomalies


def _apply_artifact_filter(
    band_data: xr.DataArray, artifact_mask: np.ndarray, fill_method: str
) -> xr.DataArray:
    """Apply filtering to remove artifacts from spectral band.

    Args:
        band_data: Spectral band data
        artifact_mask: Boolean mask of artifacts to remove
        fill_method: Method for filling filtered areas

    Returns:
        Filtered band data
    """
    filtered_data = band_data.copy()

    if fill_method == "interpolation":
        # Use spatial interpolation to fill artifact areas
        values = filtered_data.values.copy()

        # Simple interpolation using surrounding valid pixels
        for i in range(1, values.shape[0] - 1):
            for j in range(1, values.shape[1] - 1):
                if artifact_mask[i, j]:
                    # Get surrounding valid pixels
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if not artifact_mask[i + di, j + dj]:
                                neighbors.append(values[i + di, j + dj])

                    if neighbors:
                        values[i, j] = np.mean(neighbors)

        filtered_data.values = values

    elif fill_method == "nan":
        # Set artifact areas to NaN
        filtered_data = filtered_data.where(~artifact_mask, np.nan)

    elif fill_method == "median":
        # Replace with local median of valid pixels
        values = filtered_data.values.copy()
        local_median = ndimage.median_filter(values, size=5)
        values[artifact_mask] = local_median[artifact_mask]
        filtered_data.values = values

    return filtered_data


def _apply_spectral_smoothing(dataset: xr.Dataset) -> xr.Dataset:
    """Apply spatial smoothing to reduce noise in spectral bands.

    Args:
        dataset: Satellite imagery dataset

    Returns:
        Smoothed dataset
    """
    smoothed_dataset = dataset.copy()

    # Apply gentle Gaussian smoothing to reduce noise
    sigma = 0.8  # Small sigma for minimal smoothing

    for band in ["red", "red_edge", "nir", "swir1"]:
        if band in smoothed_dataset:
            values = smoothed_dataset[band].values
            smoothed_values = ndimage.gaussian_filter(values, sigma=sigma)
            smoothed_dataset[band].values = smoothed_values

    return smoothed_dataset


def calculate_waf_quality_metrics(dataset: xr.Dataset) -> Dict[str, float]:
    """Calculate quality metrics for WAF filtering results.

    Args:
        dataset: Dataset with WAF mask applied

    Returns:
        Dictionary of quality metrics
    """
    if "waf_mask" not in dataset:
        return {"error": 1.0}  # Use numeric error code instead of string

    waf_mask = dataset["waf_mask"].values
    total_pixels = waf_mask.size
    valid_pixels: int = int(np.sum(waf_mask))

    metrics = {
        "valid_pixel_percentage": (valid_pixels / total_pixels) * 100,
        "filtered_pixel_percentage": ((total_pixels - valid_pixels) / total_pixels)
        * 100,
        "total_pixels": total_pixels,
        "valid_pixels": int(valid_pixels),
        "filtered_pixels": int(total_pixels - valid_pixels),
    }

    return metrics
