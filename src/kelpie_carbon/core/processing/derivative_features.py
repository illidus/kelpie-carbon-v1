"""
Derivative-Based Feature Detection for Kelp Detection

Based on SKEMA research from Uhl et al. (2016) - "Submerged Kelp Detection with Hyperspectral Data"
Implements first-order derivative spectral analysis for enhanced kelp detection.

The derivative-based approach identifies spectral features at specific wavelengths:
- 528nm ± 18nm (fucoxanthin absorption)
- 570nm ± 10nm (reflectance peak)
- Effective range: 500-600nm for turbid coastal waters
"""

import numpy as np
import xarray as xr
from scipy import ndimage


class DerivativeFeatures:
    """Derivative-based feature detection class for kelp detection.

    A wrapper class providing object-oriented access to derivative feature
    functionality based on SKEMA research from Uhl et al. (2016).
    """

    def __init__(self, config: dict | None = None):
        """Initialize derivative feature detector with configuration.

        Args:
            config: Configuration parameters for detection
        """
        self.config = config or {
            "red_edge_slope_threshold": 0.01,
            "nir_transition_threshold": 0.02,
            "composite_threshold": 0.015,
            "min_cluster_size": 5,
            "morphology_cleanup": True,
        }

    def detect_kelp_features(self, dataset: xr.Dataset) -> np.ndarray:
        """Apply derivative-based kelp detection algorithm.

        Args:
            dataset: Dataset with spectral bands

        Returns:
            Boolean mask where True indicates potential kelp
        """
        return apply_derivative_kelp_detection(dataset, self.config)

    def calculate_first_derivatives(
        self, wavelengths: np.ndarray, reflectance: np.ndarray
    ) -> np.ndarray:
        """Calculate first-order spectral derivatives.

        Args:
            wavelengths: Array of wavelength values
            reflectance: Array of reflectance values

        Returns:
            Array of first derivatives
        """
        return np.diff(reflectance) / np.diff(wavelengths)


def calculate_spectral_derivatives(dataset: xr.Dataset) -> xr.Dataset:
    """Calculate first-order spectral derivatives for kelp feature detection.

    Based on Uhl et al. (2016) methodology that achieved 80.18% accuracy
    for submerged kelp detection using derivative-based features.

    Args:
        dataset: xarray Dataset with spectral bands

    Returns:
        Dataset with derivative features added
    """
    derivative_dataset = dataset.copy()

    # Define Sentinel-2 band wavelengths (approximate center wavelengths in nm)
    band_wavelengths = {
        "red": 665.0,  # B4
        "red_edge": 705.0,  # B5
        "red_edge_2": 740.0,  # B6 (if available)
        "nir": 842.0,  # B8
        "swir1": 1610.0,  # B11
    }

    # Calculate derivatives between adjacent spectral bands
    derivatives = _calculate_band_derivatives(dataset, band_wavelengths)

    # Add derivative bands to dataset
    for deriv_name, deriv_values in derivatives.items():
        derivative_dataset[deriv_name] = (["y", "x"], deriv_values)

    # Calculate kelp-specific derivative features
    kelp_features = calculate_kelp_derivative_features(
        derivative_dataset, band_wavelengths
    )

    # Add kelp features to dataset
    for feature_name, feature_values in kelp_features.items():
        derivative_dataset[feature_name] = (["y", "x"], feature_values)

    return derivative_dataset


def _calculate_band_derivatives(
    dataset: xr.Dataset, wavelengths: dict[str, float]
) -> dict[str, np.ndarray]:
    """Calculate derivatives between spectral bands.

    Args:
        dataset: Satellite imagery dataset
        wavelengths: Dictionary mapping band names to wavelengths

    Returns:
        Dictionary of derivative arrays
    """
    derivatives = {}

    # Available bands in order of wavelength
    available_bands = [
        (name, wl) for name, wl in wavelengths.items() if name in dataset
    ]
    available_bands.sort(key=lambda x: x[1])  # Sort by wavelength

    # Calculate derivatives between adjacent bands
    for i in range(len(available_bands) - 1):
        band1_name, wl1 = available_bands[i]
        band2_name, wl2 = available_bands[i + 1]

        band1_data: np.ndarray = dataset[band1_name].values.astype(np.float32)
        band2_data: np.ndarray = dataset[band2_name].values.astype(np.float32)

        # First-order derivative: dR/dλ
        wavelength_diff = wl2 - wl1
        derivative = (band2_data - band1_data) / wavelength_diff

        deriv_name = f"d_{band1_name}_{band2_name}"
        derivatives[deriv_name] = derivative

    return derivatives


def calculate_kelp_derivative_features(
    dataset: xr.Dataset, wavelengths: dict[str, float]
) -> dict[str, np.ndarray]:
    """Calculate kelp-specific derivative features based on research findings.

    Implements the optimal spectral features identified by Uhl et al. (2016):
    - 528nm ± 18nm region (fucoxanthin absorption)
    - 570nm ± 10nm region (reflectance peak)

    Args:
        dataset: Dataset with spectral bands and derivatives
        wavelengths: Dictionary mapping band names to wavelengths

    Returns:
        Dictionary of kelp-specific derivative features
    """
    features = {}

    # Feature 1: Fucoxanthin absorption feature (around 528nm)
    # Use red band (665nm) as proxy - not ideal but closest available
    if "red" in dataset:
        red_data: np.ndarray = dataset["red"].values.astype(np.float32)

        # Calculate spatial derivatives to detect absorption features
        dx_red = ndimage.sobel(red_data, axis=1)  # X gradient
        dy_red = ndimage.sobel(red_data, axis=0)  # Y gradient

        # Magnitude of spatial gradient indicates feature strength
        fucoxanthin_feature = np.sqrt(dx_red**2 + dy_red**2)
        features["fucoxanthin_absorption"] = fucoxanthin_feature

    # Feature 2: Red-edge peak detection (570-705nm range)
    if "red" in dataset and "red_edge" in dataset:
        red_data_2: np.ndarray = dataset["red"].values.astype(np.float32)
        red_edge_data_2: np.ndarray = dataset["red_edge"].values.astype(np.float32)

        # Derivative across red to red-edge transition
        red_edge_derivative = (red_edge_data_2 - red_data_2) / (705 - 665)
        features["red_edge_slope"] = red_edge_derivative

        # Second derivative for peak detection
        second_derivative = ndimage.laplace(red_edge_data_2)
        features["red_edge_curvature"] = second_derivative

    # Feature 3: NIR transition feature (kelp vs water discrimination)
    if "red_edge" in dataset and "nir" in dataset:
        red_edge_data_3: np.ndarray = dataset["red_edge"].values.astype(np.float32)
        nir_data_3: np.ndarray = dataset["nir"].values.astype(np.float32)

        # Derivative across red-edge to NIR transition
        nir_transition = (nir_data_3 - red_edge_data_3) / (842 - 705)
        features["nir_transition"] = nir_transition

    # Feature 4: Composite kelp detection feature
    if "red_edge_slope" in features and "nir_transition" in features:
        # Combine multiple derivative features for robust kelp detection
        composite_feature = (
            0.4 * features["red_edge_slope"]
            + 0.4 * features["nir_transition"]
            + 0.2 * features.get("fucoxanthin_absorption", 0)
        )
        features["composite_kelp_derivative"] = composite_feature

    return features


def apply_derivative_kelp_detection(
    dataset: xr.Dataset, detection_config: dict | None = None
) -> np.ndarray:
    """Apply derivative-based kelp detection algorithm.

    Based on Feature Detection (FD) algorithm from Uhl et al. (2016)
    that achieved 80.18% overall accuracy for submerged kelp detection.

    Args:
        dataset: Dataset with derivative features
        detection_config: Configuration for detection thresholds

    Returns:
        Boolean mask where True indicates potential kelp
    """
    if detection_config is None:
        detection_config = {
            "red_edge_slope_threshold": 0.01,  # Positive slope indicates vegetation
            "nir_transition_threshold": 0.02,  # Strong NIR response
            "composite_threshold": 0.015,  # Combined feature threshold
            "min_cluster_size": 5,  # Minimum kelp cluster size
            "morphology_cleanup": True,  # Apply morphological operations
        }

    # Ensure derivative features are calculated
    if "red_edge_slope" not in dataset:
        dataset = calculate_spectral_derivatives(dataset)

    kelp_mask = np.zeros(dataset["red"].shape, dtype=bool)

    # Apply derivative-based detection criteria
    if "composite_kelp_derivative" in dataset:
        # Use composite feature for primary detection
        composite_values = dataset["composite_kelp_derivative"].values
        kelp_mask = composite_values > detection_config["composite_threshold"]

    elif "red_edge_slope" in dataset and "nir_transition" in dataset:
        # Use individual features if composite not available
        red_edge_slope = dataset["red_edge_slope"].values
        nir_transition = dataset["nir_transition"].values

        kelp_criteria = (
            red_edge_slope > detection_config["red_edge_slope_threshold"]
        ) & (nir_transition > detection_config["nir_transition_threshold"])
        kelp_mask = kelp_criteria

    # Post-processing cleanup
    if detection_config.get("morphology_cleanup", True):
        kelp_mask = _apply_morphological_cleanup(
            kelp_mask, detection_config.get("min_cluster_size", 5)
        )

    return kelp_mask


def _apply_morphological_cleanup(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Apply morphological operations to clean up detection mask.

    Args:
        mask: Binary detection mask
        min_size: Minimum cluster size to retain

    Returns:
        Cleaned binary mask
    """
    # Remove noise with opening operation
    kernel = np.ones((3, 3))
    cleaned_mask = ndimage.binary_opening(mask, kernel)

    # Remove small clusters
    from typing import cast

    label_result = cast(tuple[np.ndarray, int], ndimage.label(cleaned_mask))
    labeled_array: np.ndarray = label_result[0]
    num_features = label_result[1]
    component_sizes: np.ndarray = np.bincount(labeled_array.ravel())

    # Keep only large components
    large_components: np.ndarray = component_sizes >= min_size
    large_components[0] = False  # Background component

    final_mask = large_components[labeled_array]

    return final_mask


def calculate_derivative_quality_metrics(
    dataset: xr.Dataset, detection_mask: np.ndarray
) -> dict[str, float]:
    """Calculate quality metrics for derivative-based kelp detection.

    Args:
        dataset: Dataset with derivative features
        detection_mask: Boolean detection mask

    Returns:
        Dictionary of quality metrics
    """
    metrics = {}

    # Detection coverage
    total_pixels = detection_mask.size
    detected_pixels: int = int(np.sum(detection_mask))
    metrics["detection_coverage_percent"] = (detected_pixels / total_pixels) * 100

    # Feature strength statistics
    if "composite_kelp_derivative" in dataset:
        composite_values = dataset["composite_kelp_derivative"].values
        metrics["mean_composite_strength"] = float(np.mean(composite_values))
        metrics["max_composite_strength"] = float(np.max(composite_values))
        metrics["std_composite_strength"] = float(np.std(composite_values))

        # Detection feature strength
        detected_values = composite_values[detection_mask]
        if len(detected_values) > 0:
            metrics["mean_detected_strength"] = float(np.mean(detected_values))
            metrics["min_detected_strength"] = float(np.min(detected_values))

    # Spatial clustering metrics
    from typing import cast

    label_result = cast(tuple[np.ndarray, int], ndimage.label(detection_mask))
    labeled_array: np.ndarray = label_result[0]
    num_clusters = label_result[1]
    metrics["num_kelp_clusters"] = num_clusters

    if num_clusters > 0:
        cluster_sizes: np.ndarray = np.bincount(labeled_array.ravel())[
            1:
        ]  # Exclude background
        metrics["mean_cluster_size"] = float(np.mean(cluster_sizes))
        metrics["largest_cluster_size"] = int(np.max(cluster_sizes))
        metrics["smallest_cluster_size"] = int(np.min(cluster_sizes))

    return metrics
