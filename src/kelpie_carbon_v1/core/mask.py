"""Advanced masking functions for satellite imagery."""

from typing import Dict, Optional

import numpy as np
import xarray as xr
from scipy import ndimage

# Import SKEMA processing modules  
from ..processing.water_anomaly_filter import apply_water_anomaly_filter
from ..processing.derivative_features import (
    calculate_spectral_derivatives,
    apply_derivative_kelp_detection
)


def apply_mask(dataset: xr.Dataset, mask_config: Optional[Dict] = None) -> xr.Dataset:
    """Apply comprehensive masking to satellite dataset.

    Args:
        dataset: xarray Dataset with satellite bands
        mask_config: Configuration for masking parameters

    Returns:
        Dataset with masks applied and invalid data marked as NaN
    """
    if mask_config is None:
        mask_config = {
            "cloud_threshold": 0.5,
            "water_ndwi_threshold": 0.0,  # More permissive water detection
            "kelp_fai_threshold": -0.01,
            "apply_morphology": True,
            "min_kelp_cluster_size": 5,  # Smaller minimum size
        }

    # Create composite mask
    cloud_mask = create_cloud_mask(dataset, mask_config["cloud_threshold"])
    water_mask = create_water_mask(dataset, mask_config["water_ndwi_threshold"])
    kelp_mask = create_kelp_detection_mask(dataset, mask_config)

    # Combine masks - Modified logic for kelp detection
    valid_mask = ~cloud_mask  # Valid where not cloudy

    # For kelp detection, we don't require strict water detection
    # Kelp can be detected in areas with moderate water signatures
    water_or_coastal = water_mask | (
        kelp_mask & valid_mask
    )  # Include kelp areas as water-like
    kelp_pixels = kelp_mask & valid_mask  # Kelp pixels in valid (non-cloud) areas

    # Create masked dataset
    masked_dataset = dataset.copy()

    # Apply masks to all bands
    for var in ["red", "red_edge", "nir", "swir1"]:
        if var in masked_dataset:
            # Set invalid pixels to NaN
            masked_dataset[var] = masked_dataset[var].where(valid_mask, np.nan)

    # Add mask layers to dataset
    masked_dataset["cloud_mask"] = (["y", "x"], cloud_mask.astype(np.uint8))
    masked_dataset["water_mask"] = (["y", "x"], water_or_coastal.astype(np.uint8))
    masked_dataset["kelp_mask"] = (["y", "x"], kelp_pixels.astype(np.uint8))
    masked_dataset["valid_mask"] = (["y", "x"], valid_mask.astype(np.uint8))

    return masked_dataset


def create_cloud_mask(dataset: xr.Dataset, threshold: float = 0.5) -> np.ndarray:
    """Create cloud mask from cloud probability data.

    Args:
        dataset: xarray Dataset containing cloud_mask band
        threshold: Threshold for cloud detection (0-1)

    Returns:
        Boolean array where True indicates clouds
    """
    if "cloud_mask" not in dataset:
        # If no cloud mask available, create basic cloud detection
        return _create_basic_cloud_mask(dataset)

    cloud_prob = dataset["cloud_mask"].values
    cloud_mask = cloud_prob > threshold

    # Apply morphological operations to clean up mask
    kernel = np.ones((3, 3))
    cloud_mask = ndimage.binary_closing(cloud_mask, kernel)
    cloud_mask = ndimage.binary_opening(cloud_mask, kernel)

    # Add cloud shadow detection for more comprehensive masking
    shadow_mask = _detect_cloud_shadows(dataset, cloud_mask)
    combined_mask = cloud_mask | shadow_mask

    return combined_mask


def create_water_mask(dataset: xr.Dataset, ndwi_threshold: float = 0.3) -> np.ndarray:
    """Create water mask using Normalized Difference Water Index (NDWI).

    NDWI = (Green - NIR) / (Green + NIR)
    For Sentinel-2 without green band, use: (Red_Edge - NIR) / (Red_Edge + NIR)

    Args:
        dataset: xarray Dataset with spectral bands
        ndwi_threshold: Threshold for water detection

    Returns:
        Boolean array where True indicates water
    """
    red_edge: np.ndarray = dataset["red_edge"].values.astype(np.float32)
    nir: np.ndarray = dataset["nir"].values.astype(np.float32)

    # Calculate modified NDWI
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (red_edge - nir) / (red_edge + nir)
        ndwi = np.nan_to_num(ndwi, nan=0.0)

    water_mask = ndwi > ndwi_threshold

    # Clean up mask with morphological operations
    kernel = np.ones((5, 5))
    water_mask = ndimage.binary_closing(water_mask, kernel)
    water_mask = ndimage.binary_opening(water_mask, kernel)

    return water_mask


def create_kelp_detection_mask(dataset: xr.Dataset, config: Dict) -> np.ndarray:
    """Create kelp detection mask using multiple spectral indices.

    Args:
        dataset: xarray Dataset with spectral bands
        config: Configuration dictionary with thresholds

    Returns:
        Boolean array where True indicates potential kelp
    """
    # Calculate Floating Algae Index (FAI)
    fai = calculate_fai(dataset)

    # Calculate Red Edge NDVI
    red_edge_ndvi = calculate_red_edge_ndvi(dataset)

    # Kelp detection criteria
    kelp_fai = fai > config["kelp_fai_threshold"]
    kelp_ndvi = red_edge_ndvi > 0.1  # Vegetation threshold

    # Combine criteria
    kelp_mask = kelp_fai & kelp_ndvi

    if config.get("apply_morphology", True):
        # Apply morphological operations
        kernel = np.ones((3, 3))
        kelp_mask = ndimage.binary_opening(kelp_mask, kernel)

        # Remove small clusters
        min_size = config.get("min_kelp_cluster_size", 10)
        kelp_mask = remove_small_objects(kelp_mask, min_size)

    return kelp_mask


def calculate_fai(dataset: xr.Dataset) -> np.ndarray:
    """Calculate Floating Algae Index (FAI).

    FAI = NIR - (Red + (SWIR1 - Red) * (λ_NIR - λ_Red) / (λ_SWIR1 - λ_Red))

    Args:
        dataset: xarray Dataset with spectral bands

    Returns:
        FAI values as numpy array
    """
    red: np.ndarray = dataset["red"].values.astype(np.float32)
    nir: np.ndarray = dataset["nir"].values.astype(np.float32)
    swir1: np.ndarray = dataset["swir1"].values.astype(np.float32)

    # Sentinel-2 wavelengths (nm)
    lambda_red = 665
    lambda_nir = 842
    lambda_swir1 = 1610

    # Calculate FAI
    with np.errstate(divide="ignore", invalid="ignore"):
        fai = nir - (
            red
            + (swir1 - red) * (lambda_nir - lambda_red) / (lambda_swir1 - lambda_red)
        )
        fai = np.nan_to_num(fai, nan=0.0)

    return fai


def calculate_red_edge_ndvi(dataset: xr.Dataset) -> np.ndarray:
    """Calculate Red Edge NDVI for kelp detection.

    Red Edge NDVI = (NIR - Red_Edge) / (NIR + Red_Edge)

    Args:
        dataset: xarray Dataset with spectral bands

    Returns:
        Red Edge NDVI values as numpy array
    """
    red_edge: np.ndarray = dataset["red_edge"].values.astype(np.float32)
    nir: np.ndarray = dataset["nir"].values.astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        red_edge_ndvi = (nir - red_edge) / (nir + red_edge)
        red_edge_ndvi = np.nan_to_num(red_edge_ndvi, nan=0.0)

    return red_edge_ndvi


def calculate_ndre(dataset: xr.Dataset) -> np.ndarray:
    """Calculate Normalized Difference Red Edge Index (NDRE) for enhanced submerged kelp detection.

    NDRE = (RedEdge - Red) / (RedEdge + Red)

    Based on Timmer et al. (2022) research findings:
    - Uses 740nm red-edge band (red_edge_2) for optimal submerged kelp detection
    - Outperforms NDVI by detecting 18% more kelp area
    - Detection depth: 90-100cm vs 30-50cm for NDVI
    - Superior performance for submerged kelp canopy mapping

    Args:
        dataset: xarray Dataset with spectral bands including red_edge_2 (740nm)

    Returns:
        NDRE values as numpy array (-1 to 1 range)
    """
    # Use optimal 740nm red-edge band if available, fallback to 705nm
    red_edge: np.ndarray
    if "red_edge_2" in dataset:
        red_edge = dataset["red_edge_2"].values.astype(
            np.float32
        )  # B06 - 740nm (optimal)
    else:
        red_edge = dataset["red_edge"].values.astype(
            np.float32
        )  # B05 - 705nm (fallback)

    red: np.ndarray = dataset["red"].values.astype(np.float32)  # B04 - 665nm

    with np.errstate(divide="ignore", invalid="ignore"):
        ndre = (red_edge - red) / (red_edge + red)
        ndre = np.nan_to_num(ndre, nan=0.0)

    return ndre


def create_enhanced_kelp_detection_mask(
    dataset: xr.Dataset, config: Dict
) -> np.ndarray:
    """Create enhanced kelp detection mask using NDRE and traditional methods.

    Combines multiple detection approaches:
    1. NDRE-based detection (optimal for submerged kelp)
    2. Traditional Red Edge NDVI
    3. FAI (Floating Algae Index)

    Args:
        dataset: xarray Dataset with spectral bands
        config: Configuration dictionary with thresholds

    Returns:
        Boolean array where True indicates potential kelp
    """
    # Calculate enhanced NDRE (primary method)
    ndre = calculate_ndre(dataset)

    # Calculate traditional indices for comparison
    fai = calculate_fai(dataset)
    red_edge_ndvi = calculate_red_edge_ndvi(dataset)

    # Enhanced kelp detection criteria
    ndre_threshold = config.get(
        "ndre_threshold", 0.0
    )  # Conservative threshold from research
    kelp_ndre = ndre > ndre_threshold

    # Traditional criteria for validation
    kelp_fai = fai > config.get("kelp_fai_threshold", 0.01)
    kelp_ndvi = red_edge_ndvi > config.get("kelp_ndvi_threshold", 0.1)

    # Combine criteria - NDRE as primary, others as supporting evidence
    if config.get("use_enhanced_detection", True):
        # NDRE-based detection with traditional validation
        kelp_mask = kelp_ndre & (kelp_fai | kelp_ndvi)
    else:
        # Fall back to traditional method
        kelp_mask = kelp_fai & kelp_ndvi

    if config.get("apply_morphology", True):
        # Apply morphological operations
        kernel = np.ones((3, 3))
        kelp_mask = ndimage.binary_opening(kelp_mask, kernel)

        # Remove small clusters
        min_size = config.get("min_kelp_cluster_size", 10)
        kelp_mask = remove_small_objects(kelp_mask, min_size)

    return kelp_mask


def create_skema_kelp_detection_mask(
    dataset: xr.Dataset, config: Dict
) -> np.ndarray:
    """Create SKEMA-based kelp detection mask using research-validated algorithms.

    Implements the state-of-the-art SKEMA framework from UVic research:
    - Water Anomaly Filter (WAF) for sunglint/artifact removal
    - Derivative-based feature detection (80.18% accuracy in research)
    - NDRE-based submerged kelp detection
    - Enhanced red-edge spectral analysis

    Based on:
    - Uhl et al. (2016): Derivative-based feature detection
    - Timmer et al. (2022): Red-edge vs NIR performance analysis

    Args:
        dataset: xarray Dataset with spectral bands
        config: Configuration dictionary with SKEMA parameters

    Returns:
        Boolean array where True indicates potential kelp (SKEMA method)
    """
    # Step 1: Apply Water Anomaly Filter to remove surface artifacts
    if config.get("apply_waf", True):
        filtered_dataset = apply_water_anomaly_filter(dataset, config.get("waf_config"))
    else:
        filtered_dataset = dataset

    # Step 2: Calculate spectral derivatives and features
    derivative_dataset = calculate_spectral_derivatives(filtered_dataset)

    # Step 3: Apply derivative-based kelp detection
    derivative_mask = apply_derivative_kelp_detection(
        derivative_dataset, config.get("derivative_config")
    )

    # Step 4: Combine with enhanced NDRE detection
    if config.get("combine_with_ndre", True):
        # Calculate NDRE for submerged kelp detection
        ndre_mask = _apply_ndre_detection(filtered_dataset, config)
        
        # Combine derivative and NDRE approaches
        # Research shows NDRE detects 18% more kelp than traditional NDVI
        if config.get("detection_combination", "union") == "union":
            combined_mask = derivative_mask | ndre_mask
        elif config.get("detection_combination", "union") == "intersection":
            combined_mask = derivative_mask & ndre_mask
        else:  # weighted combination
            weight_derivative = config.get("derivative_weight", 0.6)
            weight_ndre = config.get("ndre_weight", 0.4)
            combined_mask = (
                (derivative_mask.astype(float) * weight_derivative + 
                 ndre_mask.astype(float) * weight_ndre) > 0.5
            )
    else:
        combined_mask = derivative_mask

    # Step 5: Apply morphological operations for cleanup
    if config.get("apply_morphology", True):
        kernel = np.ones((3, 3))
        combined_mask = ndimage.binary_opening(combined_mask, kernel)

        # Remove small clusters
        min_size = config.get("min_kelp_cluster_size", 10)
        combined_mask = remove_small_objects(combined_mask, min_size)

    return combined_mask


def _apply_ndre_detection(dataset: xr.Dataset, config: Dict) -> np.ndarray:
    """Apply NDRE-based kelp detection for submerged kelp.
    
    Uses research-validated NDRE thresholds for optimal submerged kelp detection.
    NDRE outperforms NDVI by detecting kelp at twice the depth (90-100cm vs 30-50cm).
    
    Args:
        dataset: Satellite imagery dataset
        config: Configuration parameters
        
    Returns:
        Boolean mask where True indicates kelp (NDRE method)
    """
    # Calculate NDRE (already implemented in the module)
    ndre = calculate_ndre(dataset)
    
    # Research-based threshold for submerged kelp detection
    ndre_threshold = config.get("ndre_threshold", 0.0)  # Conservative threshold from research
    
    # Additional criteria for kelp vs other vegetation
    kelp_ndre = ndre > ndre_threshold
    
    # Optional: Add water context requirement
    if config.get("require_water_context", True):
        water_mask = create_water_mask(dataset, config.get("water_threshold", 0.1))
        # Kelp should be in or near water areas
        water_expanded = ndimage.binary_dilation(water_mask, iterations=3)
        kelp_ndre = kelp_ndre & water_expanded
    
    return kelp_ndre


def remove_small_objects(binary_mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove small connected components from binary mask.

    Args:
        binary_mask: Boolean array
        min_size: Minimum number of pixels for object to be retained

    Returns:
        Cleaned binary mask
    """
    from typing import Tuple, cast
    labeled_result = ndimage.label(binary_mask)
    labeled_array, num_features = cast(Tuple[np.ndarray, int], labeled_result)

    # Count pixels in each component
    component_sizes = np.bincount(labeled_array.ravel())

    # Create mask for components larger than min_size
    large_components = component_sizes >= min_size
    large_components[0] = False  # Background component

    # Create final mask
    cleaned_mask = large_components[labeled_array]

    return cleaned_mask


def _detect_cloud_shadows(dataset: xr.Dataset, cloud_mask: np.ndarray) -> np.ndarray:
    """Detect cloud shadows using spectral analysis.

    Cloud shadows are characterized by:
    - Low reflectance in all bands
    - Low NIR/Red ratio
    - Spatial proximity to clouds (optional enhancement)

    Args:
        dataset: xarray Dataset with spectral bands
        cloud_mask: Boolean array of detected clouds

    Returns:
        Boolean array where True indicates potential cloud shadows
    """
    red: np.ndarray = dataset["red"].values.astype(np.float32)
    nir: np.ndarray = dataset["nir"].values.astype(np.float32)
    swir1: np.ndarray = dataset["swir1"].values.astype(np.float32)

    # Shadow detection criteria
    # 1. Low overall reflectance
    low_reflectance = (red < 0.15) & (nir < 0.15) & (swir1 < 0.15)

    # 2. Low NIR/Red ratio (shadows suppress NIR more than visible)
    with np.errstate(divide="ignore", invalid="ignore"):
        nir_red_ratio = nir / (red + 0.001)  # Add small value to avoid division by zero
        nir_red_ratio = np.nan_to_num(nir_red_ratio, nan=1.0)

    low_nir_ratio = nir_red_ratio < 1.2

    # 3. Not water (shadows should not be confused with clear water)
    # Use a simple NDWI-like check
    red_edge: np.ndarray = dataset["red_edge"].values.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        water_index = (red_edge - nir) / (red_edge + nir)
        water_index = np.nan_to_num(water_index, nan=0.0)

    not_water = water_index < 0.2  # Conservative threshold to avoid water

    # Combine criteria
    shadow_mask = low_reflectance & low_nir_ratio & not_water

    # Clean up shadow mask
    kernel = np.ones((3, 3))
    shadow_mask = ndimage.binary_opening(shadow_mask, kernel)

    return shadow_mask


def _create_basic_cloud_mask(dataset: xr.Dataset) -> np.ndarray:
    """Create basic cloud mask using spectral thresholds when no cloud data available.

    Args:
        dataset: xarray Dataset with spectral bands

    Returns:
        Boolean array where True indicates potential clouds
    """
    red = dataset["red"].values
    nir = dataset["nir"].values
    swir1 = dataset["swir1"].values

    # Simple cloud detection using high reflectance
    high_reflectance = (red > 0.3) & (nir > 0.3) & (swir1 > 0.2)

    # Additional criteria: low NDVI (clouds have low vegetation signal)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
        ndvi = np.nan_to_num(ndvi, nan=0.0)

    low_vegetation = ndvi < 0.2

    cloud_mask = high_reflectance & low_vegetation

    return cloud_mask


def get_mask_statistics(dataset: xr.Dataset) -> Dict[str, float]:
    """Calculate statistics for different mask types in dataset.

    Args:
        dataset: Masked xarray Dataset

    Returns:
        Dictionary with mask coverage percentages
    """
    stats = {}

    total_pixels = dataset.sizes["x"] * dataset.sizes["y"]

    if "cloud_mask" in dataset:
        cloud_pixels = int(dataset["cloud_mask"].sum())
        stats["cloud_coverage_percent"] = (cloud_pixels / total_pixels) * 100

    if "water_mask" in dataset:
        water_pixels = int(dataset["water_mask"].sum())
        stats["water_coverage_percent"] = (water_pixels / total_pixels) * 100

    if "kelp_mask" in dataset:
        kelp_pixels = int(dataset["kelp_mask"].sum())
        stats["kelp_coverage_percent"] = (kelp_pixels / total_pixels) * 100

    if "valid_mask" in dataset:
        valid_pixels = int(dataset["valid_mask"].sum())
        stats["valid_coverage_percent"] = (valid_pixels / total_pixels) * 100

    return stats
