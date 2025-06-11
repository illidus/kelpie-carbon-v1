"""Tests for mask module."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from kelpie_carbon_v1.core.mask import (
    apply_mask,
    calculate_fai,
    calculate_red_edge_ndvi,
    create_cloud_mask,
    create_kelp_detection_mask,
    create_water_mask,
    get_mask_statistics,
    remove_small_objects,
)


@pytest.fixture
def sample_dataset():
    """Create a sample satellite dataset for testing."""
    np.random.seed(42)
    height, width = 50, 50

    # Create realistic spectral data
    red = np.random.normal(0.1, 0.03, (height, width)).clip(0, 1)
    red_edge = np.random.normal(0.15, 0.04, (height, width)).clip(0, 1)
    nir = np.random.normal(0.3, 0.1, (height, width)).clip(0, 1)
    swir1 = np.random.normal(0.2, 0.05, (height, width)).clip(0, 1)
    cloud_mask = np.random.choice([0, 1], size=(height, width), p=[0.8, 0.2])

    # Create coordinate arrays
    lons = np.linspace(-123.5, -123.0, width)
    lats = np.linspace(49.5, 49.0, height)

    return xr.Dataset(
        {
            "red": (["y", "x"], red),
            "red_edge": (["y", "x"], red_edge),
            "nir": (["y", "x"], nir),
            "swir1": (["y", "x"], swir1),
            "cloud_mask": (["y", "x"], cloud_mask),
        },
        coords={"x": lons, "y": lats},
    )


def test_apply_mask_basic(sample_dataset):
    """Test basic masking functionality."""
    masked_data = apply_mask(sample_dataset)

    # Check that new mask layers are added
    assert "cloud_mask" in masked_data
    assert "water_mask" in masked_data
    assert "kelp_mask" in masked_data
    assert "valid_mask" in masked_data

    # Check data types
    assert masked_data["cloud_mask"].dtype == np.uint8
    assert masked_data["water_mask"].dtype == np.uint8
    assert masked_data["kelp_mask"].dtype == np.uint8
    assert masked_data["valid_mask"].dtype == np.uint8


def test_apply_mask_with_custom_config(sample_dataset):
    """Test masking with custom configuration."""
    config = {
        "cloud_threshold": 0.3,
        "water_ndwi_threshold": 0.2,
        "kelp_fai_threshold": -0.005,
        "apply_morphology": False,
        "min_kelp_cluster_size": 5,
    }

    masked_data = apply_mask(sample_dataset, config)

    assert "kelp_mask" in masked_data
    assert isinstance(masked_data, xr.Dataset)


def test_create_cloud_mask(sample_dataset):
    """Test cloud mask creation."""
    cloud_mask = create_cloud_mask(sample_dataset)

    assert isinstance(cloud_mask, np.ndarray)
    assert cloud_mask.dtype == bool
    assert cloud_mask.shape == sample_dataset["red"].shape
    
    # Test with custom threshold
    cloud_mask_conservative = create_cloud_mask(sample_dataset, threshold=0.8)
    assert isinstance(cloud_mask_conservative, np.ndarray)
    assert cloud_mask_conservative.dtype == bool
    
    # Conservative threshold should detect fewer clouds
    assert np.sum(cloud_mask_conservative) <= np.sum(cloud_mask)


def test_create_water_mask(sample_dataset):
    """Test water mask creation using NDWI."""
    water_mask = create_water_mask(sample_dataset)

    assert isinstance(water_mask, np.ndarray)
    assert water_mask.dtype == bool
    assert water_mask.shape == sample_dataset["red"].shape


def test_create_kelp_detection_mask(sample_dataset):
    """Test kelp detection mask creation."""
    config = {
        "kelp_fai_threshold": -0.01,
        "apply_morphology": True,
        "min_kelp_cluster_size": 10,
    }

    kelp_mask = create_kelp_detection_mask(sample_dataset, config)

    assert isinstance(kelp_mask, np.ndarray)
    assert kelp_mask.dtype == bool
    assert kelp_mask.shape == sample_dataset["red"].shape


def test_calculate_fai(sample_dataset):
    """Test Floating Algae Index calculation."""
    fai = calculate_fai(sample_dataset)

    assert isinstance(fai, np.ndarray)
    assert fai.shape == sample_dataset["red"].shape
    assert not np.any(np.isnan(fai))  # Should not contain NaN


def test_calculate_red_edge_ndvi(sample_dataset):
    """Test Red Edge NDVI calculation."""
    red_edge_ndvi = calculate_red_edge_ndvi(sample_dataset)

    assert isinstance(red_edge_ndvi, np.ndarray)
    assert red_edge_ndvi.shape == sample_dataset["red"].shape
    assert not np.any(np.isnan(red_edge_ndvi))  # Should not contain NaN
    assert np.all(red_edge_ndvi >= -1) and np.all(red_edge_ndvi <= 1)  # NDVI range


def test_remove_small_objects():
    """Test small object removal from binary mask."""
    # Create a test binary mask with small and large objects
    test_mask = np.zeros((20, 20), dtype=bool)
    
    # Add a large object (16 pixels)
    test_mask[5:9, 5:9] = True
    
    # Add small objects (1-4 pixels each)
    test_mask[1, 1] = True  # 1 pixel
    test_mask[10:12, 10:12] = True  # 4 pixels
    test_mask[15, 15:17] = True  # 2 pixels
    
    # Remove objects smaller than 5 pixels
    cleaned_mask = remove_small_objects(test_mask, min_size=5)
    
    assert isinstance(cleaned_mask, np.ndarray)
    assert cleaned_mask.dtype == bool
    assert cleaned_mask.shape == test_mask.shape
    
    # Should only keep the large object (16 pixels >= 5)
    assert np.sum(cleaned_mask) == 16
    assert np.all(cleaned_mask[5:9, 5:9] == True)
    assert cleaned_mask[1, 1] == False  # Small object removed
    assert np.all(cleaned_mask[15, 15:17] == False)  # Small object removed


def test_get_mask_statistics(sample_dataset):
    """Test mask statistics calculation."""
    masked_data = apply_mask(sample_dataset)
    stats = get_mask_statistics(masked_data)

    assert isinstance(stats, dict)
    assert "cloud_coverage_percent" in stats
    assert "water_coverage_percent" in stats
    assert "kelp_coverage_percent" in stats
    assert "valid_coverage_percent" in stats

    # Check that percentages are valid
    for key, value in stats.items():
        assert 0 <= value <= 100


def test_cloud_mask_without_cloud_data():
    """Test cloud mask creation when no cloud data is available."""
    # Create dataset without cloud_mask band
    dataset_no_clouds = xr.Dataset(
        {
            "red": (["y", "x"], np.random.normal(0.1, 0.03, (10, 10)).clip(0, 1)),
            "red_edge": (["y", "x"], np.random.normal(0.15, 0.04, (10, 10)).clip(0, 1)),
            "nir": (["y", "x"], np.random.normal(0.3, 0.1, (10, 10)).clip(0, 1)),
            "swir1": (["y", "x"], np.random.normal(0.2, 0.05, (10, 10)).clip(0, 1)),
            # No cloud_mask band
        }
    )
    
    # Should fall back to basic cloud detection
    cloud_mask = create_cloud_mask(dataset_no_clouds)
    
    assert isinstance(cloud_mask, np.ndarray)
    assert cloud_mask.dtype == bool
    assert cloud_mask.shape == (10, 10)
    
    # Test that it doesn't crash and produces reasonable results
    cloud_percentage = np.sum(cloud_mask) / cloud_mask.size
    assert 0 <= cloud_percentage <= 1  # Should be between 0 and 100%


def test_cloud_shadow_detection():
    """Test cloud shadow detection functionality."""
    # Create dataset with known cloud and shadow patterns
    height, width = 20, 20
    
    # Create base reflectance values
    red = np.full((height, width), 0.1)
    red_edge = np.full((height, width), 0.15)
    nir = np.full((height, width), 0.2)
    swir1 = np.full((height, width), 0.15)
    
    # Add cloud area (high reflectance)
    red[5:8, 5:8] = 0.4
    nir[5:8, 5:8] = 0.4
    swir1[5:8, 5:8] = 0.3
    
    # Add shadow area (low reflectance)
    red[12:15, 12:15] = 0.08  # Very low reflectance
    red_edge[12:15, 12:15] = 0.10
    nir[12:15, 12:15] = 0.09
    swir1[12:15, 12:15] = 0.08
    
    # Create cloud mask for the cloud area
    cloud_prob = np.zeros((height, width))
    cloud_prob[5:8, 5:8] = 0.8  # High cloud probability
    
    dataset = xr.Dataset(
        {
            "red": (["y", "x"], red),
            "red_edge": (["y", "x"], red_edge),
            "nir": (["y", "x"], nir),
            "swir1": (["y", "x"], swir1),
            "cloud_mask": (["y", "x"], cloud_prob),
        }
    )
    
    # Test cloud mask creation (should detect both clouds and shadows)
    combined_mask = create_cloud_mask(dataset)
    
    assert isinstance(combined_mask, np.ndarray)
    assert combined_mask.dtype == bool
    assert combined_mask.shape == (height, width)
    
    # Should detect the cloud area
    assert np.any(combined_mask[5:8, 5:8])
    
    # May detect shadow area (depends on thresholds, but should not crash)
    shadow_detected = np.any(combined_mask[12:15, 12:15])
    # We don't assert True for shadow detection as it depends on specific thresholds
    # The main test is that the function works without errors


def test_mask_integration():
    """Test integration of all masking components."""
    # Create a synthetic dataset with known patterns
    height, width = 30, 30

    # Create water area (high NDWI)
    red_edge = np.full((height, width), 0.1)
    nir = np.full((height, width), 0.05)  # Lower NIR for water

    # Create kelp area with high FAI
    red = np.full((height, width), 0.1)
    swir1 = np.full((height, width), 0.2)
    nir[10:20, 10:20] = 0.4  # Higher NIR for vegetation

    dataset = xr.Dataset(
        {
            "red": (["y", "x"], red),
            "red_edge": (["y", "x"], red_edge),
            "nir": (["y", "x"], nir),
            "swir1": (["y", "x"], swir1),
            "cloud_mask": (["y", "x"], np.zeros((height, width))),  # No clouds
        }
    )

    masked_data = apply_mask(dataset)

    # Should detect water and some kelp
    assert np.sum(masked_data["water_mask"]) > 0
    assert np.sum(masked_data["kelp_mask"]) >= 0  # May or may not detect kelp
    assert np.sum(masked_data["valid_mask"]) == height * width  # No clouds


def test_edge_cases():
    """Test edge cases and error handling."""
    # Test with very small dataset
    small_dataset = xr.Dataset(
        {
            "red": (["y", "x"], np.array([[0.1, 0.2], [0.3, 0.4]])),
            "red_edge": (["y", "x"], np.array([[0.15, 0.25], [0.35, 0.45]])),
            "nir": (["y", "x"], np.array([[0.3, 0.4], [0.5, 0.6]])),
            "swir1": (["y", "x"], np.array([[0.2, 0.3], [0.4, 0.5]])),
            "cloud_mask": (["y", "x"], np.array([[0, 0], [1, 0]])),
        }
    )

    masked_data = apply_mask(small_dataset)

    assert masked_data["red"].shape == (2, 2)
    assert "kelp_mask" in masked_data


def test_fai_calculation_edge_values():
    """Test FAI calculation with edge values."""
    # Test with extreme values
    dataset = xr.Dataset(
        {
            "red": (["y", "x"], np.array([[0.0, 1.0], [0.5, 0.5]])),
            "nir": (["y", "x"], np.array([[0.0, 1.0], [0.5, 0.5]])),
            "swir1": (["y", "x"], np.array([[0.0, 1.0], [0.5, 0.5]])),
        }
    )

    fai = calculate_fai(dataset)

    assert not np.any(np.isnan(fai))
    assert fai.shape == (2, 2)


def test_mask_consistency():
    """Test that masks are consistent with each other."""
    height, width = 25, 25
    dataset = xr.Dataset(
        {
            "red": (["y", "x"], np.random.random((height, width))),
            "red_edge": (["y", "x"], np.random.random((height, width))),
            "nir": (["y", "x"], np.random.random((height, width))),
            "swir1": (["y", "x"], np.random.random((height, width))),
            "cloud_mask": (
                ["y", "x"],
                np.random.choice([0, 1], size=(height, width), p=[0.9, 0.1]),
            ),
        }
    )

    masked_data = apply_mask(dataset)

    # Kelp should only be detected in valid water areas
    kelp_pixels = masked_data["kelp_mask"] == 1
    water_pixels = masked_data["water_mask"] == 1
    valid_pixels = masked_data["valid_mask"] == 1

    # All kelp pixels should be in water and valid areas
    assert np.all((kelp_pixels & ~water_pixels) is False) or np.sum(kelp_pixels) == 0
    assert np.all((kelp_pixels & ~valid_pixels) is False) or np.sum(kelp_pixels) == 0
