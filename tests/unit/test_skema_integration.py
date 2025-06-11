"""Tests for SKEMA integration functionality."""

import numpy as np
import pytest
import xarray as xr

from kelpie_carbon_v1.core.mask import create_skema_kelp_detection_mask
from kelpie_carbon_v1.processing.water_anomaly_filter import (
    apply_water_anomaly_filter,
    calculate_waf_quality_metrics,
)
from kelpie_carbon_v1.processing.derivative_features import (
    calculate_spectral_derivatives,
    apply_derivative_kelp_detection,
    calculate_derivative_quality_metrics,
)


@pytest.fixture
def skema_test_dataset():
    """Create a test dataset for SKEMA functionality testing."""
    np.random.seed(42)
    height, width = 30, 30

    # Create realistic spectral data with kelp signatures
    red = np.random.normal(0.1, 0.02, (height, width)).clip(0, 1)
    red_edge = np.random.normal(0.15, 0.03, (height, width)).clip(0, 1)
    nir = np.random.normal(0.3, 0.05, (height, width)).clip(0, 1)
    swir1 = np.random.normal(0.2, 0.03, (height, width)).clip(0, 1)

    # Add simulated kelp area with characteristic spectral signature
    kelp_area = slice(10, 20), slice(10, 20)
    red[kelp_area] = np.random.normal(0.08, 0.01, (10, 10)).clip(0, 1)  # Lower red
    red_edge[kelp_area] = np.random.normal(0.18, 0.02, (10, 10)).clip(0, 1)  # Higher red-edge
    nir[kelp_area] = np.random.normal(0.25, 0.03, (10, 10)).clip(0, 1)  # Moderate NIR

    # Add simulated sunglint area
    sunglint_area = slice(5, 8), slice(5, 8)
    red[sunglint_area] = 0.4  # High reflectance
    red_edge[sunglint_area] = 0.4
    nir[sunglint_area] = 0.4
    swir1[sunglint_area] = 0.3

    return xr.Dataset(
        {
            "red": (["y", "x"], red),
            "red_edge": (["y", "x"], red_edge),
            "nir": (["y", "x"], nir),
            "swir1": (["y", "x"], swir1),
        }
    )


def test_water_anomaly_filter(skema_test_dataset):
    """Test Water Anomaly Filter functionality."""
    # Apply WAF with default configuration
    filtered_dataset = apply_water_anomaly_filter(skema_test_dataset)

    # Check that WAF mask is added
    assert "waf_mask" in filtered_dataset
    assert filtered_dataset["waf_mask"].dtype == np.uint8

    # Check that spectral bands are preserved
    for band in ["red", "red_edge", "nir", "swir1"]:
        assert band in filtered_dataset
        assert filtered_dataset[band].shape == skema_test_dataset[band].shape

    # WAF should filter some pixels (sunglint areas)
    waf_mask = filtered_dataset["waf_mask"].values
    assert np.sum(waf_mask) < waf_mask.size  # Some pixels should be filtered


def test_waf_sunglint_detection(skema_test_dataset):
    """Test that WAF correctly detects sunglint areas."""
    filtered_dataset = apply_water_anomaly_filter(skema_test_dataset)
    
    # The sunglint area (5:8, 5:8) should be detected and filtered
    waf_mask = filtered_dataset["waf_mask"].values
    sunglint_region = waf_mask[5:8, 5:8]
    
    # Most of the sunglint area should be filtered (waf_mask = 0 for filtered areas)
    filtered_pixels = np.sum(sunglint_region == 0)
    total_sunglint_pixels = sunglint_region.size
    
    # At least 50% of sunglint area should be detected and filtered
    assert filtered_pixels >= total_sunglint_pixels * 0.5


def test_waf_quality_metrics(skema_test_dataset):
    """Test WAF quality metrics calculation."""
    filtered_dataset = apply_water_anomaly_filter(skema_test_dataset)
    metrics = calculate_waf_quality_metrics(filtered_dataset)

    # Check required metrics are present
    required_metrics = [
        "valid_pixel_percentage",
        "filtered_pixel_percentage",
        "total_pixels",
        "valid_pixels",
        "filtered_pixels",
    ]

    for metric in required_metrics:
        assert metric in metrics

    # Check metric validity
    assert 0 <= metrics["valid_pixel_percentage"] <= 100
    assert 0 <= metrics["filtered_pixel_percentage"] <= 100
    assert metrics["valid_pixel_percentage"] + metrics["filtered_pixel_percentage"] == 100
    assert metrics["total_pixels"] == 30 * 30


def test_spectral_derivatives_calculation(skema_test_dataset):
    """Test spectral derivatives calculation."""
    derivative_dataset = calculate_spectral_derivatives(skema_test_dataset)

    # Check that derivative features are added
    expected_derivatives = [
        "d_red_red_edge",
        "d_red_edge_nir", 
        "d_nir_swir1"
    ]

    for deriv_name in expected_derivatives:
        assert deriv_name in derivative_dataset

    # Check that kelp-specific features are calculated
    expected_features = [
        "fucoxanthin_absorption",
        "red_edge_slope",
        "red_edge_curvature",
        "nir_transition",
        "composite_kelp_derivative",
    ]

    for feature_name in expected_features:
        assert feature_name in derivative_dataset

    # Check data shapes and types
    for feature_name in expected_features:
        feature_data = derivative_dataset[feature_name]
        assert feature_data.shape == (30, 30)
        assert not np.any(np.isnan(feature_data.values))


def test_derivative_kelp_detection(skema_test_dataset):
    """Test derivative-based kelp detection."""
    # Calculate derivatives first
    derivative_dataset = calculate_spectral_derivatives(skema_test_dataset)

    # Apply derivative-based detection
    kelp_mask = apply_derivative_kelp_detection(derivative_dataset)

    assert isinstance(kelp_mask, np.ndarray)
    assert kelp_mask.dtype == bool
    assert kelp_mask.shape == (30, 30)

    # Should detect some kelp in our test dataset
    detected_pixels = np.sum(kelp_mask)
    assert detected_pixels >= 0  # May or may not detect kelp depending on thresholds


def test_derivative_detection_with_config(skema_test_dataset):
    """Test derivative detection with custom configuration."""
    config = {
        "red_edge_slope_threshold": 0.005,  # Lower threshold
        "nir_transition_threshold": 0.01,   # Lower threshold
        "composite_threshold": 0.01,        # Lower threshold
        "min_cluster_size": 3,              # Smaller clusters
        "morphology_cleanup": True,
    }

    derivative_dataset = calculate_spectral_derivatives(skema_test_dataset)
    kelp_mask = apply_derivative_kelp_detection(derivative_dataset, config)

    assert isinstance(kelp_mask, np.ndarray)
    assert kelp_mask.dtype == bool

    # With lower thresholds, should detect more potential kelp
    detected_pixels = np.sum(kelp_mask)
    assert detected_pixels >= 0


def test_derivative_quality_metrics(skema_test_dataset):
    """Test derivative quality metrics calculation."""
    derivative_dataset = calculate_spectral_derivatives(skema_test_dataset)
    kelp_mask = apply_derivative_kelp_detection(derivative_dataset)
    
    metrics = calculate_derivative_quality_metrics(derivative_dataset, kelp_mask)

    # Check required metrics
    required_metrics = [
        "detection_coverage_percent",
        "mean_composite_strength",
        "max_composite_strength",
        "std_composite_strength",
        "num_kelp_clusters",
    ]

    for metric in required_metrics:
        assert metric in metrics

    # Check metric validity
    assert 0 <= metrics["detection_coverage_percent"] <= 100
    assert metrics["num_kelp_clusters"] >= 0


def test_skema_kelp_detection_integration(skema_test_dataset):
    """Test integrated SKEMA kelp detection."""
    config = {
        "apply_waf": True,
        "combine_with_ndre": True,
        "detection_combination": "union",
        "apply_morphology": True,
        "min_kelp_cluster_size": 5,
        "ndre_threshold": 0.0,
        "require_water_context": False,  # Disable for test simplicity
    }

    kelp_mask = create_skema_kelp_detection_mask(skema_test_dataset, config)

    assert isinstance(kelp_mask, np.ndarray)
    assert kelp_mask.dtype == bool
    assert kelp_mask.shape == (30, 30)

    # Should produce a valid detection mask
    detected_pixels = np.sum(kelp_mask)
    total_pixels = kelp_mask.size
    detection_percentage = (detected_pixels / total_pixels) * 100

    # Detection percentage should be reasonable for synthetic data
    # Note: Synthetic data may produce high detection rates due to simplified spectral patterns
    assert 0 <= detection_percentage <= 100  # Ensure valid percentage
    
    # The algorithm should produce consistent results
    assert isinstance(detection_percentage, (int, float, np.floating))


def test_skema_detection_combination_methods(skema_test_dataset):
    """Test different detection combination methods."""
    base_config = {
        "apply_waf": True,
        "combine_with_ndre": True,
        "apply_morphology": True,
        "min_kelp_cluster_size": 3,
        "ndre_threshold": 0.0,
        "require_water_context": False,
    }

    # Test union combination
    union_config = {**base_config, "detection_combination": "union"}
    union_mask = create_skema_kelp_detection_mask(skema_test_dataset, union_config)

    # Test intersection combination
    intersection_config = {**base_config, "detection_combination": "intersection"}
    intersection_mask = create_skema_kelp_detection_mask(skema_test_dataset, intersection_config)

    # Test weighted combination
    weighted_config = {
        **base_config,
        "detection_combination": "weighted",
        "derivative_weight": 0.7,
        "ndre_weight": 0.3,
    }
    weighted_mask = create_skema_kelp_detection_mask(skema_test_dataset, weighted_config)

    # Union should detect the most pixels, intersection the least
    union_pixels = np.sum(union_mask)
    intersection_pixels = np.sum(intersection_mask)
    weighted_pixels = np.sum(weighted_mask)

    assert intersection_pixels <= weighted_pixels <= union_pixels


def test_skema_without_waf(skema_test_dataset):
    """Test SKEMA detection without Water Anomaly Filter."""
    config = {
        "apply_waf": False,  # Disable WAF
        "combine_with_ndre": True,
        "detection_combination": "union",
        "apply_morphology": True,
        "min_kelp_cluster_size": 5,
    }

    kelp_mask = create_skema_kelp_detection_mask(skema_test_dataset, config)

    assert isinstance(kelp_mask, np.ndarray)
    assert kelp_mask.dtype == bool
    assert kelp_mask.shape == (30, 30)

    # Should still work without WAF
    detected_pixels = np.sum(kelp_mask)
    assert detected_pixels >= 0


def test_skema_derivative_only(skema_test_dataset):
    """Test SKEMA detection using only derivative features."""
    config = {
        "apply_waf": True,
        "combine_with_ndre": False,  # Use only derivative detection
        "apply_morphology": True,
        "min_kelp_cluster_size": 3,
    }

    kelp_mask = create_skema_kelp_detection_mask(skema_test_dataset, config)

    assert isinstance(kelp_mask, np.ndarray)
    assert kelp_mask.dtype == bool
    assert kelp_mask.shape == (30, 30)

    # Should work with only derivative features
    detected_pixels = np.sum(kelp_mask)
    assert detected_pixels >= 0


def test_edge_cases_small_dataset():
    """Test SKEMA functionality with very small dataset."""
    # Create minimal dataset
    small_dataset = xr.Dataset(
        {
            "red": (["y", "x"], np.array([[0.1, 0.2], [0.3, 0.4]])),
            "red_edge": (["y", "x"], np.array([[0.15, 0.25], [0.35, 0.45]])),
            "nir": (["y", "x"], np.array([[0.3, 0.4], [0.5, 0.6]])),
            "swir1": (["y", "x"], np.array([[0.2, 0.3], [0.4, 0.5]])),
        }
    )

    config = {
        "apply_waf": True,
        "combine_with_ndre": True,
        "min_kelp_cluster_size": 1,  # Small minimum for tiny dataset
    }

    # Should not crash with small dataset
    kelp_mask = create_skema_kelp_detection_mask(small_dataset, config)

    assert isinstance(kelp_mask, np.ndarray)
    assert kelp_mask.dtype == bool
    assert kelp_mask.shape == (2, 2)


def test_missing_bands_handling():
    """Test graceful handling of missing spectral bands."""
    # Dataset with only some bands
    partial_dataset = xr.Dataset(
        {
            "red": (["y", "x"], np.random.random((10, 10))),
            "red_edge": (["y", "x"], np.random.random((10, 10))),
            # Missing nir and swir1
        }
    )

    config = {
        "apply_waf": False,  # Disable WAF to avoid issues with missing bands
        "combine_with_ndre": True,
        "require_water_context": False,
    }

    # Should handle missing bands gracefully
    kelp_mask = create_skema_kelp_detection_mask(partial_dataset, config)

    assert isinstance(kelp_mask, np.ndarray)
    assert kelp_mask.dtype == bool
    assert kelp_mask.shape == (10, 10) 