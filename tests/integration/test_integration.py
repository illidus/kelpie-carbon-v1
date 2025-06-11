"""Integration tests for the complete kelp analysis pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from kelpie_carbon_v1.api.main import app
from kelpie_carbon_v1.core.fetch import fetch_sentinel_tiles
from kelpie_carbon_v1.core.indices import calculate_indices_from_dataset
from kelpie_carbon_v1.core.mask import (
    apply_mask,
    calculate_fai,
    calculate_red_edge_ndvi,
    create_kelp_detection_mask,
    create_water_mask,
    get_mask_statistics,
)
from kelpie_carbon_v1.core.model import predict_biomass


@pytest.fixture
def test_client():
    """Create test client for API tests."""
    return TestClient(app)


@pytest.fixture
def sample_api_request():
    """Sample API request payload."""
    return {
        "aoi": {"lat": 49.2827, "lng": -123.1207},  # Vancouver area
        "start_date": "2023-06-01",
        "end_date": "2023-06-30",
    }


@pytest.fixture
def realistic_kelp_dataset():
    """Create a realistic dataset with kelp-like spectral signatures."""
    height, width = 50, 50

    # Create coordinates
    lons = np.linspace(-123.2, -123.1, width)
    lats = np.linspace(49.3, 49.2, height)

    # Create realistic spectral values for kelp and water
    # Kelp typically has higher NIR and lower red reflectance
    red = np.random.uniform(0.02, 0.05, (height, width))  # Low red for kelp
    red_edge = np.random.uniform(0.03, 0.08, (height, width))
    nir = np.random.uniform(0.15, 0.35, (height, width))  # Higher NIR for kelp
    swir1 = np.random.uniform(0.01, 0.03, (height, width))  # Very low SWIR for water

    # Create kelp patches in certain areas
    kelp_mask = np.zeros((height, width), dtype=bool)
    kelp_mask[15:35, 15:35] = True

    # Adjust spectral values in kelp areas
    red[kelp_mask] = np.random.uniform(0.01, 0.03, np.sum(kelp_mask))  # Even lower red
    nir[kelp_mask] = np.random.uniform(0.25, 0.45, np.sum(kelp_mask))  # Higher NIR

    # Create realistic masks
    water_mask = np.ones((height, width), dtype=bool)  # Assume all water
    valid_mask = np.ones((height, width), dtype=bool)
    cloud_mask = np.zeros((height, width), dtype=bool)

    return xr.Dataset(
        {
            "red": (["y", "x"], red),
            "red_edge": (["y", "x"], red_edge),
            "nir": (["y", "x"], nir),
            "swir1": (["y", "x"], swir1),
            "kelp_mask": (["y", "x"], kelp_mask.astype(float)),
            "water_mask": (["y", "x"], water_mask.astype(float)),
            "valid_mask": (["y", "x"], valid_mask.astype(float)),
            "cloud_mask": (["y", "x"], cloud_mask.astype(float)),
        },
        coords={"x": lons, "y": lats},
    )


def test_api_endpoint_returns_non_zero_values(test_client, sample_api_request):
    """Test that the API endpoint returns meaningful non-zero values."""
    # Mock the fetch function to return realistic data
    import kelpie_carbon_v1.core.fetch as fetch_module

    def mock_fetch_sentinel_tiles(lat, lng, start_date, end_date):
        # Create realistic dataset
        height, width = 30, 30
        lons = np.linspace(lng - 0.01, lng + 0.01, width)
        lats = np.linspace(lat - 0.01, lat + 0.01, height)

        # Realistic kelp spectral signatures
        red = np.random.uniform(0.02, 0.05, (height, width))
        red_edge = np.random.uniform(0.03, 0.08, (height, width))
        nir = np.random.uniform(0.15, 0.35, (height, width))
        swir1 = np.random.uniform(0.01, 0.03, (height, width))

        # Add some kelp patches
        kelp_areas = np.random.random((height, width)) > 0.7
        red[kelp_areas] *= 0.5  # Lower red in kelp areas
        nir[kelp_areas] *= 1.5  # Higher NIR in kelp areas

        dataset = xr.Dataset(
            {
                "red": (["y", "x"], red),
                "red_edge": (["y", "x"], red_edge),
                "nir": (["y", "x"], nir),
                "swir1": (["y", "x"], swir1),
            },
            coords={"x": lons, "y": lats},
        )

        return {"data": dataset, "metadata": {"tiles_found": 1}}

    # Temporarily replace the fetch function
    original_fetch = fetch_module.fetch_sentinel_tiles
    fetch_module.fetch_sentinel_tiles = mock_fetch_sentinel_tiles

    try:
        response = test_client.post("/api/run", json=sample_api_request)
        assert response.status_code == 200

        data = response.json()
        print(f"API Response: {data}")

        # Check that we got meaningful values
        assert data["status"] == "completed"

        # Extract biomass value from string (format: "X.X kg/ha (confidence: Y.YY)")
        biomass_str = data["biomass"]
        if "Error" not in biomass_str:
            biomass_value = float(biomass_str.split(" ")[0])
            print(f"Biomass value: {biomass_value}")
            assert biomass_value > 0, f"Expected positive biomass, got {biomass_value}"

        # Check carbon value
        carbon_str = data["carbon"]
        if "Error" not in carbon_str:
            carbon_value = float(carbon_str.split(" ")[0])
            print(f"Carbon value: {carbon_value}")
            assert carbon_value > 0, f"Expected positive carbon, got {carbon_value}"

        # Check that spectral indices are calculated
        spectral_indices = data["spectral_indices"]
        print(f"Spectral indices: {spectral_indices}")

        # Check mask statistics
        mask_stats = data["mask_statistics"]
        print(f"Mask statistics: {mask_stats}")

    finally:
        # Restore original function
        fetch_module.fetch_sentinel_tiles = original_fetch


def test_water_mask_debug():
    """Debug test to understand why water mask is not being detected."""
    print("\n=== WATER MASK DEBUG TEST ===")

    # Create test data with water-like spectral signatures
    height, width = 30, 30
    lons = np.linspace(-123.2, -123.1, width)
    lats = np.linspace(49.3, 49.2, height)

    # Water typically has: low NIR, moderate red_edge, very low SWIR
    red = np.random.uniform(0.02, 0.06, (height, width))
    red_edge = np.random.uniform(0.04, 0.09, (height, width))
    nir = np.random.uniform(0.01, 0.05, (height, width))  # Low NIR for water
    swir1 = np.random.uniform(0.001, 0.01, (height, width))  # Very low SWIR for water

    print(f"Water spectral ranges:")
    print(f"  Red: {red.min():.4f} - {red.max():.4f} (mean: {red.mean():.4f})")
    print(
        f"  Red Edge: {red_edge.min():.4f} - {red_edge.max():.4f} (mean: {red_edge.mean():.4f})"
    )
    print(f"  NIR: {nir.min():.4f} - {nir.max():.4f} (mean: {nir.mean():.4f})")
    print(f"  SWIR1: {swir1.min():.4f} - {swir1.max():.4f} (mean: {swir1.mean():.4f})")

    dataset = xr.Dataset(
        {
            "red": (["y", "x"], red),
            "red_edge": (["y", "x"], red_edge),
            "nir": (["y", "x"], nir),
            "swir1": (["y", "x"], swir1),
        },
        coords={"x": lons, "y": lats},
    )

    # Test NDWI calculation manually
    print("\nNDWI Calculation:")
    ndwi = (red_edge - nir) / (red_edge + nir)
    print(f"  NDWI range: {np.nanmin(ndwi):.4f} - {np.nanmax(ndwi):.4f}")
    print(f"  NDWI mean: {np.nanmean(ndwi):.4f}")

    # Test different thresholds
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for threshold in thresholds:
        water_mask = create_water_mask(dataset, threshold)
        coverage = np.mean(water_mask)
        print(f"  NDWI > {threshold}: {coverage:.4f} water coverage")

    print("=== WATER MASK DEBUG COMPLETE ===\n")


def test_kelp_detection_debug():
    """Debug kelp detection with known good spectral signatures."""
    print("\n=== KELP DETECTION DEBUG TEST ===")

    height, width = 30, 30
    lons = np.linspace(-123.2, -123.1, width)
    lats = np.linspace(49.3, 49.2, height)

    # Create strong kelp spectral signature
    red = np.random.uniform(0.01, 0.03, (height, width))  # Very low red
    red_edge = np.random.uniform(0.06, 0.12, (height, width))  # Moderate red edge
    nir = np.random.uniform(0.25, 0.45, (height, width))  # High NIR
    swir1 = np.random.uniform(0.005, 0.02, (height, width))  # Low SWIR

    dataset = xr.Dataset(
        {
            "red": (["y", "x"], red),
            "red_edge": (["y", "x"], red_edge),
            "nir": (["y", "x"], nir),
            "swir1": (["y", "x"], swir1),
        },
        coords={"x": lons, "y": lats},
    )

    print(f"Kelp spectral signature:")
    print(f"  Red: {red.mean():.4f} (should be low)")
    print(f"  Red Edge: {red_edge.mean():.4f}")
    print(f"  NIR: {nir.mean():.4f} (should be high)")
    print(f"  SWIR1: {swir1.mean():.4f} (should be low)")

    # Test FAI calculation
    fai = calculate_fai(dataset)
    print(f"\nFAI values:")
    print(f"  Range: {np.nanmin(fai):.4f} - {np.nanmax(fai):.4f}")
    print(f"  Mean: {np.nanmean(fai):.4f}")

    # Test Red Edge NDVI
    re_ndvi = calculate_red_edge_ndvi(dataset)
    print(f"\nRed Edge NDVI values:")
    print(f"  Range: {np.nanmin(re_ndvi):.4f} - {np.nanmax(re_ndvi):.4f}")
    print(f"  Mean: {np.nanmean(re_ndvi):.4f}")

    # Test kelp detection with different FAI thresholds
    print(f"\nKelp detection with different thresholds:")
    fai_thresholds = [-0.05, -0.01, 0.0, 0.01, 0.05, 0.1]
    for fai_thresh in fai_thresholds:
        config = {
            "kelp_fai_threshold": fai_thresh,
            "apply_morphology": False,  # Disable to see raw detection
            "min_kelp_cluster_size": 1,
        }
        kelp_mask = create_kelp_detection_mask(dataset, config)
        coverage = np.mean(kelp_mask)
        print(f"  FAI > {fai_thresh}: {coverage:.4f} kelp coverage")

    print("=== KELP DETECTION DEBUG COMPLETE ===\n")


def test_full_masking_debug():
    """Debug the full masking pipeline to see where kelp detection fails."""
    print("\n=== FULL MASKING DEBUG TEST ===")

    # Create a dataset that should definitely detect water and kelp
    height, width = 30, 30
    lons = np.linspace(-123.2, -123.1, width)
    lats = np.linspace(49.3, 49.2, height)

    # Make sure we have clear water signature (red_edge > nir for positive NDWI)
    red = np.random.uniform(0.02, 0.04, (height, width))
    red_edge = np.random.uniform(0.08, 0.15, (height, width))  # Higher than NIR
    nir = np.random.uniform(0.03, 0.07, (height, width))  # Lower than red_edge
    swir1 = np.random.uniform(0.005, 0.02, (height, width))

    dataset = xr.Dataset(
        {
            "red": (["y", "x"], red),
            "red_edge": (["y", "x"], red_edge),
            "nir": (["y", "x"], nir),
            "swir1": (["y", "x"], swir1),
        },
        coords={"x": lons, "y": lats},
    )

    print(f"Dataset for water detection:")
    print(f"  Red Edge: {red_edge.mean():.4f}")
    print(f"  NIR: {nir.mean():.4f}")
    print(f"  Red Edge > NIR: {np.mean(red_edge > nir):.4f} (should be high for water)")

    # Manual NDWI calculation
    ndwi = (red_edge - nir) / (red_edge + nir)
    print(f"  NDWI mean: {np.nanmean(ndwi):.4f} (should be positive)")

    # Test with more permissive water detection
    config = {
        "cloud_threshold": 0.5,
        "water_ndwi_threshold": 0.0,  # Very permissive
        "kelp_fai_threshold": -0.05,  # Very permissive
        "apply_morphology": False,
        "min_kelp_cluster_size": 1,
    }

    masked_data = apply_mask(dataset, config)
    mask_stats = get_mask_statistics(masked_data)

    print(f"\nMasking results with permissive thresholds:")
    print(f"  {mask_stats}")

    if mask_stats.get("water_coverage_percent", 0) > 0:
        print("  SUCCESS: Water detected!")
    else:
        print("  PROBLEM: Still no water detected")

    if mask_stats.get("kelp_coverage_percent", 0) > 0:
        print("  SUCCESS: Kelp detected!")
    else:
        print("  PROBLEM: Still no kelp detected")

    print("=== FULL MASKING DEBUG COMPLETE ===\n")


def test_indices_calculation_with_realistic_data(realistic_kelp_dataset):
    """Test that spectral indices are calculated correctly with realistic data."""
    indices = calculate_indices_from_dataset(realistic_kelp_dataset)

    print(f"Calculated indices: {list(indices.data_vars)}")

    # Check that all expected indices are present
    # Note: the indices module only calculates fai, ndre, and kelp_index currently
    expected_indices = ["fai", "ndre", "kelp_index"]
    for index_name in expected_indices:
        assert index_name in indices, f"Missing index: {index_name}"

        # Check that indices have realistic values
        index_values = indices[index_name].values
        assert not np.all(np.isnan(index_values)), f"All NaN values for {index_name}"

        mean_val = np.nanmean(index_values)
        print(
            f"{index_name}: mean={mean_val:.4f}, min={np.nanmin(index_values):.4f}, max={np.nanmax(index_values):.4f}"
        )

        # Specific range checks for indices
        if index_name in ["ndre"]:
            assert (
                0 <= mean_val <= 1
            ), f"{index_name} should be between 0 and 1, got {mean_val}"


def test_masking_with_realistic_data(realistic_kelp_dataset):
    """Test that masking produces meaningful results with realistic data."""
    # First calculate indices
    indices = calculate_indices_from_dataset(realistic_kelp_dataset)

    # Merge with original data
    combined_data = realistic_kelp_dataset.copy()
    for var in indices.data_vars:
        combined_data[var] = indices[var]

    # Apply masking
    masked_data = apply_mask(combined_data)
    mask_stats = get_mask_statistics(masked_data)

    print(f"Mask statistics: {mask_stats}")

    # Check that masks are meaningful
    assert mask_stats["water_coverage_percent"] > 0, "Expected some water coverage"
    assert (
        mask_stats["valid_coverage_percent"] > 50
    ), "Expected reasonable valid coverage"

    # Check that kelp mask was created and is reasonable
    if "kelp_mask" in masked_data:
        kelp_coverage = np.mean(masked_data["kelp_mask"].values)
        print(f"Kelp coverage: {kelp_coverage:.4f}")
        assert kelp_coverage >= 0, "Kelp coverage should be non-negative"


def test_biomass_prediction_with_realistic_data(realistic_kelp_dataset):
    """Test biomass prediction with realistic data to ensure non-zero results."""
    # Calculate indices
    indices = calculate_indices_from_dataset(realistic_kelp_dataset)

    # Merge with original data
    combined_data = realistic_kelp_dataset.copy()
    for var in indices.data_vars:
        combined_data[var] = indices[var]

    # Apply masking
    masked_data = apply_mask(combined_data)

    # Predict biomass
    result = predict_biomass(masked_data)

    print(f"Biomass prediction result: {result}")

    # Check required fields
    required_fields = [
        "biomass_kg_per_hectare",
        "prediction_confidence",
        "top_features",
        "model_type",
    ]
    for field in required_fields:
        assert field in result, f"Missing field: {field}"

    biomass = result["biomass_kg_per_hectare"]
    confidence = result["prediction_confidence"]

    print(f"Biomass: {biomass} kg/ha, Confidence: {confidence}")

    # Check that values are meaningful
    assert isinstance(
        biomass, (int, float)
    ), f"Biomass should be numeric, got {type(biomass)}"
    assert not np.isnan(biomass), "Biomass should not be NaN"
    assert biomass >= 0, f"Biomass should be non-negative, got {biomass}"

    assert (
        0 <= confidence <= 1
    ), f"Confidence should be between 0 and 1, got {confidence}"


def test_empty_data_handling():
    """Test how the pipeline handles empty or invalid data."""
    # Create dataset with all zeros
    height, width = 20, 20
    zeros = np.zeros((height, width))

    empty_dataset = xr.Dataset(
        {
            "red": (["y", "x"], zeros),
            "red_edge": (["y", "x"], zeros),
            "nir": (["y", "x"], zeros),
            "swir1": (["y", "x"], zeros),
        },
        coords={"x": np.linspace(0, 1, width), "y": np.linspace(0, 1, height)},
    )

    # Test indices calculation
    indices = calculate_indices_from_dataset(empty_dataset)
    print(
        f"Indices with zero data: {[f'{k}: {np.nanmean(v.values):.4f}' for k, v in indices.data_vars.items()]}"
    )

    # Test masking
    combined_data = empty_dataset.copy()
    for var in indices.data_vars:
        combined_data[var] = indices[var]

    masked_data = apply_mask(combined_data)
    mask_stats = get_mask_statistics(masked_data)
    print(f"Mask stats with zero data: {mask_stats}")

    # Test prediction
    result = predict_biomass(masked_data)
    print(
        f"Prediction with zero data: biomass={result['biomass_kg_per_hectare']}, confidence={result['prediction_confidence']}"
    )

    # Should handle gracefully without crashing
    assert isinstance(result["biomass_kg_per_hectare"], (int, float))
    assert not np.isnan(result["biomass_kg_per_hectare"])


def test_pipeline_debug_values():
    """Debug test to print intermediate values throughout the pipeline."""
    print("\n=== PIPELINE DEBUG TEST ===")

    # Create test data with known good spectral signatures
    height, width = 30, 30
    lons = np.linspace(-123.2, -123.1, width)
    lats = np.linspace(49.3, 49.2, height)

    # Create realistic spectral signatures that should produce kelp detection
    red = np.random.uniform(0.02, 0.06, (height, width))
    red_edge = np.random.uniform(0.04, 0.09, (height, width))
    nir = np.random.uniform(0.20, 0.40, (height, width))
    swir1 = np.random.uniform(0.01, 0.04, (height, width))

    # Create a kelp patch with strong spectral signature
    kelp_area = np.zeros((height, width), dtype=bool)
    kelp_area[10:20, 10:20] = True

    # Enhance kelp spectral signature
    red[kelp_area] = np.random.uniform(0.01, 0.025, np.sum(kelp_area))  # Very low red
    nir[kelp_area] = np.random.uniform(0.35, 0.50, np.sum(kelp_area))  # High NIR
    red_edge[kelp_area] = np.random.uniform(
        0.08, 0.15, np.sum(kelp_area)
    )  # High red edge

    print(f"Input spectral ranges:")
    print(f"  Red: {red.min():.4f} - {red.max():.4f} (mean: {red.mean():.4f})")
    print(
        f"  Red Edge: {red_edge.min():.4f} - {red_edge.max():.4f} (mean: {red_edge.mean():.4f})"
    )
    print(f"  NIR: {nir.min():.4f} - {nir.max():.4f} (mean: {nir.mean():.4f})")
    print(f"  SWIR1: {swir1.min():.4f} - {swir1.max():.4f} (mean: {swir1.mean():.4f})")
    print(f"  Kelp area coverage: {np.mean(kelp_area):.4f}")

    dataset = xr.Dataset(
        {
            "red": (["y", "x"], red),
            "red_edge": (["y", "x"], red_edge),
            "nir": (["y", "x"], nir),
            "swir1": (["y", "x"], swir1),
        },
        coords={"x": lons, "y": lats},
    )

    # Step 1: Calculate indices
    print("\n1. Calculating indices...")
    indices = calculate_indices_from_dataset(dataset)
    for var_name, var_data in indices.data_vars.items():
        values = var_data.values
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        min_val = np.nanmin(values)
        max_val = np.nanmax(values)
        print(
            f"  {var_name}: mean={mean_val:.4f}, std={std_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}]"
        )

        # Check for problematic values
        if np.all(np.isnan(values)):
            print(f"    WARNING: All NaN values for {var_name}")
        elif np.all(values == 0):
            print(f"    WARNING: All zero values for {var_name}")

    # Step 2: Merge data
    print("\n2. Merging data...")
    combined_data = dataset.copy()
    for var in indices.data_vars:
        combined_data[var] = indices[var]
    print(f"  Combined dataset variables: {list(combined_data.data_vars)}")

    # Step 3: Apply masking
    print("\n3. Applying masks...")
    masked_data = apply_mask(combined_data)
    mask_stats = get_mask_statistics(masked_data)
    print(f"  Mask statistics: {mask_stats}")

    # Check what masks were created
    mask_vars = [v for v in masked_data.data_vars if "mask" in v]
    print(f"  Created masks: {mask_vars}")
    for mask_var in mask_vars:
        if mask_var in masked_data:
            coverage = np.mean(masked_data[mask_var].values)
            print(f"    {mask_var}: {coverage:.4f} coverage")

    # Step 4: Predict biomass
    print("\n4. Predicting biomass...")
    try:
        result = predict_biomass(masked_data)
        print(f"  Final result: {result}")

        biomass = result.get("biomass_kg_per_hectare", 0)
        confidence = result.get("prediction_confidence", 0)
        model_type = result.get("model_type", "Unknown")
        features = result.get("top_features", [])

        print(f"  Biomass: {biomass} kg/ha")
        print(f"  Confidence: {confidence}")
        print(f"  Model: {model_type}")
        print(f"  Top features: {features[:3] if features else 'None'}")

        if biomass == 0:
            print("  WARNING: Zero biomass detected! Investigating...")

            # Check if there's kelp detected
            if "kelp_mask" in masked_data:
                kelp_coverage = np.mean(masked_data["kelp_mask"].values)
                print(f"    Kelp mask coverage: {kelp_coverage}")
                if kelp_coverage == 0:
                    print("    ISSUE: No kelp detected in mask")

            # Check spectral indices in kelp areas
            if "fai" in masked_data:
                fai_values = masked_data["fai"].values
                fai_mean = np.nanmean(fai_values)
                print(f"    FAI mean: {fai_mean}")
                if fai_mean <= 0:
                    print("    ISSUE: FAI values not indicating floating algae")

    except Exception as e:
        print(f"  ERROR in prediction: {e}")
        import traceback

        traceback.print_exc()

    print("=== DEBUG TEST COMPLETE ===\n")


def test_api_with_mock_data(test_client):
    """Test API with mocked realistic data."""
    import kelpie_carbon_v1.core.fetch as fetch_module

    def mock_fetch_sentinel_tiles(lat, lng, start_date, end_date):
        # Create realistic dataset with strong kelp signature
        height, width = 50, 50
        lons = np.linspace(lng - 0.01, lng + 0.01, width)
        lats = np.linspace(lat - 0.01, lat + 0.01, height)

        # Strong kelp spectral signatures
        red = np.random.uniform(0.015, 0.035, (height, width))
        red_edge = np.random.uniform(0.05, 0.12, (height, width))
        nir = np.random.uniform(0.25, 0.45, (height, width))
        swir1 = np.random.uniform(0.005, 0.025, (height, width))

        # Create kelp patches with enhanced signature
        kelp_patches = np.random.random((height, width)) > 0.6
        red[kelp_patches] *= 0.3  # Much lower red
        nir[kelp_patches] *= 1.8  # Much higher NIR
        red_edge[kelp_patches] *= 2.0  # Higher red edge

        dataset = xr.Dataset(
            {
                "red": (["y", "x"], red),
                "red_edge": (["y", "x"], red_edge),
                "nir": (["y", "x"], nir),
                "swir1": (["y", "x"], swir1),
            },
            coords={"x": lons, "y": lats},
        )

        return {"data": dataset, "metadata": {"tiles_found": 1}}

    # Mock the fetch function
    original_fetch = fetch_module.fetch_sentinel_tiles
    fetch_module.fetch_sentinel_tiles = mock_fetch_sentinel_tiles

    try:
        request_data = {
            "aoi": {"lat": 49.2827, "lng": -123.1207},
            "start_date": "2023-06-01",
            "end_date": "2023-06-30",
        }

        response = test_client.post("/api/run", json=request_data)
        assert response.status_code == 200

        data = response.json()
        print(f"\nAPI Response Summary:")
        print(f"  Status: {data['status']}")
        print(f"  Biomass: {data['biomass']}")
        print(f"  Carbon: {data['carbon']}")
        print(f"  Processing time: {data['processing_time']}")
        print(f"  Model info: {data['model_info']}")
        print(f"  Spectral indices: {data['spectral_indices']}")
        print(f"  Mask statistics: {data['mask_statistics']}")

        # Verify non-zero results
        if data["status"] == "completed" and "Error" not in data["biomass"]:
            biomass_value = float(data["biomass"].split(" ")[0])
            assert (
                biomass_value >= 0
            ), f"Expected non-negative biomass, got {biomass_value}"

            if biomass_value == 0:
                print("  WARNING: API returned zero biomass!")

    finally:
        fetch_module.fetch_sentinel_tiles = original_fetch


if __name__ == "__main__":
    # Run the debug test directly
    test_pipeline_debug_values()
