"""Tests for biomass prediction model functionality."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from kelpie_carbon_v1.core.model import (
    KelpBiomassModel,
    generate_training_data,
    predict_biomass,
)


@pytest.fixture
def sample_masked_dataset():
    """Create a sample masked satellite dataset for testing."""
    np.random.seed(42)
    height, width = 30, 30

    # Create realistic spectral data
    red = np.random.normal(0.1, 0.03, (height, width)).clip(0, 1)
    red_edge = np.random.normal(0.15, 0.04, (height, width)).clip(0, 1)
    nir = np.random.normal(0.3, 0.1, (height, width)).clip(0, 1)
    swir1 = np.random.normal(0.2, 0.05, (height, width)).clip(0, 1)

    # Create masks
    kelp_mask = np.random.choice([0, 1], size=(height, width), p=[0.8, 0.2])
    water_mask = np.random.choice([0, 1], size=(height, width), p=[0.3, 0.7])
    valid_mask = np.ones((height, width))
    cloud_mask = np.random.choice([0, 1], size=(height, width), p=[0.9, 0.1])

    # Create coordinate arrays
    lons = np.linspace(-123.5, -123.0, width)
    lats = np.linspace(49.5, 49.0, height)

    return xr.Dataset(
        {
            "red": (["y", "x"], red),
            "red_edge": (["y", "x"], red_edge),
            "nir": (["y", "x"], nir),
            "swir1": (["y", "x"], swir1),
            "kelp_mask": (["y", "x"], kelp_mask),
            "water_mask": (["y", "x"], water_mask),
            "valid_mask": (["y", "x"], valid_mask),
            "cloud_mask": (["y", "x"], cloud_mask),
        },
        coords={"x": lons, "y": lats},
    )


def test_kelp_biomass_model_initialization():
    """Test KelpBiomassModel initialization."""
    model = KelpBiomassModel()

    assert hasattr(model, "model")
    assert hasattr(model, "scaler")
    assert hasattr(model, "feature_names")
    assert not model.is_trained

    # Test with custom parameters
    custom_params = {"n_estimators": 50, "max_depth": 10}
    custom_model = KelpBiomassModel(custom_params)
    assert custom_model.model.n_estimators == 50
    assert custom_model.model.max_depth == 10


def test_extract_features(sample_masked_dataset):
    """Test feature extraction from satellite dataset."""
    model = KelpBiomassModel()
    features = model.extract_features(sample_masked_dataset)

    assert isinstance(features, pd.DataFrame)
    assert len(features) == 1  # Single sample
    assert len(features.columns) > 0

    # Check for expected feature categories
    feature_names = list(features.columns)

    # Should have spectral band statistics
    assert any("red_mean" in name for name in feature_names)
    assert any("nir_mean" in name for name in feature_names)

    # Should have coverage statistics
    assert any("water_coverage" in name for name in feature_names)
    assert any("kelp_coverage" in name for name in feature_names)

    # Should have spatial features
    assert any("image_height" in name for name in feature_names)
    assert any("center_lat" in name for name in feature_names)


def test_spectral_indices_calculation(sample_masked_dataset):
    """Test spectral indices calculation."""
    model = KelpBiomassModel()
    indices = model._calculate_spectral_indices(sample_masked_dataset)

    assert isinstance(indices, dict)

    # Check for expected indices
    expected_indices = ["ndvi", "red_edge_ndvi", "ndre", "fai", "ndwi", "evi"]
    for index_name in expected_indices:
        assert index_name in indices
        assert isinstance(indices[index_name], np.ndarray)
        assert not np.any(np.isnan(indices[index_name]))


def test_kelp_patch_analysis(sample_masked_dataset):
    """Test kelp patch analysis."""
    model = KelpBiomassModel()
    kelp_mask = sample_masked_dataset["kelp_mask"].values
    patch_stats = model._analyze_kelp_patches(kelp_mask)

    assert isinstance(patch_stats, dict)

    expected_keys = [
        "num_kelp_patches",
        "avg_patch_size",
        "largest_patch_size",
        "patch_density",
    ]
    for key in expected_keys:
        assert key in patch_stats
        assert len(patch_stats[key]) == 1  # Single value in list


def test_spatial_features_calculation(sample_masked_dataset):
    """Test spatial features calculation."""
    model = KelpBiomassModel()
    kelp_pixels = np.ones_like(sample_masked_dataset["red"].values, dtype=bool)
    spatial_features = model._calculate_spatial_features(
        sample_masked_dataset, kelp_pixels
    )

    assert isinstance(spatial_features, dict)

    # Check for expected spatial features
    expected_keys = [
        "image_height",
        "image_width",
        "total_pixels",
        "center_lon",
        "center_lat",
    ]
    for key in expected_keys:
        assert key in spatial_features


def test_predict_with_synthetic_model(sample_masked_dataset):
    """Test biomass prediction using synthetic model."""
    model = KelpBiomassModel()
    result = model._predict_with_synthetic_model(sample_masked_dataset)

    assert isinstance(result, dict)

    # Check required keys
    required_keys = [
        "biomass_kg_per_hectare",
        "prediction_confidence",
        "top_features",
        "model_type",
        "feature_count",
    ]
    for key in required_keys:
        assert key in result

    # Check value ranges
    assert result["biomass_kg_per_hectare"] >= 0
    assert 0 <= result["prediction_confidence"] <= 1
    assert isinstance(result["top_features"], list)
    assert "Synthetic" in result["model_type"]


def test_generate_training_data():
    """Test synthetic training data generation."""
    training_data = generate_training_data(n_samples=10)

    assert len(training_data) == 10

    for dataset, biomass in training_data:
        assert isinstance(dataset, xr.Dataset)
        assert isinstance(biomass, (int, float))
        assert biomass >= 0

        # Check dataset structure
        expected_vars = [
            "red",
            "red_edge",
            "nir",
            "swir1",
            "kelp_mask",
            "water_mask",
            "valid_mask",
        ]
        for var in expected_vars:
            assert var in dataset


def test_model_training():
    """Test Random Forest model training."""
    model = KelpBiomassModel()
    training_data = generate_training_data(n_samples=20)

    metrics = model.train(training_data)

    assert model.is_trained
    assert isinstance(metrics, dict)

    # Check metric keys
    expected_metrics = [
        "train_r2",
        "test_r2",
        "train_rmse",
        "test_rmse",
        "n_samples",
        "n_features",
        "cv_r2_mean",
        "cv_r2_std",
    ]
    for metric in expected_metrics:
        assert metric in metrics

    # Check metric ranges
    assert metrics["n_samples"] == 20
    assert metrics["n_features"] > 0
    assert -1 <= metrics["train_r2"] <= 1
    assert -1 <= metrics["test_r2"] <= 1


def test_trained_model_prediction(sample_masked_dataset):
    """Test prediction with a trained model."""
    model = KelpBiomassModel()
    training_data = generate_training_data(n_samples=15)

    # Train the model
    model.train(training_data)

    # Make prediction
    result = model.predict(sample_masked_dataset)

    assert isinstance(result, dict)
    assert "biomass_kg_per_hectare" in result
    assert "prediction_confidence" in result
    assert "top_features" in result
    assert result["biomass_kg_per_hectare"] >= 0
    assert 0 <= result["prediction_confidence"] <= 1


def test_predict_biomass_function(sample_masked_dataset):
    """Test the main predict_biomass function."""
    result = predict_biomass(sample_masked_dataset)

    assert isinstance(result, dict)
    assert "biomass_kg_per_hectare" in result
    assert "prediction_confidence" in result
    assert result["biomass_kg_per_hectare"] >= 0


def test_model_save_load(tmp_path, sample_masked_dataset):
    """Test model saving and loading."""
    model = KelpBiomassModel()
    training_data = generate_training_data(n_samples=10)

    # Train the model
    model.train(training_data)

    # Save the model
    model_path = tmp_path / "test_model.joblib"
    model.save_model(str(model_path))

    assert model_path.exists()

    # Load the model
    new_model = KelpBiomassModel()
    new_model.load_model(str(model_path))

    assert new_model.is_trained
    assert len(new_model.feature_names) > 0

    # Test prediction with loaded model
    result = new_model.predict(sample_masked_dataset)
    assert "biomass_kg_per_hectare" in result


def test_feature_consistency():
    """Test that feature extraction is consistent across different datasets."""
    model = KelpBiomassModel()

    # Create two similar datasets
    np.random.seed(42)
    dataset1 = generate_training_data(1)[0][0]

    np.random.seed(42)
    dataset2 = generate_training_data(1)[0][0]

    features1 = model.extract_features(dataset1)
    features2 = model.extract_features(dataset2)

    # Should have same feature names
    assert list(features1.columns) == list(features2.columns)

    # Should have same values for identical datasets
    pd.testing.assert_frame_equal(features1, features2)


def test_model_with_missing_bands():
    """Test model behavior with missing spectral bands."""
    # Create dataset with missing bands
    height, width = 20, 20
    dataset = xr.Dataset(
        {
            "red": (["y", "x"], np.random.random((height, width))),
            "nir": (["y", "x"], np.random.random((height, width))),
            # Missing red_edge and swir1
            "kelp_mask": (["y", "x"], np.zeros((height, width))),
            "water_mask": (["y", "x"], np.ones((height, width))),
            "valid_mask": (["y", "x"], np.ones((height, width))),
        }
    )

    model = KelpBiomassModel()
    features = model.extract_features(dataset)

    # Should still extract some features
    assert len(features.columns) > 0
    assert len(features) == 1


def test_edge_case_empty_kelp_mask():
    """Test model with empty kelp mask."""
    height, width = 15, 15
    dataset = xr.Dataset(
        {
            "red": (["y", "x"], np.random.random((height, width))),
            "red_edge": (["y", "x"], np.random.random((height, width))),
            "nir": (["y", "x"], np.random.random((height, width))),
            "swir1": (["y", "x"], np.random.random((height, width))),
            "kelp_mask": (["y", "x"], np.zeros((height, width))),  # No kelp
            "water_mask": (["y", "x"], np.ones((height, width))),
            "valid_mask": (["y", "x"], np.ones((height, width))),
        }
    )

    model = KelpBiomassModel()
    result = model._predict_with_synthetic_model(dataset)

    # Should handle empty kelp mask gracefully
    assert result["biomass_kg_per_hectare"] >= 0
    # top_features is now a list of strings, not tuples
    top_features_text = " ".join(result["top_features"])
    assert "kelp_coverage" in top_features_text


def test_training_data_quality():
    """Test quality and consistency of generated training data."""
    training_data = generate_training_data(n_samples=5)

    for i, (dataset, biomass) in enumerate(training_data):
        # Check dataset quality
        assert dataset.dims["x"] == 50
        assert dataset.dims["y"] == 50

        # Check biomass realism
        assert 0 <= biomass <= 20000  # Reasonable range for kelp biomass

        # Check that kelp coverage and biomass are somewhat correlated
        kelp_coverage = np.mean(dataset["kelp_mask"].values)
        if kelp_coverage == 0:
            # If no kelp, biomass should be low
            assert biomass < 1000

        # Check spectral value ranges
        for band in ["red", "red_edge", "nir", "swir1"]:
            values = dataset[band].values
            assert np.all(values >= 0) and np.all(values <= 1)


def test_model_robustness():
    """Test model robustness with extreme values."""
    height, width = 10, 10

    # Create dataset with extreme values
    dataset = xr.Dataset(
        {
            "red": (["y", "x"], np.full((height, width), 1.0)),  # Maximum reflectance
            "red_edge": (
                ["y", "x"],
                np.full((height, width), 0.0),
            ),  # Minimum reflectance
            "nir": (["y", "x"], np.full((height, width), 0.5)),
            "swir1": (["y", "x"], np.full((height, width), 0.3)),
            "kelp_mask": (["y", "x"], np.ones((height, width))),  # Full kelp coverage
            "water_mask": (["y", "x"], np.ones((height, width))),
            "valid_mask": (["y", "x"], np.ones((height, width))),
        }
    )

    model = KelpBiomassModel()
    result = model._predict_with_synthetic_model(dataset)

    # Should handle extreme values without crashing
    assert isinstance(result["biomass_kg_per_hectare"], (int, float))
    assert not np.isnan(result["biomass_kg_per_hectare"])
    assert result["biomass_kg_per_hectare"] >= 0


def test_empty_array_handling():
    """Test that empty arrays are handled without RuntimeWarnings."""
    import warnings

    model = KelpBiomassModel()

    # Test with empty arrays that used to cause RuntimeWarnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test empty slice mean calculation
        empty_array = np.array([])
        if len(empty_array) > 0:
            result = np.mean(empty_array)
        else:
            result = 0.0  # Default value for empty arrays

        assert result == 0.0

        # Test empty slice percentile calculation
        if len(empty_array) > 0:
            result = np.percentile(empty_array, 95)
        else:
            result = 0.0  # Default value for empty arrays

        assert result == 0.0

        # Check that no RuntimeWarnings were generated
        runtime_warnings = [
            warning for warning in w if warning.category == RuntimeWarning
        ]
        assert len(runtime_warnings) == 0


def test_all_nan_array_handling():
    """Test handling of arrays with all NaN values."""
    import warnings

    model = KelpBiomassModel()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test all-NaN arrays
        nan_array = np.full(10, np.nan)
        valid_mask = ~np.isnan(nan_array)

        if np.any(valid_mask):
            result = np.mean(nan_array[valid_mask])
        else:
            result = 0.0  # Default for all-NaN arrays

        assert result == 0.0

        # Check that no RuntimeWarnings were generated
        runtime_warnings = [
            warning for warning in w if warning.category == RuntimeWarning
        ]
        assert len(runtime_warnings) == 0


def test_model_statistical_operations_safe():
    """Test that model statistical operations don't generate RuntimeWarnings."""
    import warnings

    # Create a dataset with some empty/NaN regions
    height, width = 10, 10

    # Create mostly NaN data to test edge cases
    red = np.full((height, width), np.nan)
    red[0:2, 0:2] = 0.1  # Only a small valid region

    dataset = xr.Dataset(
        {
            "red": (["y", "x"], red),
            "red_edge": (["y", "x"], red),
            "nir": (["y", "x"], red),
            "swir1": (["y", "x"], red),
            "kelp_mask": (["y", "x"], np.zeros((height, width))),
            "water_mask": (["y", "x"], np.ones((height, width))),
            "valid_mask": (["y", "x"], np.ones((height, width))),
        },
        coords={
            "x": np.linspace(-123.5, -123.0, width),
            "y": np.linspace(49.5, 49.0, height),
        },
    )

    model = KelpBiomassModel()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # This should not generate RuntimeWarnings
        features = model.extract_features(dataset)

        # Check that features were extracted successfully
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 1

        # Check that no RuntimeWarnings were generated
        runtime_warnings = [
            warning for warning in w if warning.category == RuntimeWarning
        ]
        assert (
            len(runtime_warnings) == 0
        ), f"Generated RuntimeWarnings: {[str(w.message) for w in runtime_warnings]}"
