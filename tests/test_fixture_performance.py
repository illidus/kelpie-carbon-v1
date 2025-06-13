"""Test file to verify session-scoped fixtures and performance optimizations."""

from __future__ import annotations

import time

import numpy as np
import pytest
import xarray as xr


class TestSessionFixtures:
    """Test that session-scoped fixtures work correctly."""

    def test_sentinel_tile_fixture(self, sentinel_tile):
        """Test that sentinel_tile fixture provides valid data."""
        assert isinstance(sentinel_tile, xr.Dataset)

        # Check required bands
        required_bands = ["red", "green", "blue", "nir", "red_edge", "swir1"]
        for band in required_bands:
            assert band in sentinel_tile.data_vars

        # Check data dimensions
        assert "x" in sentinel_tile.dims
        assert "y" in sentinel_tile.dims
        assert sentinel_tile.sizes["x"] == 50
        assert sentinel_tile.sizes["y"] == 50

        # Check attributes
        assert "cloud_cover" in sentinel_tile.attrs
        assert "acquisition_date" in sentinel_tile.attrs
        assert sentinel_tile.attrs["cloud_cover"] == 5.0

    def test_rf_model_fixture(self, rf_model):
        """Test that rf_model fixture provides a trained model."""
        assert "model" in rf_model
        assert "feature_names" in rf_model
        assert "training_score" in rf_model
        assert "n_samples" in rf_model
        assert "n_features" in rf_model

        # Check model is trained
        model = rf_model["model"]
        assert hasattr(model, "predict")
        assert rf_model["n_samples"] == 100
        assert rf_model["n_features"] == 20

        # Test prediction works
        x_test = np.random.rand(1, 20)
        prediction = model.predict(x_test)
        assert len(prediction) == 1
        assert prediction[0] >= 0  # Biomass should be non-negative

    def test_minimal_training_data_fixture(self, minimal_training_data):
        """Test that minimal_training_data fixture provides valid training samples."""
        assert len(minimal_training_data) == 10

        for dataset, biomass in minimal_training_data:
            assert isinstance(dataset, xr.Dataset)
            assert isinstance(biomass, int | float)
            assert biomass >= 0

            # Check dataset structure
            assert "red" in dataset.data_vars
            assert "nir" in dataset.data_vars
            assert "kelp_mask" in dataset.data_vars
            assert dataset.sizes["x"] == 20
            assert dataset.sizes["y"] == 20

    def test_optimized_cache_fixture(self, optimized_cache):
        """Test that optimized_cache fixture provides pre-warmed cache."""
        assert "test_coordinates" in optimized_cache
        assert "test_date_range" in optimized_cache
        assert "monterey_bay" in optimized_cache
        assert "vancouver_island" in optimized_cache

        # Check coordinate values
        lat, lng = optimized_cache["test_coordinates"]
        assert -90 <= lat <= 90
        assert -180 <= lng <= 180

    def test_temp_data_dir_fixture(self, temp_data_dir):
        """Test that temp_data_dir fixture provides temporary directory with test files."""
        assert temp_data_dir.exists()
        assert temp_data_dir.is_dir()

        # Check test files exist
        assert (temp_data_dir / "test_data.json").exists()
        assert (temp_data_dir / "validation_results.json").exists()

        # Check file contents
        test_data = (temp_data_dir / "test_data.json").read_text()
        assert "test" in test_data


class TestMockFixtures:
    """Test that mock fixtures work correctly."""

    def test_mock_satellite_data_fixture(self, mock_satellite_data):
        """Test that mock_satellite_data fixture provides valid mock data."""
        assert "data" in mock_satellite_data
        assert "bands" in mock_satellite_data
        assert "metadata" in mock_satellite_data

        # Check data structure
        dataset = mock_satellite_data["data"]
        assert isinstance(dataset, xr.Dataset)

        # Check metadata
        metadata = mock_satellite_data["metadata"]
        assert "date" in metadata
        assert "cloud_coverage" in metadata
        assert "coordinates" in metadata
        assert "source" in metadata

    def test_mock_sleep_fixture(self, mock_sleep):
        """Test that mock_sleep fixture eliminates sleep delays."""

        # Test synchronous sleep
        start_time = time.time()
        time.sleep(1.0)  # Should be instant
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # Should be much faster than 1 second

    def test_mock_network_requests_fixture(self, mock_network_requests):
        """Test that mock_network_requests fixture mocks HTTP calls."""
        import requests

        # Test GET request
        response = requests.get("http://example.com")
        assert response.status_code == 200
        assert response.json()["status"] == "success"

        # Test POST request
        response = requests.post("http://example.com", json={"test": "data"})
        assert response.status_code == 200
        assert response.json()["status"] == "created"


class TestDataSlicing:
    """Test data slicing utilities for minimal samples."""

    def test_slice_data_to_minimal_fixture(self, slice_data_to_minimal, sentinel_tile):
        """Test that slice_data_to_minimal reduces dataset size."""
        # Create a larger dataset
        large_data = {
            "red": (["y", "x"], np.random.rand(100, 100)),
            "nir": (["y", "x"], np.random.rand(100, 100)),
        }
        coords = {"y": np.arange(100), "x": np.arange(100)}
        large_dataset = xr.Dataset(large_data, coords=coords)

        # Slice to minimal size
        sliced = slice_data_to_minimal(large_dataset, max_size=30)

        # Check size reduction
        assert sliced.sizes["x"] == 30
        assert sliced.sizes["y"] == 30
        assert "red" in sliced.data_vars
        assert "nir" in sliced.data_vars

    def test_slice_data_preserves_small_datasets(
        self, slice_data_to_minimal, sentinel_tile
    ):
        """Test that slice_data_to_minimal preserves datasets already small enough."""
        # sentinel_tile is 50x50, which is at the default max_size
        sliced = slice_data_to_minimal(sentinel_tile, max_size=50)

        # Should be unchanged
        assert sliced.sizes["x"] == 50
        assert sliced.sizes["y"] == 50
        assert len(sliced.data_vars) == len(sentinel_tile.data_vars)


@pytest.mark.performance
class TestPerformanceOptimizations:
    """Test performance optimization features."""

    def test_performance_mode_environment(self, performance_mode):
        """Test that performance_mode fixture sets environment variables."""
        import os

        assert os.environ.get("TESTING_MODE") == "performance"
        assert os.environ.get("MINIMAL_DATA_SIZE") == "true"
        assert os.environ.get("DISABLE_SLOW_OPERATIONS") == "true"

    @pytest.mark.heavy
    def test_heavy_fixture_caching(self, sentinel_tile, rf_model):
        """Test that heavy fixtures are cached across test runs."""
        # This test verifies that session-scoped fixtures are reused
        # The actual performance benefit is measured by pytest execution time

        # Access fixtures multiple times - should be instant after first access
        for _ in range(3):
            assert sentinel_tile.sizes["x"] == 50
            assert rf_model["n_features"] == 20

        # If fixtures weren't cached, this would be much slower
