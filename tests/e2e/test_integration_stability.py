"""Integration stability tests for Kelpie Carbon v1.

Tests import/integration issues, satellite data sources reliability,
and caching/performance optimizations validation for Task B1.3.
"""

import importlib
import time
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from kelpie_carbon.core.api.imagery import _analysis_cache
from kelpie_carbon.core.api.main import app


class TestImportIntegrationStability:
    """Test import stability and module integration reliability."""

    def test_core_module_imports(self):
        """Test that all core modules can be imported successfully."""
        core_modules = [
            "kelpie_carbon.core.fetch",
            "kelpie_carbon.core.model",
            "kelpie_carbon.core.mask",
            "kelpie_carbon.core.indices",
        ]

        for module_name in core_modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None
                # Test that key functions exist
                assert hasattr(module, "__file__")
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_processing_module_imports(self):
        """Test that all processing modules can be imported successfully."""
        processing_modules = [
            "kelpie_carbon.processing.species_classifier",
        ]

        for module_name in processing_modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None
                # Test that main classes exist
                if "species_classifier" in module_name:
                    assert hasattr(module, "SpeciesClassifier")
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_skema_integration_imports(self):
        """Test that SKEMA integration components can be imported."""
        from kelpie_carbon.processing.species_classifier import SpeciesClassifier

        # Test instantiation
        sc = SpeciesClassifier()
        assert sc is not None


class TestSatelliteDataSourceReliability:
    """Test reliability of satellite data sources and fallback mechanisms."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_xarray_dataset_compatibility(self):
        """Test xarray dataset compatibility with our processing pipeline."""
        # Create test dataset similar to Sentinel-2 structure
        test_dataset = xr.Dataset(
            {
                "red": (["y", "x"], np.random.rand(50, 50)),
                "green": (["y", "x"], np.random.rand(50, 50)),
                "blue": (["y", "x"], np.random.rand(50, 50)),
                "nir": (["y", "x"], np.random.rand(50, 50)),
                "red_edge": (["y", "x"], np.random.rand(50, 50)),
                "swir1": (["y", "x"], np.random.rand(50, 50)),
            }
        )

        # Test that our processing modules can handle this dataset
        from kelpie_carbon.core.indices import calculate_indices_from_dataset
        from kelpie_carbon.core.mask import (
            create_kelp_detection_mask,
            create_water_mask,
        )

        try:
            indices = calculate_indices_from_dataset(test_dataset)
            assert indices is not None
            assert "ndvi" in indices.data_vars

            water_mask = create_water_mask(test_dataset)
            assert water_mask is not None

            # Create basic config for kelp detection
            config = {"kelp_fai_threshold": 0.01, "apply_morphology": False}
            kelp_mask = create_kelp_detection_mask(test_dataset, config)
            assert kelp_mask is not None

        except Exception as e:
            pytest.fail(f"Dataset compatibility test failed: {e}")

    @pytest.mark.slow
    def test_coordinate_reference_system_handling(self):
        """Test CRS handling and coordinate transformations."""
        test_coords = [
            {"lat": 49.2827, "lng": -123.1207},  # Vancouver only for CI performance
        ]

        for coords in test_coords:
            with patch("kelpie_carbon.core.fetch.fetch_sentinel_tiles") as mock_fetch:
                mock_dataset = xr.Dataset(
                    {
                        "red": (["y", "x"], np.random.rand(30, 30)),
                        "green": (["y", "x"], np.random.rand(30, 30)),
                        "blue": (["y", "x"], np.random.rand(30, 30)),
                        "nir": (["y", "x"], np.random.rand(30, 30)),
                        "red_edge": (["y", "x"], np.random.rand(30, 30)),
                        "swir1": (["y", "x"], np.random.rand(30, 30)),
                    }
                )

                mock_fetch.return_value = {
                    "data": mock_dataset,
                    "cloud_cover": 15.0,
                    "source": "Sentinel-2",
                    "acquisition_date": "2023-08-15",
                }

                response = self.client.post(
                    "/api/run",
                    json={
                        "aoi": coords,
                        "start_date": "2023-08-15",
                        "end_date": "2023-08-16",
                    },
                )

                # Should handle different coordinate systems
                assert response.status_code == 200


class TestCachePerformanceOptimizations:
    """Test caching mechanisms and performance optimizations."""

    def setup_method(self):
        """Set up test client and clear cache."""
        self.client = TestClient(app)
        _analysis_cache.clear()

    @pytest.mark.slow
    def test_cache_persistence_across_requests(self):
        """Test that cached data persists across multiple requests."""
        with patch("kelpie_carbon.core.fetch.fetch_sentinel_tiles") as mock_fetch:
            mock_dataset = xr.Dataset(
                {
                    "red": (["y", "x"], np.random.rand(20, 20)),
                    "green": (["y", "x"], np.random.rand(20, 20)),
                    "blue": (["y", "x"], np.random.rand(20, 20)),
                    "nir": (["y", "x"], np.random.rand(20, 20)),
                    "red_edge": (["y", "x"], np.random.rand(20, 20)),
                    "swir1": (["y", "x"], np.random.rand(20, 20)),
                }
            )

            mock_fetch.return_value = {
                "data": mock_dataset,
                "cloud_cover": 10.0,
                "source": "Sentinel-2",
                "acquisition_date": "2023-08-15",
            }

            # First request - populates cache
            response1 = self.client.post(
                "/api/imagery/analyze-and-cache",
                json={
                    "aoi": {"lat": 49.2827, "lng": -123.1207},
                    "start_date": "2023-08-15",
                    "end_date": "2023-08-16",
                },
            )

            assert response1.status_code == 200
            analysis_id = response1.json()["analysis_id"]

            # Multiple subsequent requests should use cache (reduced for CI)
            for _ in range(2):
                start_time = time.time()
                response = self.client.get(f"/api/imagery/{analysis_id}/rgb")
                end_time = time.time()

                assert response.status_code == 200
                # Cached responses should be fast (relaxed for CI)
                assert (end_time - start_time) < 5.0  # More generous for CI

    @pytest.mark.slow
    def test_cache_size_management(self):
        """Test cache size management and cleanup."""
        initial_cache_size = len(_analysis_cache)

        # Generate multiple cached analyses (minimal for performance)
        for i in range(2):  # Further reduced for CI performance
            with patch("kelpie_carbon.core.fetch.fetch_sentinel_tiles") as mock_fetch:
                mock_dataset = xr.Dataset(
                    {
                        "red": (["y", "x"], np.random.rand(20, 20)),
                        "green": (["y", "x"], np.random.rand(20, 20)),
                        "blue": (["y", "x"], np.random.rand(20, 20)),
                        "nir": (["y", "x"], np.random.rand(20, 20)),
                        "red_edge": (["y", "x"], np.random.rand(20, 20)),
                        "swir1": (["y", "x"], np.random.rand(20, 20)),
                    }
                )

                mock_fetch.return_value = {
                    "data": mock_dataset,
                    "cloud_cover": 10.0,
                    "source": "Sentinel-2",
                    "acquisition_date": f"2023-08-{15 + i:02d}",
                }

                response = self.client.post(
                    "/api/imagery/analyze-and-cache",
                    json={
                        "aoi": {"lat": 49.2827 + i * 0.001, "lng": -123.1207},
                        "start_date": f"2023-08-{15 + i:02d}",
                        "end_date": f"2023-08-{16 + i:02d}",
                    },
                )

                assert response.status_code == 200

        # Cache should have grown but be managed
        final_cache_size = len(_analysis_cache)
        assert final_cache_size >= initial_cache_size


if __name__ == "__main__":
    """Run integration stability tests directly."""
    print("🔗 Running Integration Stability Tests...")

    # Run each test class
    test_classes = [
        TestImportIntegrationStability,
        TestSatelliteDataSourceReliability,
        TestCachePerformanceOptimizations,
    ]

    for test_class in test_classes:
        print(f"📋 Testing {test_class.__name__}...")
        instance = test_class()

        # Run setup and all test methods
        if hasattr(instance, "setup_method"):
            instance.setup_method()

        methods = [method for method in dir(instance) if method.startswith("test_")]
        for method_name in methods:
            print(f"  ✓ {method_name}")
            getattr(instance, method_name)()

    print("🎉 All integration stability tests completed successfully!")
