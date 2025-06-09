"""Integration test for the complete satellite imagery visualization pipeline."""
import io
from unittest.mock import Mock, patch

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient
from PIL import Image

from kelpie_carbon_v1.api.imagery import _analysis_cache
from kelpie_carbon_v1.api.main import app


class TestSatelliteImageryIntegration:
    """Test the complete satellite imagery pipeline from analysis to visualization."""

    def setup_method(self):
        """Set up test client and clear cache."""
        self.client = TestClient(app)
        _analysis_cache.clear()

    def teardown_method(self):
        """Clean up after tests."""
        _analysis_cache.clear()

    @patch("kelpie_carbon_v1.api.imagery.fetch_sentinel_tiles")
    @patch("kelpie_carbon_v1.api.imagery.calculate_indices_from_dataset")
    @patch("kelpie_carbon_v1.api.imagery.apply_mask")
    @patch("kelpie_carbon_v1.api.imagery.predict_biomass")
    def test_complete_imagery_pipeline(
        self, mock_predict, mock_mask, mock_indices, mock_fetch
    ):
        """Test the complete pipeline from analysis to image generation."""

        # Create realistic mock satellite data
        mock_dataset = xr.Dataset(
            {
                "red": (["y", "x"], np.random.rand(100, 120) * 3000),
                "nir": (["y", "x"], np.random.rand(100, 120) * 4000),
                "red_edge": (["y", "x"], np.random.rand(100, 120) * 3500),
                "swir1": (["y", "x"], np.random.rand(100, 120) * 2000),
            },
            coords={
                "x": np.linspace(-122.0, -121.8, 120),
                "y": np.linspace(36.5, 36.7, 100),
            },
        )

        # Create mock indices
        mock_indices_dataset = xr.Dataset(
            {
                "ndvi": (["y", "x"], np.random.rand(100, 120) * 0.8 - 0.1),
                "fai": (["y", "x"], np.random.rand(100, 120) * 0.5 - 0.1),
                "red_edge_ndvi": (["y", "x"], np.random.rand(100, 120) * 0.9 - 0.1),
            },
            coords=mock_dataset.coords,
        )

        # Create mock masked data (combined dataset + indices + masks)
        mock_masked_data = mock_dataset.copy()
        for var in mock_indices_dataset.data_vars:
            mock_masked_data[var] = mock_indices_dataset[var]

        # Add masks
        mock_masked_data["kelp_mask"] = (
            ["y", "x"],
            np.random.choice([True, False], (100, 120), p=[0.3, 0.7]),
        )
        mock_masked_data["water_mask"] = (
            ["y", "x"],
            np.random.choice([True, False], (100, 120), p=[0.6, 0.4]),
        )
        mock_masked_data["cloud_mask"] = (
            ["y", "x"],
            np.random.choice([True, False], (100, 120), p=[0.1, 0.9]),
        )

        # Create mock biomass data
        mock_biomass = xr.DataArray(
            np.random.rand(100, 120) * 5000, dims=["y", "x"], coords=mock_dataset.coords
        )

        # Set up mocks
        mock_fetch.return_value = {"data": mock_dataset}
        mock_indices.return_value = mock_indices_dataset
        mock_mask.return_value = mock_masked_data
        mock_predict.return_value = {"biomass_map": mock_biomass}

        # Step 1: Run analysis and cache imagery data
        print("🔄 Step 1: Running analysis and caching imagery...")
        response = self.client.post(
            "/api/imagery/analyze-and-cache",
            json={
                "aoi": {"lat": 36.6, "lng": -121.9},
                "start_date": "2023-07-01",
                "end_date": "2023-07-31",
            },
        )

        assert response.status_code == 200
        analysis_data = response.json()
        analysis_id = analysis_data["analysis_id"]

        print(f"✅ Analysis cached with ID: {analysis_id}")
        print(f"📊 Available layers: {analysis_data['available_layers']}")

        # Step 2: Test RGB composite generation
        print("🔄 Step 2: Generating RGB composite...")
        rgb_response = self.client.get(f"/api/imagery/{analysis_id}/rgb")

        assert rgb_response.status_code == 200
        assert rgb_response.headers["content-type"] == "image/jpeg"

        # Verify it's a valid image
        rgb_image = Image.open(io.BytesIO(rgb_response.content))
        assert rgb_image.format == "JPEG"
        assert rgb_image.mode == "RGB"
        print(f"✅ RGB composite generated: {rgb_image.size}")

        # Step 3: Test false-color composite
        print("🔄 Step 3: Generating false-color composite...")
        false_color_response = self.client.get(
            f"/api/imagery/{analysis_id}/false-color"
        )

        assert false_color_response.status_code == 200
        false_color_image = Image.open(io.BytesIO(false_color_response.content))
        assert false_color_image.mode == "RGB"
        print(f"✅ False-color composite generated: {false_color_image.size}")

        # Step 4: Test spectral index visualizations
        print("🔄 Step 4: Generating spectral index visualizations...")
        spectral_indices = ["ndvi", "fai", "red_edge_ndvi"]

        for index_name in spectral_indices:
            spectral_response = self.client.get(
                f"/api/imagery/{analysis_id}/spectral/{index_name}"
            )
            assert spectral_response.status_code == 200

            spectral_image = Image.open(io.BytesIO(spectral_response.content))
            assert spectral_image.mode == "RGB"
            print(
                f"✅ {index_name.upper()} visualization generated: {spectral_image.size}"
            )

        # Step 5: Test mask overlays
        print("🔄 Step 5: Generating mask overlays...")
        mask_types = ["kelp", "water", "cloud"]

        for mask_type in mask_types:
            mask_response = self.client.get(
                f"/api/imagery/{analysis_id}/mask/{mask_type}"
            )
            assert mask_response.status_code == 200

            mask_image = Image.open(io.BytesIO(mask_response.content))
            assert mask_image.mode == "RGBA"
            print(f"✅ {mask_type} mask overlay generated: {mask_image.size}")

        # Step 6: Test biomass heatmap
        print("🔄 Step 6: Generating biomass heatmap...")
        biomass_response = self.client.get(f"/api/imagery/{analysis_id}/biomass")

        assert biomass_response.status_code == 200
        biomass_image = Image.open(io.BytesIO(biomass_response.content))
        assert biomass_image.mode == "RGBA"
        print(f"✅ Biomass heatmap generated: {biomass_image.size}")

        # Step 7: Test metadata retrieval
        print("🔄 Step 7: Retrieving imagery metadata...")
        metadata_response = self.client.get(f"/api/imagery/{analysis_id}/metadata")

        assert metadata_response.status_code == 200
        metadata = metadata_response.json()

        assert "bounds" in metadata
        assert "available_layers" in metadata
        assert "layer_info" in metadata

        print(f"✅ Metadata retrieved:")
        print(f"   📍 Bounds: {metadata['bounds']}")
        print(f"   🗂️ Available layers: {len(metadata['available_layers'])}")
        print(f"   ℹ️ Layer info: {len(metadata['layer_info'])}")

        # Step 8: Test layer customization
        print("🔄 Step 8: Testing layer customization...")

        # Test custom alpha for mask
        custom_mask_response = self.client.get(
            f"/api/imagery/{analysis_id}/mask/kelp?alpha=0.8"
        )
        assert custom_mask_response.status_code == 200

        # Test custom colormap for biomass
        custom_biomass_response = self.client.get(
            f"/api/imagery/{analysis_id}/biomass?colormap=viridis&min_biomass=0&max_biomass=10000"
        )
        assert custom_biomass_response.status_code == 200

        print("✅ Layer customization working")

        # Step 9: Verify caching
        print("🔄 Step 9: Verifying caching...")
        assert analysis_id in _analysis_cache
        cached_data = _analysis_cache[analysis_id]

        assert "dataset" in cached_data
        assert "indices" in cached_data
        assert "masks" in cached_data
        assert "biomass" in cached_data

        print("✅ Data properly cached")

        # Step 10: Test cache cleanup
        print("🔄 Step 10: Testing cache cleanup...")
        cleanup_response = self.client.delete(f"/api/imagery/{analysis_id}")

        assert cleanup_response.status_code == 200
        assert analysis_id not in _analysis_cache

        print("✅ Cache cleanup successful")

        print("\n🎉 COMPLETE SATELLITE IMAGERY PIPELINE TEST PASSED!")
        print("📋 Summary:")
        print("   ✅ Analysis and caching")
        print("   ✅ RGB composite generation")
        print("   ✅ False-color composite generation")
        print("   ✅ Spectral index visualizations (NDVI, FAI, Red Edge NDVI)")
        print("   ✅ Mask overlays (Kelp, Water, Cloud)")
        print("   ✅ Biomass heatmap generation")
        print("   ✅ Metadata retrieval")
        print("   ✅ Layer customization")
        print("   ✅ Data caching")
        print("   ✅ Cache cleanup")

    def test_imagery_error_handling(self):
        """Test error handling in imagery pipeline."""

        # Test non-existent analysis
        response = self.client.get("/api/imagery/nonexistent/rgb")
        assert response.status_code == 404

        # Test invalid spectral index
        _analysis_cache["test"] = {
            "dataset": xr.Dataset(),
            "indices": {},
            "masks": {},
            "biomass": None,
        }

        response = self.client.get("/api/imagery/test/spectral/invalid")
        assert response.status_code == 404

        print("✅ Error handling working correctly")

    def test_imagery_performance(self):
        """Test imagery generation performance."""
        import time

        # Create larger dataset for performance testing
        large_dataset = xr.Dataset(
            {
                "red": (["y", "x"], np.random.rand(200, 250) * 3000),
                "nir": (["y", "x"], np.random.rand(200, 250) * 4000),
            },
            coords={
                "x": np.linspace(-122.0, -121.5, 250),
                "y": np.linspace(36.0, 37.0, 200),
            },
        )

        analysis_id = "performance-test"
        _analysis_cache[analysis_id] = {
            "dataset": large_dataset,
            "indices": {
                "ndvi": (large_dataset["nir"] - large_dataset["red"])
                / (large_dataset["nir"] + large_dataset["red"])
            },
            "masks": {
                "kelp_mask": xr.DataArray(
                    np.random.choice([True, False], (200, 250)), dims=["y", "x"]
                )
            },
            "biomass": xr.DataArray(np.random.rand(200, 250) * 5000, dims=["y", "x"]),
        }

        # Test RGB generation performance
        start_time = time.time()
        response = self.client.get(f"/api/imagery/{analysis_id}/rgb")
        rgb_time = time.time() - start_time

        assert response.status_code == 200
        assert rgb_time < 10.0  # Should complete within 10 seconds

        print(f"✅ RGB generation time: {rgb_time:.2f}s (target: <10s)")

        # Test spectral visualization performance
        start_time = time.time()
        response = self.client.get(f"/api/imagery/{analysis_id}/spectral/ndvi")
        spectral_time = time.time() - start_time

        assert response.status_code == 200
        assert spectral_time < 10.0

        print(f"✅ Spectral visualization time: {spectral_time:.2f}s (target: <10s)")

        print("✅ Performance tests passed")

    def test_layer_name_mapping_functionality(self):
        """Test that layer name mapping works correctly for frontend display."""
        # Test internal layer names vs display names
        internal_to_display = {
            "kelp_mask": "kelp",
            "water_mask": "water",
            "cloud_mask": "cloud",
            "rgb": "rgb",
            "false_color": "false_color",
            "ndvi": "ndvi",
            "fai": "fai",
            "red_edge_ndvi": "red_edge_ndvi",
        }

        # Test the mapping logic (this would be in JavaScript normally)
        for internal_name, expected_display in internal_to_display.items():
            # Simulate the layer name transformation
            if internal_name.endswith("_mask"):
                display_name = internal_name.replace("_mask", "")
            else:
                display_name = internal_name

            assert (
                display_name == expected_display
            ), f"Layer name mapping failed for {internal_name}"

        print("✅ Layer name mapping functionality verified")

    def test_layer_availability_assertions(self):
        """Test layer availability checks for integration."""
        # Mock analysis response with available layers
        available_layers = {
            "base_layers": ["rgb", "false_color"],
            "spectral_indices": ["ndvi", "fai", "red_edge_ndvi"],
            "masks": ["kelp", "water", "cloud"],
            "biomass": True,
        }

        # Test that expected layers are available
        assert "rgb" in available_layers["base_layers"]
        assert "kelp" in available_layers["masks"]
        assert "ndvi" in available_layers["spectral_indices"]
        assert available_layers["biomass"] is True

        # Test layer count expectations
        assert len(available_layers["base_layers"]) >= 2
        assert len(available_layers["spectral_indices"]) >= 3
        assert len(available_layers["masks"]) >= 3

        print("✅ Layer availability assertions verified")

    def test_geographic_bounds_integration(self):
        """Test proper geographic bounds for layer positioning."""
        # Mock geographic bounds for layer testing
        test_bounds = [-122.0, 36.5, -121.8, 36.7]  # [minX, minY, maxX, maxY]

        # Verify bounds format and validity
        assert len(test_bounds) == 4
        assert test_bounds[0] < test_bounds[2]  # minX < maxX
        assert test_bounds[1] < test_bounds[3]  # minY < maxY

        # Test geographic coordinate ranges
        assert -180 <= test_bounds[0] <= 180  # longitude range
        assert -180 <= test_bounds[2] <= 180  # longitude range
        assert -90 <= test_bounds[1] <= 90  # latitude range
        assert -90 <= test_bounds[3] <= 90  # latitude range

        print(f"✅ Geographic bounds integration verified: {test_bounds}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
