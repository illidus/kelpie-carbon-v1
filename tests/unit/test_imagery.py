"""Tests for satellite imagery visualization functionality."""

import io

import numpy as np
import pytest
import xarray as xr
from PIL import Image

from kelpie_carbon_v1.imagery import (
    generate_biomass_heatmap,
    generate_cloud_mask_overlay,
    generate_false_color_composite,
    generate_kelp_mask_overlay,
    generate_rgb_composite,
    generate_spectral_visualization,
    generate_water_mask_overlay,
)
from kelpie_carbon_v1.imagery.utils import (
    apply_colormap,
    array_to_image,
    calculate_histogram_stretch,
    create_rgba_overlay,
    enhance_contrast,
    get_image_bounds,
    normalize_band,
    normalize_to_0_1,
)


class TestImageryUtils:
    """Test utility functions for imagery processing."""

    def test_normalize_band(self):
        """Test band normalization."""
        # Create test data
        data = xr.DataArray(np.random.rand(50, 50) * 1000, dims=["y", "x"])

        normalized = normalize_band(data)

        assert normalized.dtype == np.uint8
        assert normalized.min() >= 0
        assert normalized.max() <= 255
        assert normalized.shape == data.shape

    def test_normalize_band_with_nan(self):
        """Test band normalization with NaN values."""
        data_array = np.random.rand(20, 20) * 500
        data_array[0:5, 0:5] = np.nan

        data = xr.DataArray(data_array, dims=["y", "x"])
        normalized = normalize_band(data)

        assert normalized.dtype == np.uint8
        assert not np.any(np.isnan(normalized))

    def test_normalize_to_0_1(self):
        """Test data normalization to 0-1 range."""
        data = xr.DataArray(
            np.random.rand(30, 30) * 100 - 50, dims=["y", "x"]  # Range -50 to 50
        )

        normalized = normalize_to_0_1(data)

        assert normalized.min() >= 0
        assert normalized.max() <= 1
        assert normalized.shape == data.shape

    def test_apply_colormap(self):
        """Test colormap application."""
        data = np.random.rand(25, 25)

        colored = apply_colormap(data, "viridis")

        assert colored.shape == (*data.shape, 4)  # RGBA
        assert colored.dtype == np.uint8
        assert colored.min() >= 0
        assert colored.max() <= 255

    def test_array_to_image_rgb(self):
        """Test converting RGB array to PIL Image."""
        rgb_array = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)

        image = array_to_image(rgb_array)

        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert image.size == (40, 40)

    def test_array_to_image_rgba(self):
        """Test converting RGBA array to PIL Image."""
        rgba_array = np.random.randint(0, 256, (40, 40, 4), dtype=np.uint8)

        image = array_to_image(rgba_array)

        assert isinstance(image, Image.Image)
        assert image.mode == "RGBA"

    def test_array_to_image_grayscale(self):
        """Test converting grayscale array to PIL Image."""
        gray_array = np.random.randint(0, 256, (40, 40), dtype=np.uint8)

        image = array_to_image(gray_array)

        assert isinstance(image, Image.Image)
        assert image.mode == "L"

    def test_get_image_bounds(self):
        """Test extracting geographical bounds."""
        # Create dataset with coordinate information
        dataset = xr.Dataset(
            {"test_var": (["y", "x"], np.random.rand(20, 30))},
            coords={"x": np.linspace(-122, -121, 30), "y": np.linspace(36, 37, 20)},
        )

        bounds = get_image_bounds(dataset)

        assert len(bounds) == 4
        min_lon, min_lat, max_lon, max_lat = bounds
        assert min_lon < max_lon
        assert min_lat < max_lat

    def test_create_rgba_overlay(self):
        """Test creating RGBA overlay from mask."""
        mask = np.random.choice([True, False], (30, 30))
        color = (255, 0, 0)  # Red
        alpha = 0.7

        rgba = create_rgba_overlay(mask, color, alpha)

        assert rgba.shape == (*mask.shape, 4)
        assert rgba.dtype == np.uint8

        # Check that masked areas have the correct color
        masked_pixels = rgba[mask]
        if len(masked_pixels) > 0:
            assert np.all(masked_pixels[:, 0] == 255)  # Red
            assert np.all(masked_pixels[:, 1] == 0)  # Green
            assert np.all(masked_pixels[:, 2] == 0)  # Blue
            assert np.all(masked_pixels[:, 3] == int(alpha * 255))  # Alpha

    def test_enhance_contrast(self):
        """Test contrast enhancement."""
        array = np.random.randint(0, 256, (25, 25, 3), dtype=np.uint8)

        enhanced = enhance_contrast(array, gamma=0.8)

        assert enhanced.shape == array.shape
        assert enhanced.dtype == np.uint8

    def test_calculate_histogram_stretch(self):
        """Test histogram stretch calculation."""
        data = np.random.normal(0, 1, (100, 100))

        min_val, max_val = calculate_histogram_stretch(data, percent=5.0)

        assert min_val < max_val
        assert isinstance(min_val, float)
        assert isinstance(max_val, float)


class TestImageGenerators:
    """Test image generation functions."""

    def setup_method(self):
        """Set up test data."""
        # Create a mock satellite dataset
        self.dataset = xr.Dataset(
            {
                "red": (["y", "x"], np.random.rand(50, 60) * 3000),
                "nir": (["y", "x"], np.random.rand(50, 60) * 4000),
                "red_edge": (["y", "x"], np.random.rand(50, 60) * 3500),
            },
            coords={"x": np.linspace(-122, -121, 60), "y": np.linspace(36, 37, 50)},
        )

    def test_generate_rgb_composite_red_only(self):
        """Test RGB composite with only red band."""
        image = generate_rgb_composite(self.dataset, red_band="red")

        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert image.size == (60, 50)

    def test_generate_rgb_composite_missing_band(self):
        """Test RGB composite with missing band."""
        with pytest.raises(ValueError, match="Red band 'missing' not found"):
            generate_rgb_composite(self.dataset, red_band="missing")

    def test_generate_false_color_composite(self):
        """Test false-color composite generation."""
        image = generate_false_color_composite(
            self.dataset, nir_band="nir", red_band="red", green_band="red_edge"
        )

        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert image.size == (60, 50)

    def test_generate_false_color_missing_bands(self):
        """Test false-color composite with missing bands."""
        with pytest.raises(ValueError, match="Missing bands"):
            generate_false_color_composite(
                self.dataset, nir_band="missing", red_band="red", green_band="red_edge"
            )

    def test_generate_spectral_visualization(self):
        """Test spectral index visualization."""
        # Create a spectral index
        ndvi = (self.dataset["nir"] - self.dataset["red"]) / (
            self.dataset["nir"] + self.dataset["red"]
        )

        image = generate_spectral_visualization(ndvi, colormap="RdYlGn")

        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert image.size == (60, 50)


class TestOverlayGenerators:
    """Test overlay generation functions."""

    def setup_method(self):
        """Set up test data."""
        # Create mock dataset with masks
        self.dataset = xr.Dataset(
            {
                "kelp_mask": (
                    ["y", "x"],
                    np.random.choice([True, False], (40, 50), p=[0.3, 0.7]),
                ),
                "water_mask": (
                    ["y", "x"],
                    np.random.choice([True, False], (40, 50), p=[0.6, 0.4]),
                ),
                "cloud_mask": (
                    ["y", "x"],
                    np.random.choice([True, False], (40, 50), p=[0.1, 0.9]),
                ),
            }
        )

    def test_generate_kelp_mask_overlay(self):
        """Test kelp mask overlay generation."""
        image = generate_kelp_mask_overlay(self.dataset, alpha=0.6)

        assert isinstance(image, Image.Image)
        assert image.mode == "RGBA"
        assert image.size == (50, 40)

    def test_generate_water_mask_overlay(self):
        """Test water mask overlay generation."""
        image = generate_water_mask_overlay(self.dataset, alpha=0.4)

        assert isinstance(image, Image.Image)
        assert image.mode == "RGBA"

    def test_generate_cloud_mask_overlay(self):
        """Test cloud mask overlay generation."""
        image = generate_cloud_mask_overlay(self.dataset, alpha=0.5)

        assert isinstance(image, Image.Image)
        assert image.mode == "RGBA"

    def test_generate_mask_overlay_missing_data(self):
        """Test mask overlay with missing data."""
        empty_dataset = xr.Dataset()

        with pytest.raises(ValueError, match="does not contain kelp_mask"):
            generate_kelp_mask_overlay(empty_dataset)

    def test_generate_biomass_heatmap(self):
        """Test biomass heatmap generation."""
        # Create biomass data
        biomass_data = xr.DataArray(
            np.random.rand(35, 45) * 5000, dims=["y", "x"]  # Biomass in kg/ha
        )

        image = generate_biomass_heatmap(biomass_data, colormap="hot")

        assert isinstance(image, Image.Image)
        assert image.mode == "RGBA"
        assert image.size == (45, 35)

    def test_generate_biomass_heatmap_no_data(self):
        """Test biomass heatmap with no valid data."""
        # Create biomass data with all zeros/NaN
        biomass_data = xr.DataArray(np.zeros((20, 25)), dims=["y", "x"])

        image = generate_biomass_heatmap(biomass_data)

        assert isinstance(image, Image.Image)
        assert image.mode == "RGBA"
        # Should return transparent image
        array = np.array(image)
        assert np.all(array[:, :, 3] == 0)  # All alpha values should be 0

    def test_generate_biomass_heatmap_custom_scaling(self):
        """Test biomass heatmap with custom scaling."""
        biomass_data = xr.DataArray(np.random.rand(30, 30) * 3000, dims=["y", "x"])

        image = generate_biomass_heatmap(biomass_data, min_biomass=0, max_biomass=5000)

        assert isinstance(image, Image.Image)
        assert image.mode == "RGBA"


class TestImageryIntegration:
    """Integration tests for imagery functionality."""

    def test_full_imagery_pipeline(self):
        """Test complete imagery generation pipeline."""
        # Create comprehensive test dataset
        dataset = xr.Dataset(
            {
                "red": (["y", "x"], np.random.rand(30, 40) * 2000),
                "nir": (["y", "x"], np.random.rand(30, 40) * 3000),
                "red_edge": (["y", "x"], np.random.rand(30, 40) * 2500),
            }
        )

        # Calculate spectral indices
        ndvi = (dataset["nir"] - dataset["red"]) / (dataset["nir"] + dataset["red"])
        fai = dataset["nir"] - (
            dataset["red"]
            + (dataset["red_edge"] - dataset["red"]) * ((865 - 665) / (705 - 665))
        )

        # Create masks
        kelp_mask = fai > 0.1
        water_mask = ndvi < 0.1

        masks_dataset = xr.Dataset(
            {
                "kelp_mask": kelp_mask,
                "water_mask": water_mask,
            }
        )

        # Test RGB generation
        rgb_image = generate_rgb_composite(dataset)
        assert isinstance(rgb_image, Image.Image)

        # Test spectral visualization
        ndvi_image = generate_spectral_visualization(ndvi, colormap="RdYlGn")
        assert isinstance(ndvi_image, Image.Image)

        # Test mask overlays
        kelp_overlay = generate_kelp_mask_overlay(masks_dataset)
        water_overlay = generate_water_mask_overlay(masks_dataset)

        assert isinstance(kelp_overlay, Image.Image)
        assert isinstance(water_overlay, Image.Image)

    def test_imagery_with_real_satellite_data_structure(self):
        """Test imagery generation with realistic satellite data structure."""
        # Create dataset with Sentinel-2-like structure
        dataset = xr.Dataset(
            {
                "red": (
                    ["y", "x"],
                    np.random.rand(100, 120) * 0.3,
                ),  # Reflectance values
                "nir": (["y", "x"], np.random.rand(100, 120) * 0.4),
                "red_edge": (["y", "x"], np.random.rand(100, 120) * 0.35),
                "swir1": (["y", "x"], np.random.rand(100, 120) * 0.2),
            },
            coords={
                "x": np.linspace(-122.0, -121.8, 120),
                "y": np.linspace(36.5, 36.7, 100),
            },
        )

        # Test that bounds extraction works
        bounds = get_image_bounds(dataset)
        assert len(bounds) == 4

        # Test false-color composite
        false_color = generate_false_color_composite(dataset)
        assert false_color.size == (120, 100)

    def test_memory_efficiency(self):
        """Test memory efficiency with larger datasets."""
        # Create a larger dataset to test memory usage
        large_dataset = xr.Dataset(
            {
                "red": (["y", "x"], np.random.rand(200, 250) * 3000),
                "nir": (["y", "x"], np.random.rand(200, 250) * 4000),
            }
        )

        # Test that large images can be generated without errors
        image = generate_rgb_composite(large_dataset)
        assert isinstance(image, Image.Image)
        assert image.size == (250, 200)

    def test_error_handling(self):
        """Test error handling in imagery generation."""
        # Empty dataset
        empty_dataset = xr.Dataset()

        with pytest.raises(ValueError):
            generate_rgb_composite(empty_dataset)

        # Dataset with wrong structure
        wrong_dataset = xr.Dataset({"wrong_band": (["y", "x"], np.random.rand(10, 10))})

        with pytest.raises(ValueError):
            generate_false_color_composite(wrong_dataset)

    def test_image_quality_metrics(self):
        """Test that generated images meet quality standards."""
        dataset = xr.Dataset(
            {
                "red": (["y", "x"], np.random.rand(50, 60) * 2000),
                "nir": (["y", "x"], np.random.rand(50, 60) * 3000),
            }
        )

        image = generate_rgb_composite(dataset)

        # Convert to array for analysis
        img_array = np.array(image)

        # Check that image has variation (not all the same color)
        assert img_array.std() > 0

        # Check that image uses full dynamic range
        assert img_array.min() < 100  # Some dark areas
        assert img_array.max() > 150  # Some bright areas


if __name__ == "__main__":
    pytest.main([__file__])
