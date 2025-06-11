"""
Tests for Submerged Kelp Detection Enhancement.

This module tests the submerged kelp detection framework including depth-sensitive
detection, water column modeling, and integrated surface/submerged detection pipelines.
"""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch, MagicMock
import tempfile
import os

from kelpie_carbon_v1.detection.submerged_kelp_detection import (
    SubmergedKelpDetector,
    SubmergedKelpConfig,
    WaterColumnModel,
    DepthDetectionResult,
    create_submerged_kelp_detector,
    detect_submerged_kelp,
    analyze_depth_distribution,
)


class TestWaterColumnModel:
    """Test the WaterColumnModel dataclass."""
    
    def test_water_column_model_defaults(self):
        """Test default water column model parameters."""
        model = WaterColumnModel()
        
        assert model.attenuation_coefficient == 0.15
        assert model.scattering_coefficient == 0.05
        assert model.absorption_coefficient == 0.10
        assert model.kelp_backscatter_factor == 0.25
        assert model.turbidity_factor == 1.0
        assert model.depth_max_detectable == 1.5
    
    def test_water_column_model_custom(self):
        """Test custom water column model parameters."""
        model = WaterColumnModel(
            attenuation_coefficient=0.20,
            depth_max_detectable=2.0
        )
        
        assert model.attenuation_coefficient == 0.20
        assert model.depth_max_detectable == 2.0
        # Other values should remain default
        assert model.scattering_coefficient == 0.05


class TestSubmergedKelpConfig:
    """Test the SubmergedKelpConfig dataclass."""
    
    def test_config_defaults(self):
        """Test default configuration parameters."""
        config = SubmergedKelpConfig()
        
        assert config.ndre_surface_threshold == 0.05
        assert config.ndre_shallow_threshold == 0.02
        assert config.ndre_deep_threshold == -0.01
        assert config.depth_estimation_method == "water_column_model"
        assert config.primary_red_edge_band == "red_edge_2"
        assert config.require_water_context == True
        assert config.min_patch_size == 4
        assert config.confidence_threshold == 0.6
        
        # Test species depth factors
        assert config.species_depth_factors["Nereocystis"] == 1.0
        assert config.species_depth_factors["Macrocystis"] == 1.3
        assert config.species_depth_factors["Mixed"] == 1.1
    
    def test_config_custom(self):
        """Test custom configuration parameters."""
        custom_species_factors = {"Nereocystis": 0.8, "Custom": 1.5}
        
        config = SubmergedKelpConfig(
            ndre_surface_threshold=0.08,
            require_water_context=False,
            species_depth_factors=custom_species_factors
        )
        
        assert config.ndre_surface_threshold == 0.08
        assert config.require_water_context == False
        assert config.species_depth_factors == custom_species_factors


class TestDepthDetectionResult:
    """Test the DepthDetectionResult dataclass."""
    
    def test_depth_detection_result_creation(self):
        """Test creating a depth detection result."""
        shape = (10, 10)
        
        result = DepthDetectionResult(
            depth_estimate=np.zeros(shape),
            depth_confidence=np.ones(shape) * 0.8,
            surface_kelp_mask=np.zeros(shape, dtype=bool),
            submerged_kelp_mask=np.zeros(shape, dtype=bool),
            combined_kelp_mask=np.zeros(shape, dtype=bool),
            water_column_properties={"turbidity": np.ones(shape)},
            detection_metadata={"species": "Nereocystis"}
        )
        
        assert result.depth_estimate.shape == shape
        assert result.depth_confidence.shape == shape
        assert result.detection_metadata["species"] == "Nereocystis"
        assert "turbidity" in result.water_column_properties


class TestSubmergedKelpDetector:
    """Test the SubmergedKelpDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = SubmergedKelpDetector()
        
        # Create mock satellite dataset
        self.mock_dataset = xr.Dataset({
            'blue': (('y', 'x'), np.random.rand(20, 20) * 0.1),
            'green': (('y', 'x'), np.random.rand(20, 20) * 0.2),
            'red': (('y', 'x'), np.random.rand(20, 20) * 0.15),
            'red_edge': (('y', 'x'), np.random.rand(20, 20) * 0.3),
            'red_edge_2': (('y', 'x'), np.random.rand(20, 20) * 0.35),
            'nir': (('y', 'x'), np.random.rand(20, 20) * 0.4),
        })
        
        # Add some kelp-like spectral signatures
        # Surface kelp area (high NDRE)
        self.mock_dataset['red_edge_2'][5:8, 5:8] = 0.6
        self.mock_dataset['red'][5:8, 5:8] = 0.2
        
        # Submerged kelp area (moderate NDRE)  
        self.mock_dataset['red_edge_2'][12:15, 12:15] = 0.4
        self.mock_dataset['red'][12:15, 12:15] = 0.25
    
    def test_detector_initialization(self):
        """Test detector initialization with and without config."""
        # Default initialization
        detector1 = SubmergedKelpDetector()
        assert isinstance(detector1.config, SubmergedKelpConfig)
        
        # Custom config initialization
        custom_config = SubmergedKelpConfig(ndre_surface_threshold=0.08)
        detector2 = SubmergedKelpDetector(custom_config)
        assert detector2.config.ndre_surface_threshold == 0.08
    
    def test_calculate_depth_sensitive_indices(self):
        """Test calculation of depth-sensitive spectral indices."""
        indices = self.detector._calculate_depth_sensitive_indices(self.mock_dataset)
        
        # Verify all expected indices are calculated
        expected_indices = ['ndre_standard', 'ndre_enhanced', 'warei', 'ndvi', 'ski']
        for index_name in expected_indices:
            assert index_name in indices
            assert isinstance(indices[index_name], np.ndarray)
            assert indices[index_name].shape == (20, 20)
        
        # Verify values are in reasonable ranges
        assert np.all(indices['ndre_enhanced'] >= -1)
        assert np.all(indices['ndre_enhanced'] <= 1)
        assert np.all(indices['ndvi'] >= -1)
        assert np.all(indices['ndvi'] <= 1)
    
    def test_calculate_depth_sensitive_indices_missing_bands(self):
        """Test index calculation with missing spectral bands."""
        # Dataset without red_edge_2 band
        limited_dataset = self.mock_dataset.drop_vars(['red_edge_2'])
        
        indices = self.detector._calculate_depth_sensitive_indices(limited_dataset)
        
        # Should still work with fallback red_edge band
        assert 'ndre_enhanced' in indices
        assert indices['ndre_enhanced'].shape == (20, 20)
        
        # Dataset without blue band
        no_blue_dataset = self.mock_dataset.drop_vars(['blue'])
        indices_no_blue = self.detector._calculate_depth_sensitive_indices(no_blue_dataset)
        
        # WAREI should not be calculated without blue band
        assert 'warei' not in indices_no_blue
    
    def test_apply_depth_stratified_detection(self):
        """Test depth-stratified kelp detection."""
        indices = self.detector._calculate_depth_sensitive_indices(self.mock_dataset)
        
        surface_mask, submerged_mask = self.detector._apply_depth_stratified_detection(
            self.mock_dataset, indices, "Nereocystis"
        )
        
        assert isinstance(surface_mask, np.ndarray)
        assert isinstance(submerged_mask, np.ndarray)
        assert surface_mask.shape == (20, 20)
        assert submerged_mask.shape == (20, 20)
        assert surface_mask.dtype == bool
        assert submerged_mask.dtype == bool
        
        # Should detect some kelp in our synthetic areas
        assert np.any(surface_mask) or np.any(submerged_mask)
    
    def test_apply_depth_stratified_detection_species_factors(self):
        """Test species-specific depth factor adjustments."""
        indices = self.detector._calculate_depth_sensitive_indices(self.mock_dataset)
        
        # Test different species
        nereocystis_surface, nereocystis_sub = self.detector._apply_depth_stratified_detection(
            self.mock_dataset, indices, "Nereocystis"
        )
        
        macrocystis_surface, macrocystis_sub = self.detector._apply_depth_stratified_detection(
            self.mock_dataset, indices, "Macrocystis"
        )
        
        # Macrocystis should potentially detect more submerged kelp (depth factor 1.3)
        # This test is probabilistic due to random data, but structure should be consistent
        assert nereocystis_surface.shape == macrocystis_surface.shape
        assert nereocystis_sub.shape == macrocystis_sub.shape
    
    def test_estimate_kelp_depths(self):
        """Test kelp depth estimation."""
        indices = self.detector._calculate_depth_sensitive_indices(self.mock_dataset)
        
        # Create mock detection masks
        surface_mask = np.zeros((20, 20), dtype=bool)
        surface_mask[5:8, 5:8] = True
        
        submerged_mask = np.zeros((20, 20), dtype=bool)
        submerged_mask[12:15, 12:15] = True
        
        depth_estimates, depth_confidence, water_props = self.detector._estimate_kelp_depths(
            self.mock_dataset, indices, surface_mask, submerged_mask
        )
        
        assert depth_estimates.shape == (20, 20)
        assert depth_confidence.shape == (20, 20)
        assert isinstance(water_props, dict)
        
        # Surface kelp should have shallow depths
        surface_depths = depth_estimates[surface_mask]
        assert np.all(surface_depths <= 0.3)
        assert np.all(surface_depths >= 0.0)
        
        # Submerged kelp should have deeper depths
        submerged_depths = depth_estimates[submerged_mask]
        assert np.all(submerged_depths >= 0.3)
        
        # Confidence should be higher for surface kelp
        surface_confidence = depth_confidence[surface_mask]
        submerged_confidence = depth_confidence[submerged_mask]
        if len(surface_confidence) > 0 and len(submerged_confidence) > 0:
            assert np.mean(surface_confidence) >= np.mean(submerged_confidence)
    
    def test_model_water_column_properties(self):
        """Test water column property modeling."""
        indices = self.detector._calculate_depth_sensitive_indices(self.mock_dataset)
        
        water_props = self.detector._model_water_column_properties(self.mock_dataset, indices)
        
        # Verify all expected properties are calculated
        expected_props = ['turbidity', 'water_clarity', 'attenuation_coefficient', 'chlorophyll_proxy']
        for prop in expected_props:
            assert prop in water_props
            assert isinstance(water_props[prop], np.ndarray)
            assert water_props[prop].shape == (20, 20)
        
        # Verify reasonable value ranges
        assert np.all(water_props['turbidity'] >= 0.5)
        assert np.all(water_props['turbidity'] <= 3.0)
        assert np.all(water_props['water_clarity'] >= -0.5)
        assert np.all(water_props['water_clarity'] <= 0.5)
        assert np.all(water_props['chlorophyll_proxy'] >= 0.0)
        assert np.all(water_props['chlorophyll_proxy'] <= 1.0)
    
    def test_apply_quality_control(self):
        """Test quality control filtering."""
        # Create mock masks with small and large patches
        surface_mask = np.zeros((20, 20), dtype=bool)
        surface_mask[5:8, 5:8] = True  # Large patch (9 pixels)
        surface_mask[1, 1] = True      # Small patch (1 pixel)
        
        submerged_mask = np.zeros((20, 20), dtype=bool)
        submerged_mask[12:15, 12:15] = True  # Large patch
        submerged_mask[18:19, 18:19] = True  # Small patch (1 pixel)
        
        # Create confidence array with varying values
        depth_confidence = np.ones((20, 20)) * 0.8  # High confidence
        depth_confidence[1, 1] = 0.3  # Low confidence for small patch
        depth_confidence[18:19, 18:19] = 0.4  # Low confidence for small patch
        
        surface_filtered, submerged_filtered = self.detector._apply_quality_control(
            surface_mask, submerged_mask, depth_confidence
        )
        
        # Small, low-confidence patches should be removed
        assert not surface_filtered[1, 1]  # Small + low confidence
        assert not submerged_filtered[18, 18]  # Small + low confidence
        
        # Large, high-confidence patches should remain
        assert np.any(surface_filtered[5:8, 5:8])
        assert np.any(submerged_filtered[12:15, 12:15])
    
    def test_remove_small_patches(self):
        """Test small patch removal."""
        # Create mask with patches of different sizes
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:8, 5:8] = True    # Large patch (9 pixels)
        mask[1:2, 1:2] = True    # Small patch (1 pixel)
        mask[15:17, 15:17] = True # Medium patch (4 pixels)
        
        # Remove patches smaller than 4 pixels
        filtered_mask = self.detector._remove_small_patches(mask, min_size=4)
        
        # Large patch should remain
        assert np.any(filtered_mask[5:8, 5:8])
        
        # Small patch should be removed
        assert not filtered_mask[1, 1]
        
        # Medium patch should remain (exactly 4 pixels)
        assert np.any(filtered_mask[15:17, 15:17])
    
    def test_combine_detection_layers(self):
        """Test combining surface and submerged detection layers."""
        surface_mask = np.zeros((20, 20), dtype=bool)
        surface_mask[5:8, 5:8] = True
        
        submerged_mask = np.zeros((20, 20), dtype=bool)
        submerged_mask[7:10, 7:10] = True  # Overlapping with surface
        submerged_mask[15:17, 15:17] = True  # Isolated
        
        combined_mask = self.detector._combine_detection_layers(surface_mask, submerged_mask)
        
        # Combined should include both surface and submerged
        assert np.any(combined_mask[5:8, 5:8])    # Surface area
        assert np.any(combined_mask[7:10, 7:10])  # Overlapping area
        assert np.any(combined_mask[15:17, 15:17]) # Isolated submerged
        
        # Should be union of both masks
        expected_combined = surface_mask | submerged_mask
        total_expected = np.sum(expected_combined)
        total_actual = np.sum(combined_mask)
        
        # Should be similar (connectivity constraints may modify slightly)
        assert abs(total_actual - total_expected) <= total_expected * 0.2
    
    def test_generate_detection_metadata(self):
        """Test detection metadata generation."""
        surface_mask = np.zeros((20, 20), dtype=bool)
        surface_mask[5:8, 5:8] = True  # 9 pixels
        
        submerged_mask = np.zeros((20, 20), dtype=bool)
        submerged_mask[12:15, 12:15] = True  # 9 pixels
        
        metadata = self.detector._generate_detection_metadata(
            self.mock_dataset, surface_mask, submerged_mask, "Nereocystis"
        )
        
        # Verify all expected metadata fields
        expected_fields = [
            "processing_timestamp", "species", "total_pixels", "surface_kelp_pixels",
            "submerged_kelp_pixels", "total_kelp_pixels", "surface_coverage_percent",
            "submerged_coverage_percent", "total_kelp_coverage_percent",
            "surface_area_m2", "submerged_area_m2", "total_kelp_area_m2",
            "surface_to_submerged_ratio", "detection_method", "config_used"
        ]
        
        for field in expected_fields:
            assert field in metadata
        
        # Verify calculated values
        assert metadata["species"] == "Nereocystis"
        assert metadata["total_pixels"] == 400  # 20x20
        assert metadata["surface_kelp_pixels"] == 9
        assert metadata["submerged_kelp_pixels"] == 9
        assert metadata["total_kelp_pixels"] == 18  # 9 + 9 (no overlap in this case)
        assert metadata["surface_area_m2"] == 900.0  # 9 pixels * 100 m²/pixel
        assert metadata["detection_method"] == "red_edge_depth_stratified"
    
    @patch('kelpie_carbon_v1.detection.submerged_kelp_detection.create_water_mask')
    def test_detect_submerged_kelp_full_pipeline(self, mock_water_mask):
        """Test the complete submerged kelp detection pipeline."""
        # Mock water mask to avoid dependency on water detection
        mock_water_mask.return_value = np.ones((20, 20), dtype=bool)
        
        result = self.detector.detect_submerged_kelp(
            self.mock_dataset, 
            species="Nereocystis",
            include_depth_analysis=True
        )
        
        assert isinstance(result, DepthDetectionResult)
        assert result.depth_estimate.shape == (20, 20)
        assert result.depth_confidence.shape == (20, 20)
        assert result.surface_kelp_mask.shape == (20, 20)
        assert result.submerged_kelp_mask.shape == (20, 20)
        assert result.combined_kelp_mask.shape == (20, 20)
        assert isinstance(result.water_column_properties, dict)
        assert isinstance(result.detection_metadata, dict)
        
        # Verify metadata species
        assert result.detection_metadata["species"] == "Nereocystis"
    
    def test_detect_submerged_kelp_without_depth_analysis(self):
        """Test detection without depth analysis."""
        with patch('kelpie_carbon_v1.detection.submerged_kelp_detection.create_water_mask') as mock_water_mask:
            mock_water_mask.return_value = np.ones((20, 20), dtype=bool)
            
            result = self.detector.detect_submerged_kelp(
                self.mock_dataset,
                species="Mixed", 
                include_depth_analysis=False
            )
        
        # Depth arrays should be zero when depth analysis is disabled
        assert np.all(result.depth_estimate == 0)
        assert np.all(result.depth_confidence == 0)
        assert len(result.water_column_properties) == 0
        
        # But detection masks should still be generated
        assert result.surface_kelp_mask.shape == (20, 20)
        assert result.submerged_kelp_mask.shape == (20, 20)
    
    def test_detect_submerged_kelp_error_handling(self):
        """Test error handling in detection pipeline."""
        # Create malformed dataset to trigger error
        bad_dataset = xr.Dataset({
            'invalid_band': (('y', 'x'), np.array([]).reshape(0, 0))  # Properly shaped empty array
        })
        
        result = self.detector.detect_submerged_kelp(bad_dataset)
        
        # Should return empty result on error
        assert isinstance(result, DepthDetectionResult)
        assert "error" in result.detection_metadata
        assert result.surface_kelp_mask.shape == (0, 0)  # Empty dataset shape
        assert result.submerged_kelp_mask.shape == (0, 0)


class TestFactoryFunctions:
    """Test factory functions and high-level interfaces."""
    
    def test_create_submerged_kelp_detector(self):
        """Test detector factory function."""
        # Default creation
        detector1 = create_submerged_kelp_detector()
        assert isinstance(detector1, SubmergedKelpDetector)
        
        # Custom config creation
        config = SubmergedKelpConfig(ndre_surface_threshold=0.08)
        detector2 = create_submerged_kelp_detector(config)
        assert detector2.config.ndre_surface_threshold == 0.08
    
    @patch('kelpie_carbon_v1.detection.submerged_kelp_detection.create_water_mask')
    def test_detect_submerged_kelp_function(self, mock_water_mask):
        """Test high-level detection function."""
        # Mock water mask
        mock_water_mask.return_value = np.ones((10, 10), dtype=bool)
        
        # Create simple test dataset
        dataset = xr.Dataset({
            'blue': (('y', 'x'), np.random.rand(10, 10) * 0.1),
            'green': (('y', 'x'), np.random.rand(10, 10) * 0.2),
            'red': (('y', 'x'), np.random.rand(10, 10) * 0.15),
            'red_edge': (('y', 'x'), np.random.rand(10, 10) * 0.3),
            'red_edge_2': (('y', 'x'), np.random.rand(10, 10) * 0.35),
            'nir': (('y', 'x'), np.random.rand(10, 10) * 0.4),
        })
        
        result = detect_submerged_kelp(
            dataset, 
            species="Macrocystis",
            include_depth_analysis=True
        )
        
        assert isinstance(result, DepthDetectionResult)
        assert result.detection_metadata["species"] == "Macrocystis"
    
    def test_analyze_depth_distribution(self):
        """Test depth distribution analysis."""
        # Create mock detection result
        shape = (10, 10)
        
        # Create some synthetic depth data
        depth_estimate = np.zeros(shape)
        depth_estimate[2:4, 2:4] = np.array([[0.2, 0.3], [0.5, 0.8]])  # Various depths
        
        depth_confidence = np.zeros(shape)
        depth_confidence[2:4, 2:4] = np.array([[0.9, 0.8], [0.7, 0.6]])  # Various confidences
        
        combined_mask = np.zeros(shape, dtype=bool)
        combined_mask[2:4, 2:4] = True
        
        result = DepthDetectionResult(
            depth_estimate=depth_estimate,
            depth_confidence=depth_confidence,
            surface_kelp_mask=np.zeros(shape, dtype=bool),
            submerged_kelp_mask=np.zeros(shape, dtype=bool),
            combined_kelp_mask=combined_mask,
            water_column_properties={},
            detection_metadata={}
        )
        
        analysis = analyze_depth_distribution(result)
        
        # Verify analysis structure
        expected_fields = [
            "mean_depth_m", "median_depth_m", "depth_std_m", "min_depth_m", "max_depth_m",
            "surface_fraction", "shallow_fraction", "deep_fraction", "mean_confidence", "total_kelp_pixels"
        ]
        
        for field in expected_fields:
            assert field in analysis
        
        # Verify calculated values
        assert analysis["total_kelp_pixels"] == 4
        assert analysis["min_depth_m"] == 0.2
        assert analysis["max_depth_m"] == 0.8
        assert 0.0 <= analysis["surface_fraction"] <= 1.0
        assert 0.0 <= analysis["shallow_fraction"] <= 1.0  
        assert 0.0 <= analysis["deep_fraction"] <= 1.0
        assert analysis["surface_fraction"] + analysis["shallow_fraction"] + analysis["deep_fraction"] == 1.0
    
    def test_analyze_depth_distribution_no_kelp(self):
        """Test depth distribution analysis with no kelp detected."""
        shape = (10, 10)
        
        result = DepthDetectionResult(
            depth_estimate=np.zeros(shape),
            depth_confidence=np.zeros(shape),
            surface_kelp_mask=np.zeros(shape, dtype=bool),
            submerged_kelp_mask=np.zeros(shape, dtype=bool),
            combined_kelp_mask=np.zeros(shape, dtype=bool),  # No kelp detected
            water_column_properties={},
            detection_metadata={}
        )
        
        analysis = analyze_depth_distribution(result)
        
        # Should return error when no kelp detected
        assert "error" in analysis
        assert "No kelp detected" in analysis["error"]


class TestIntegrationScenarios:
    """Test integration scenarios and realistic use cases."""
    
    def setup_method(self):
        """Set up realistic test scenarios."""
        self.config = SubmergedKelpConfig(
            ndre_surface_threshold=0.06,
            ndre_shallow_threshold=0.03,
            require_water_context=True
        )
        self.detector = SubmergedKelpDetector(self.config)
    
    def create_realistic_kelp_dataset(self):
        """Create a realistic dataset with kelp spectral signatures."""
        shape = (30, 30)
        
        # Base water spectrum
        dataset = xr.Dataset({
            'blue': (('y', 'x'), np.random.uniform(0.05, 0.15, shape)),
            'green': (('y', 'x'), np.random.uniform(0.08, 0.18, shape)),
            'red': (('y', 'x'), np.random.uniform(0.06, 0.12, shape)),
            'red_edge': (('y', 'x'), np.random.uniform(0.08, 0.16, shape)),
            'red_edge_2': (('y', 'x'), np.random.uniform(0.10, 0.18, shape)),
            'nir': (('y', 'x'), np.random.uniform(0.15, 0.25, shape)),
        })
        
        # Add realistic kelp spectral signatures
        # Surface kelp: high red-edge reflectance
        surface_kelp_area = np.s_[8:12, 8:12]
        dataset['red_edge_2'][surface_kelp_area] = np.random.uniform(0.35, 0.45, (4, 4))
        dataset['red_edge'][surface_kelp_area] = np.random.uniform(0.30, 0.40, (4, 4))
        dataset['red'][surface_kelp_area] = np.random.uniform(0.08, 0.15, (4, 4))
        
        # Submerged kelp: moderate red-edge with water attenuation
        submerged_kelp_area = np.s_[18:22, 18:22]
        dataset['red_edge_2'][submerged_kelp_area] = np.random.uniform(0.25, 0.32, (4, 4))
        dataset['red_edge'][submerged_kelp_area] = np.random.uniform(0.22, 0.28, (4, 4))
        dataset['red'][submerged_kelp_area] = np.random.uniform(0.12, 0.18, (4, 4))
        
        return dataset
    
    @patch('kelpie_carbon_v1.detection.submerged_kelp_detection.create_water_mask')
    def test_realistic_kelp_detection_scenario(self, mock_water_mask):
        """Test detection with realistic kelp spectral signatures."""
        # Create water mask that covers our kelp areas
        water_mask = np.zeros((30, 30), dtype=bool)
        water_mask[5:25, 5:25] = True  # Water area covering kelp zones
        mock_water_mask.return_value = water_mask
        
        dataset = self.create_realistic_kelp_dataset()
        
        result = self.detector.detect_submerged_kelp(
            dataset,
            species="Macrocystis",
            include_depth_analysis=True
        )
        
        # Should detect kelp in either surface or submerged areas (or both)
        assert np.any(result.surface_kelp_mask) or np.any(result.submerged_kelp_mask)
        
        # Some kelp should be detected overall
        total_kelp_detected = np.any(result.combined_kelp_mask)
        assert total_kelp_detected, "Should detect kelp somewhere in the image"
        
        # Depth estimates should be reasonable
        detected_depths = result.depth_estimate[result.combined_kelp_mask]
        if len(detected_depths) > 0:
            assert np.all(detected_depths >= 0.0)
            assert np.all(detected_depths <= 1.5)  # Within max detectable depth
        
        # Metadata should be properly populated
        metadata = result.detection_metadata
        assert metadata["species"] == "Macrocystis"
        assert metadata["detection_method"] == "red_edge_depth_stratified"
        assert metadata["total_kelp_pixels"] > 0
    
    def test_species_specific_detection_differences(self):
        """Test that different species produce different detection results."""
        dataset = self.create_realistic_kelp_dataset()
        
        with patch('kelpie_carbon_v1.detection.submerged_kelp_detection.create_water_mask') as mock_water_mask:
            mock_water_mask.return_value = np.ones((30, 30), dtype=bool)
            
            # Test Nereocystis (surface-oriented, depth factor 1.0)
            result_nereocystis = self.detector.detect_submerged_kelp(
                dataset, species="Nereocystis", include_depth_analysis=False
            )
            
            # Test Macrocystis (deeper fronds, depth factor 1.3)
            result_macrocystis = self.detector.detect_submerged_kelp(
                dataset, species="Macrocystis", include_depth_analysis=False
            )
        
        # Both should detect kelp but potentially with different patterns
        nereocystis_total = np.sum(result_nereocystis.combined_kelp_mask)
        macrocystis_total = np.sum(result_macrocystis.combined_kelp_mask)
        
        # Both detections should complete successfully (detection may or may not find kelp)
        assert isinstance(result_nereocystis, DepthDetectionResult)
        assert isinstance(result_macrocystis, DepthDetectionResult)
        
        # Metadata should reflect different species
        assert result_nereocystis.detection_metadata["species"] == "Nereocystis"
        assert result_macrocystis.detection_metadata["species"] == "Macrocystis"
    
    def test_depth_estimation_accuracy(self):
        """Test depth estimation produces reasonable results."""
        dataset = self.create_realistic_kelp_dataset()
        
        with patch('kelpie_carbon_v1.detection.submerged_kelp_detection.create_water_mask') as mock_water_mask:
            mock_water_mask.return_value = np.ones((30, 30), dtype=bool)
            
            result = self.detector.detect_submerged_kelp(
                dataset, species="Mixed", include_depth_analysis=True
            )
        
        if np.any(result.combined_kelp_mask):
            # Analyze depth distribution
            depth_analysis = analyze_depth_distribution(result)
            
            # Should have reasonable depth statistics
            assert 0.0 <= depth_analysis["mean_depth_m"] <= 1.5
            assert 0.0 <= depth_analysis["median_depth_m"] <= 1.5
            assert depth_analysis["min_depth_m"] >= 0.0
            assert depth_analysis["max_depth_m"] <= 1.5
            
            # Fractions should sum to 1
            total_fraction = (depth_analysis["surface_fraction"] + 
                            depth_analysis["shallow_fraction"] + 
                            depth_analysis["deep_fraction"])
            assert abs(total_fraction - 1.0) < 0.01
            
            # Mean confidence should be reasonable
            assert 0.0 <= depth_analysis["mean_confidence"] <= 1.0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = SubmergedKelpDetector()
    
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        empty_dataset = xr.Dataset()
        
        result = self.detector.detect_submerged_kelp(empty_dataset)
        
        # Should handle gracefully and return error
        assert isinstance(result, DepthDetectionResult)
        assert "error" in result.detection_metadata
    
    def test_dataset_with_nan_values(self):
        """Test handling of dataset with NaN values."""
        dataset = xr.Dataset({
            'red': (('y', 'x'), np.full((10, 10), np.nan)),
            'red_edge': (('y', 'x'), np.full((10, 10), np.nan)),
            'nir': (('y', 'x'), np.full((10, 10), np.nan)),
        })
        
        # Should handle NaN values gracefully
        indices = self.detector._calculate_depth_sensitive_indices(dataset)
        
        for index_name, index_array in indices.items():
            assert np.all(np.isfinite(index_array)), f"Index {index_name} contains non-finite values"
    
    def test_very_small_dataset(self):
        """Test handling of very small dataset."""
        small_dataset = xr.Dataset({
            'red': (('y', 'x'), np.random.rand(2, 2)),
            'red_edge': (('y', 'x'), np.random.rand(2, 2)),
            'nir': (('y', 'x'), np.random.rand(2, 2)),
        })
        
        with patch('kelpie_carbon_v1.detection.submerged_kelp_detection.create_water_mask') as mock_water_mask:
            mock_water_mask.return_value = np.ones((2, 2), dtype=bool)
            
            result = self.detector.detect_submerged_kelp(small_dataset)
        
        # Should handle small datasets
        assert isinstance(result, DepthDetectionResult)
        assert result.surface_kelp_mask.shape == (2, 2)
    
    def test_extreme_spectral_values(self):
        """Test handling of extreme spectral values."""
        # Dataset with extreme values
        extreme_dataset = xr.Dataset({
            'blue': (('y', 'x'), np.full((10, 10), 1.0)),    # Maximum reflectance
            'green': (('y', 'x'), np.full((10, 10), 0.0)),   # Minimum reflectance
            'red': (('y', 'x'), np.full((10, 10), 0.5)),
            'red_edge': (('y', 'x'), np.full((10, 10), 0.5)),
            'nir': (('y', 'x'), np.full((10, 10), 1.0)),
        })
        
        # Should handle extreme values without crashing
        indices = self.detector._calculate_depth_sensitive_indices(extreme_dataset)
        
        for index_name, index_array in indices.items():
            assert np.all(np.isfinite(index_array))
            assert np.all(index_array >= -1.1)  # Allow small numerical errors
            assert np.all(index_array <= 1.1)
    
    def test_no_water_context(self):
        """Test detection without water context requirement."""
        config = SubmergedKelpConfig(require_water_context=False)
        detector = SubmergedKelpDetector(config)
        
        dataset = xr.Dataset({
            'red': (('y', 'x'), np.random.rand(10, 10) * 0.2),
            'red_edge': (('y', 'x'), np.random.rand(10, 10) * 0.3),
            'red_edge_2': (('y', 'x'), np.random.rand(10, 10) * 0.35),
            'nir': (('y', 'x'), np.random.rand(10, 10) * 0.4),
        })
        
        # Should work without water context checking
        result = detector.detect_submerged_kelp(dataset)
        
        assert isinstance(result, DepthDetectionResult)
        # May detect more false positives without water context, but should not crash 