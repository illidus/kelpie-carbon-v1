"""
Tests for Environmental Robustness Testing Framework.

This module tests the environmental condition validation for SKEMA kelp detection,
including tidal effects, water clarity, and seasonal variations.
"""

import pytest
import numpy as np
import xarray as xr
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from kelpie_carbon_v1.validation.environmental_testing import (
    EnvironmentalRobustnessValidator,
    EnvironmentalCondition,
    EnvironmentalTestResult,
)


class TestEnvironmentalRobustnessValidator:
    """Test the EnvironmentalRobustnessValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = EnvironmentalRobustnessValidator()
        
        # Create mock satellite dataset
        self.mock_dataset = xr.Dataset({
            'red': (('y', 'x'), np.random.rand(50, 50) * 0.3),
            'green': (('y', 'x'), np.random.rand(50, 50) * 0.4),
            'blue': (('y', 'x'), np.random.rand(50, 50) * 0.2),
            'nir': (('y', 'x'), np.random.rand(50, 50) * 0.6),
            'red_edge': (('y', 'x'), np.random.rand(50, 50) * 0.5),
        })
        
        # Create mock detection mask
        self.mock_detection_mask = np.random.rand(50, 50) * 0.3
    
    def test_environmental_conditions_definition(self):
        """Test that environmental conditions are properly defined."""
        conditions = self.validator.get_environmental_conditions()
        
        assert len(conditions) >= 4
        assert all(isinstance(c, EnvironmentalCondition) for c in conditions)
        
        # Check for key condition types
        condition_names = [c.name for c in conditions]
        assert any("tidal" in name or "tide" in name for name in condition_names)
        assert any("turbid" in name for name in condition_names)
        assert any("clear" in name for name in condition_names)
        assert any("season" in name for name in condition_names)
    
    def test_environmental_condition_structure(self):
        """Test environmental condition data structure."""
        conditions = self.validator.get_environmental_conditions()
        
        for condition in conditions:
            assert hasattr(condition, 'name')
            assert hasattr(condition, 'description')
            assert hasattr(condition, 'parameters')
            assert hasattr(condition, 'expected_behavior')
            assert hasattr(condition, 'tolerance')
            
            assert isinstance(condition.name, str)
            assert isinstance(condition.description, str)
            assert isinstance(condition.parameters, dict)
            assert isinstance(condition.expected_behavior, str)
            assert isinstance(condition.tolerance, float)
    
    def test_tidal_correction_application(self):
        """Test tidal height correction application."""
        # Test low tide condition
        low_tide_condition = EnvironmentalCondition(
            name="test_low_tide",
            description="Test low tide",
            parameters={
                "tidal_height": -1.0,
                "correction_factor": -0.225,
            },
            expected_behavior="Reduced extent"
        )
        
        corrected_mask = self.validator.apply_tidal_correction(
            self.mock_detection_mask, low_tide_condition
        )
        
        assert corrected_mask.shape == self.mock_detection_mask.shape
        assert np.all(corrected_mask >= 0)  # No negative values
        assert np.all(corrected_mask <= 1)  # No values > 1
        
        # Low tide should generally reduce detection
        assert np.mean(corrected_mask) <= np.mean(self.mock_detection_mask)
    
    def test_tidal_correction_high_tide(self):
        """Test high tide correction increases detection."""
        high_tide_condition = EnvironmentalCondition(
            name="test_high_tide",
            description="Test high tide",
            parameters={
                "tidal_height": 1.0,
                "correction_factor": -0.225,
            },
            expected_behavior="Increased extent"
        )
        
        corrected_mask = self.validator.apply_tidal_correction(
            self.mock_detection_mask, high_tide_condition
        )
        
        # High tide should generally increase detection
        assert np.mean(corrected_mask) >= np.mean(self.mock_detection_mask)
    
    def test_tidal_correction_no_parameters(self):
        """Test tidal correction with no tidal parameters."""
        no_tidal_condition = EnvironmentalCondition(
            name="test_no_tidal",
            description="Test without tidal parameters",
            parameters={},
            expected_behavior="No change"
        )
        
        corrected_mask = self.validator.apply_tidal_correction(
            self.mock_detection_mask, no_tidal_condition
        )
        
        # Should return unchanged mask
        np.testing.assert_array_equal(corrected_mask, self.mock_detection_mask)
    
    def test_consistency_score_calculation(self):
        """Test spatial consistency score calculation."""
        # Test with uniform detection (high consistency)
        uniform_mask = np.full((50, 50), 0.5)
        consistency_high = self.validator._calculate_consistency_score(uniform_mask)
        assert consistency_high > 0.8  # Should be very consistent
        
        # Test with highly variable detection (low consistency)
        variable_mask = np.random.rand(50, 50)
        consistency_low = self.validator._calculate_consistency_score(variable_mask)
        assert consistency_low < consistency_high
        
        # Test with zero detection
        zero_mask = np.zeros((50, 50))
        consistency_zero = self.validator._calculate_consistency_score(zero_mask)
        assert consistency_zero == 0.0
    
    def test_condition_success_evaluation(self):
        """Test condition success evaluation logic."""
        # Test tidal condition success
        tidal_condition = EnvironmentalCondition(
            name="low_tide_test",
            description="Test condition",
            parameters={},
            expected_behavior="Test"
        )
        
        # Detection rate within expected range for tidal conditions
        assert self.validator._evaluate_condition_success(0.15, tidal_condition, 0.8) == True
        
        # Detection rate outside expected range
        assert self.validator._evaluate_condition_success(0.5, tidal_condition, 0.8) == False
        assert self.validator._evaluate_condition_success(0.01, tidal_condition, 0.8) == False
    
    def test_condition_success_different_types(self):
        """Test success evaluation for different condition types."""
        # Turbid water condition
        turbid_condition = EnvironmentalCondition(
            name="turbid_water_test",
            description="Test turbid condition",
            parameters={},
            expected_behavior="Test"
        )
        
        assert self.validator._evaluate_condition_success(0.10, turbid_condition, 0.8) == True
        assert self.validator._evaluate_condition_success(0.5, turbid_condition, 0.8) == False
        
        # Clear water condition
        clear_condition = EnvironmentalCondition(
            name="clear_water_test",
            description="Test clear condition",
            parameters={},
            expected_behavior="Test"
        )
        
        assert self.validator._evaluate_condition_success(0.20, clear_condition, 0.8) == True
        
        # Peak season condition
        peak_condition = EnvironmentalCondition(
            name="peak_season_test",
            description="Test peak season",
            parameters={},
            expected_behavior="Test"
        )
        
        assert self.validator._evaluate_condition_success(0.25, peak_condition, 0.8) == True
    
    def test_failed_result_creation(self):
        """Test creation of failed test results."""
        test_condition = EnvironmentalCondition(
            name="test_condition",
            description="Test condition",
            parameters={},
            expected_behavior="Test"
        )
        
        failed_result = self.validator._create_failed_result(
            test_condition, "Test error message"
        )
        
        assert isinstance(failed_result, EnvironmentalTestResult)
        assert failed_result.condition == test_condition
        assert failed_result.detection_rate == 0.0
        assert failed_result.consistency_score == 0.0
        assert failed_result.success == False
        assert "error" in failed_result.metadata
        assert failed_result.metadata["error"] == "Test error message"
    
    @pytest.mark.asyncio
    @patch('kelpie_carbon_v1.validation.environmental_testing.fetch_sentinel_tiles')
    @patch('kelpie_carbon_v1.validation.environmental_testing.create_skema_kelp_detection_mask')
    async def test_environmental_condition_testing_success(self, mock_create_mask, mock_fetch):
        """Test successful environmental condition testing."""
        # Mock satellite data fetch - return dict with 'data' key as expected by the function
        mock_fetch.return_value = {"data": self.mock_dataset}
        
        # Mock detection mask creation
        mock_create_mask.return_value = self.mock_detection_mask
        
        test_condition = EnvironmentalCondition(
            name="test_condition",
            description="Test environmental condition",
            parameters={
                "tidal_height": 0.5,
                "correction_factor": -0.225,
            },
            expected_behavior="Test behavior"
        )
        
        result = await self.validator.test_environmental_condition(
            test_condition, 50.0, -126.0, "2023-07-01", "2023-07-31"
        )
        
        assert isinstance(result, EnvironmentalTestResult)
        assert result.condition == test_condition
        assert result.detection_rate >= 0.0
        assert result.consistency_score >= 0.0
        assert isinstance(result.success, bool)
        assert "location" in result.metadata
        assert "condition_parameters" in result.metadata
        
        # Verify mocks were called
        mock_fetch.assert_called_once()
        mock_create_mask.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('kelpie_carbon_v1.validation.environmental_testing.fetch_sentinel_tiles')
    async def test_environmental_condition_testing_no_data(self, mock_fetch):
        """Test environmental condition testing with no satellite data."""
        # Mock no data available
        mock_fetch.return_value = None
        
        test_condition = EnvironmentalCondition(
            name="test_condition",
            description="Test condition",
            parameters={},
            expected_behavior="Test"
        )
        
        result = await self.validator.test_environmental_condition(
            test_condition, 50.0, -126.0, "2023-07-01", "2023-07-31"
        )
        
        assert result.success == False
        assert "error" in result.metadata
        assert "No satellite data available" in result.metadata["error"]
    
    @pytest.mark.asyncio
    @patch('kelpie_carbon_v1.validation.environmental_testing.fetch_sentinel_tiles')
    async def test_environmental_condition_testing_empty_dataset(self, mock_fetch):
        """Test environmental condition testing with empty dataset."""
        # Mock empty dataset
        empty_dataset = xr.Dataset({})
        mock_fetch.return_value = empty_dataset
        
        test_condition = EnvironmentalCondition(
            name="test_condition",
            description="Test condition",
            parameters={},
            expected_behavior="Test"
        )
        
        result = await self.validator.test_environmental_condition(
            test_condition, 50.0, -126.0, "2023-07-01", "2023-07-31"
        )
        
        assert result.success == False
        assert "error" in result.metadata
    
    @pytest.mark.asyncio
    @patch('kelpie_carbon_v1.validation.environmental_testing.fetch_sentinel_tiles')
    async def test_environmental_condition_testing_exception(self, mock_fetch):
        """Test environmental condition testing with exception."""
        # Mock exception during fetch
        mock_fetch.side_effect = Exception("Test exception")
        
        test_condition = EnvironmentalCondition(
            name="test_condition",
            description="Test condition",
            parameters={},
            expected_behavior="Test"
        )
        
        result = await self.validator.test_environmental_condition(
            test_condition, 50.0, -126.0, "2023-07-01", "2023-07-31"
        )
        
        assert result.success == False
        assert "error" in result.metadata
        assert "Test exception" in result.metadata["error"]
    
    def test_report_generation(self):
        """Test environmental testing report generation."""
        # Create mock results
        successful_result = EnvironmentalTestResult(
            condition=EnvironmentalCondition("test1", "desc1", {}, "behavior1"),
            detection_rate=0.15,
            consistency_score=0.8,
            performance_metrics={"mean_detection": 0.15, "std_detection": 0.05},
            success=True,
            timestamp=datetime.now(),
            metadata={}
        )
        
        failed_result = EnvironmentalTestResult(
            condition=EnvironmentalCondition("test2", "desc2", {}, "behavior2"),
            detection_rate=0.0,
            consistency_score=0.0,
            performance_metrics={},
            success=False,
            timestamp=datetime.now(),
            metadata={"error": "Test error"}
        )
        
        results = [successful_result, failed_result]
        report = self.validator._generate_report(results)
        
        assert "timestamp" in report
        assert "summary" in report
        assert "detailed_results" in report
        
        # Check summary
        summary = report["summary"]
        assert summary["total_conditions"] == 2
        assert summary["successful_tests"] == 1
        assert summary["success_rate"] == 0.5
        
        # Check detailed results
        detailed = report["detailed_results"]
        assert len(detailed) == 2
        assert detailed[0]["condition_name"] == "test1"
        assert detailed[0]["success"] == True
        assert detailed[1]["condition_name"] == "test2"
        assert detailed[1]["success"] == False
    
    @pytest.mark.asyncio
    @patch('kelpie_carbon_v1.validation.environmental_testing.EnvironmentalRobustnessValidator.test_environmental_condition')
    async def test_comprehensive_testing(self, mock_test_condition):
        """Test comprehensive environmental testing."""
        # Mock individual condition testing
        mock_result = EnvironmentalTestResult(
            condition=EnvironmentalCondition("test", "desc", {}, "behavior"),
            detection_rate=0.15,
            consistency_score=0.8,
            performance_metrics={"mean_detection": 0.15},
            success=True,
            timestamp=datetime.now(),
            metadata={}
        )
        mock_test_condition.return_value = mock_result
        
        report = await self.validator.run_comprehensive_testing(
            50.0, -126.0, "2023-07-15"
        )
        
        assert "timestamp" in report
        assert "summary" in report
        assert "detailed_results" in report
        
        # Should have tested all environmental conditions
        expected_conditions = len(self.validator.get_environmental_conditions())
        assert mock_test_condition.call_count == expected_conditions
        assert len(report["detailed_results"]) == expected_conditions


class TestEnvironmentalTestingIntegration:
    """Integration tests for environmental testing."""
    
    @pytest.mark.asyncio
    @patch('kelpie_carbon_v1.validation.environmental_testing.EnvironmentalRobustnessValidator.test_environmental_condition', new_callable=AsyncMock)
    async def test_tidal_effects_convenience_function(self, mock_test_condition):
        """Test tidal effects convenience function."""
        from kelpie_carbon_v1.validation.environmental_testing import validate_tidal_effects
        
        # Mock the individual condition testing that happens in validate_tidal_effects
        mock_result = EnvironmentalTestResult(
            condition=EnvironmentalCondition("tidal_test", "desc", {}, "behavior"),
            detection_rate=0.15,
            consistency_score=0.8,
            performance_metrics={"mean_detection": 0.15},
            success=True,
            timestamp=datetime.now(),
            metadata={}
        )
        mock_test_condition.return_value = mock_result
        
        result = await validate_tidal_effects(50.0, -126.0, "2023-07-15")
        
        assert "summary" in result
        assert "detailed_results" in result
        # Should have called test_environmental_condition for each tidal condition
        assert mock_test_condition.call_count > 0
    
    def test_environmental_testing_imports(self):
        """Test that all required imports are available."""
        from kelpie_carbon_v1.validation.environmental_testing import (
            EnvironmentalRobustnessValidator,
            EnvironmentalCondition,
            EnvironmentalTestResult,
        )
        
        # Verify classes and functions are properly imported
        assert EnvironmentalRobustnessValidator is not None
        assert EnvironmentalCondition is not None
        assert EnvironmentalTestResult is not None


class TestEnvironmentalConditionValidation:
    """Test environmental condition validation logic."""
    
    def test_tidal_correction_factors(self):
        """Test that tidal correction factors match research values."""
        validator = EnvironmentalRobustnessValidator()
        conditions = validator.get_environmental_conditions()
        
        # Find tidal conditions
        tidal_conditions = [c for c in conditions if "tide" in c.name or "tidal" in c.name]
        
        for condition in tidal_conditions:
            if "correction_factor" in condition.parameters:
                correction_factor = condition.parameters["correction_factor"]
                
                # Verify correction factors match Timmer et al. (2024) research
                # Low current: -22.5% per meter, High current: -35.5% per meter
                assert correction_factor in [-0.225, -0.355], \
                    f"Correction factor {correction_factor} doesn't match research values"
    
    def test_water_clarity_parameters(self):
        """Test water clarity condition parameters."""
        validator = EnvironmentalRobustnessValidator()
        conditions = validator.get_environmental_conditions()
        
        # Find water clarity conditions
        clarity_conditions = [c for c in conditions if "turbid" in c.name or "clear" in c.name]
        
        for condition in clarity_conditions:
            if "secchi_depth" in condition.parameters:
                secchi_depth = condition.parameters["secchi_depth"]
                
                if "turbid" in condition.name:
                    # Turbid water should have Secchi depth < 4m
                    assert secchi_depth < 4.0, \
                        f"Turbid condition has Secchi depth {secchi_depth} >= 4m"
                elif "clear" in condition.name:
                    # Clear water should have Secchi depth > 7m
                    assert secchi_depth > 7.0, \
                        f"Clear condition has Secchi depth {secchi_depth} <= 7m"
    
    def test_seasonal_parameters(self):
        """Test seasonal condition parameters."""
        validator = EnvironmentalRobustnessValidator()
        conditions = validator.get_environmental_conditions()
        
        # Find seasonal conditions
        seasonal_conditions = [c for c in conditions if "season" in c.name]
        
        for condition in seasonal_conditions:
            if "growth_factor" in condition.parameters:
                growth_factor = condition.parameters["growth_factor"]
                
                if "peak" in condition.name:
                    # Peak season should have growth factor > 1.0
                    assert growth_factor > 1.0, \
                        f"Peak season has growth factor {growth_factor} <= 1.0"
                elif "off" in condition.name:
                    # Off season should have growth factor < 1.0
                    assert growth_factor < 1.0, \
                        f"Off season has growth factor {growth_factor} >= 1.0"


# Integration test for real-world conditions
@pytest.mark.integration
class TestEnvironmentalRealWorldScenarios:
    """Test environmental conditions with realistic scenarios."""
    
    def test_broughton_archipelago_tidal_scenario(self):
        """Test realistic tidal scenario for Broughton Archipelago."""
        validator = EnvironmentalRobustnessValidator()
        
        # Create realistic tidal condition for Broughton Archipelago
        # (Strong tidal currents common in this region)
        realistic_condition = EnvironmentalCondition(
            name="broughton_high_tide",
            description="High tide conditions at Broughton Archipelago",
            parameters={
                "tidal_height": 1.5,  # 1.5m above mean (realistic for region)
                "current_speed": 20.0,  # 20 cm/s (strong currents)
                "correction_factor": -0.355,  # High current correction
            },
            expected_behavior="Increased extent with current correction"
        )
        
        # Test tidal correction
        mock_detection = np.random.rand(50, 50) * 0.2  # Base 20% detection
        corrected = validator.apply_tidal_correction(mock_detection, realistic_condition)
        
        # Should increase detection due to high tide
        assert np.mean(corrected) > np.mean(mock_detection)
        
        # Should still be within reasonable bounds
        assert np.all(corrected >= 0)
        assert np.all(corrected <= 1)
    
    def test_monterey_bay_clear_water_scenario(self):
        """Test clear water scenario for Monterey Bay."""
        validator = EnvironmentalRobustnessValidator()
        
        # Monterey Bay typically has very clear water
        clear_water_condition = EnvironmentalCondition(
            name="monterey_clear_water",
            description="Clear water conditions at Monterey Bay",
            parameters={
                "secchi_depth": 12.0,  # Very clear water
                "turbidity_factor": 1.0,
                "waf_intensity": 0.9,  # Reduced WAF needed
            },
            expected_behavior="Optimal detection in clear water"
        )
        
        # Test success evaluation
        success = validator._evaluate_condition_success(0.25, clear_water_condition, 0.8)
        assert success == True  # Should be successful with good detection rate
    
    def test_seasonal_variation_realistic_ranges(self):
        """Test seasonal variations with realistic detection ranges."""
        validator = EnvironmentalRobustnessValidator()
        
        # Peak season (July-September) - should have high detection
        peak_success = validator._evaluate_condition_success(
            0.3,  # 30% detection rate
            EnvironmentalCondition("peak_season_test", "", {}, ""),
            0.8  # consistency score
        )
        assert peak_success == True
        
        # Off season (October-April) - should accept lower detection
        off_season_success = validator._evaluate_condition_success(
            0.1,  # 10% detection rate
            EnvironmentalCondition("off_season_test", "", {}, ""),
            0.8  # consistency score
        )
        # This should be evaluated as acceptable for off-season
        # The actual implementation may need adjustment based on this test 