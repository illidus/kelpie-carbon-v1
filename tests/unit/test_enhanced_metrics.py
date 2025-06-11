"""
Tests for Enhanced Validation Metrics - Task ML1
Tests RMSE, MAE, R² accuracy metrics for biomass and carbon validation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from kelpie_carbon_v1.validation.enhanced_metrics import (
    EnhancedValidationMetrics,
    ValidationCoordinate,
    BiomassValidationData,
    ValidationMetricsResult,
    create_enhanced_validation_metrics,
    validate_four_coordinate_sites,
    calculate_validation_summary
)


class TestEnhancedValidationMetrics:
    """Test suite for enhanced validation metrics functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics_calculator = EnhancedValidationMetrics()
        
        # Sample validation data
        self.predicted_biomass = np.array([1.2, 2.1, 0.8, 1.5, 2.3])
        self.observed_biomass = np.array([1.0, 2.0, 1.0, 1.4, 2.1])
        
        # Sample coordinates
        self.bc_coordinate = ValidationCoordinate(
            name="British Columbia Test",
            latitude=50.1163,
            longitude=-125.2735,
            species="Nereocystis luetkeana",
            region="Pacific Northwest"
        )
        
        self.ca_coordinate = ValidationCoordinate(
            name="California Test",
            latitude=36.6002,
            longitude=-121.9015,
            species="Macrocystis pyrifera",
            region="California Current"
        )
    
    def test_calculate_biomass_accuracy_metrics_basic(self):
        """Test basic RMSE, MAE, R² calculation for biomass."""
        metrics = self.metrics_calculator.calculate_biomass_accuracy_metrics(
            self.predicted_biomass, self.observed_biomass
        )
        
        # Check that primary metrics exist
        assert 'rmse_biomass_kg_m2' in metrics
        assert 'mae_biomass_kg_m2' in metrics
        assert 'r2_biomass_correlation' in metrics
        
        # Check metric values are reasonable
        assert metrics['rmse_biomass_kg_m2'] >= 0
        assert metrics['mae_biomass_kg_m2'] >= 0
        assert -1 <= metrics['r2_biomass_correlation'] <= 1
        
        # Check additional metrics
        assert 'pearson_correlation' in metrics
        assert 'uncertainty_bounds_95' in metrics
        assert metrics['n_valid_points'] == 5
    
    def test_calculate_carbon_accuracy_metrics_basic(self):
        """Test basic RMSE, MAE, R² calculation for carbon."""
        carbon_factors = {'carbon_content_ratio': 0.30}
        
        metrics = self.metrics_calculator.calculate_carbon_accuracy_metrics(
            self.predicted_biomass, self.observed_biomass, carbon_factors
        )
        
        # Check that primary carbon metrics exist
        assert 'rmse_carbon_tc_hectare' in metrics
        assert 'mae_carbon_tc_hectare' in metrics
        assert 'r2_carbon_correlation' in metrics
        
        # Check conversion factor is used
        assert metrics['carbon_content_ratio_used'] == 0.30
        
        # Check sequestration metrics
        assert 'sequestration_rate_rmse_tc_year' in metrics
        assert 'carbon_bias_percentage' in metrics
    
    def test_empty_arrays_handling(self):
        """Test handling of empty input arrays."""
        empty_array = np.array([])
        
        biomass_metrics = self.metrics_calculator.calculate_biomass_accuracy_metrics(
            empty_array, empty_array
        )
        
        # Should return empty metrics without crashing
        assert biomass_metrics['rmse_biomass_kg_m2'] == 0.0
        assert biomass_metrics['mae_biomass_kg_m2'] == 0.0
        assert biomass_metrics['r2_biomass_correlation'] == 0.0
        assert biomass_metrics['n_valid_points'] == 0
    
    def test_nan_handling(self):
        """Test handling of NaN values in input data."""
        predicted_with_nan = np.array([1.2, np.nan, 0.8, 1.5, 2.3])
        observed_with_nan = np.array([1.0, 2.0, np.nan, 1.4, 2.1])
        
        metrics = self.metrics_calculator.calculate_biomass_accuracy_metrics(
            predicted_with_nan, observed_with_nan
        )
        
        # Should process only valid data points (3 out of 5)
        assert metrics['n_valid_points'] == 3
        assert metrics['rmse_biomass_kg_m2'] >= 0
        assert not np.isnan(metrics['rmse_biomass_kg_m2'])
    
    def test_mismatched_array_lengths(self):
        """Test handling of mismatched array lengths."""
        predicted_short = np.array([1.2, 2.1, 0.8])
        observed_long = np.array([1.0, 2.0, 1.0, 1.4, 2.1])
        
        metrics = self.metrics_calculator.calculate_biomass_accuracy_metrics(
            predicted_short, observed_long
        )
        
        # Should process only the overlapping length (3)
        assert metrics['n_valid_points'] == 3
        assert metrics['rmse_biomass_kg_m2'] >= 0
    
    def test_validation_coordinates_constants(self):
        """Test that validation coordinates are properly defined."""
        coords = self.metrics_calculator.VALIDATION_COORDINATES
        
        assert len(coords) == 4
        
        # Check specific coordinates
        bc_coord = next(c for c in coords if c.name == "British Columbia")
        assert bc_coord.latitude == 50.1163
        assert bc_coord.longitude == -125.2735
        assert bc_coord.species == "Nereocystis luetkeana"
        
        ca_coord = next(c for c in coords if c.name == "California")
        assert ca_coord.latitude == 36.6002
        assert ca_coord.longitude == -121.9015
        assert ca_coord.species == "Macrocystis pyrifera"
        
        tasmania_coord = next(c for c in coords if c.name == "Tasmania")
        assert tasmania_coord.latitude == -43.1
        assert tasmania_coord.longitude == 147.3
        
        broughton_coord = next(c for c in coords if c.name == "Broughton Archipelago")
        assert broughton_coord.latitude == 50.0833
        assert broughton_coord.longitude == -126.1667
    
    def test_species_carbon_ratios(self):
        """Test species-specific carbon content ratios."""
        ratios = self.metrics_calculator.SPECIES_CARBON_RATIOS
        
        assert 'Nereocystis luetkeana' in ratios
        assert 'Macrocystis pyrifera' in ratios
        assert 'Mixed' in ratios
        
        # Check reasonable values (should be between 0.2 and 0.4)
        for species, ratio in ratios.items():
            assert 0.2 <= ratio <= 0.4
    
    def test_validate_model_predictions_against_real_data(self):
        """Test comprehensive validation against multiple sites."""
        # Create sample validation data
        validation_data = [
            BiomassValidationData(
                coordinate=self.bc_coordinate,
                observed_biomass_kg_m2=self.observed_biomass,
                predicted_biomass_kg_m2=self.predicted_biomass,
                carbon_content_ratio=0.30,
                measurement_dates=['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'],
                measurement_metadata={'source': 'test_data'}
            ),
            BiomassValidationData(
                coordinate=self.ca_coordinate,
                observed_biomass_kg_m2=self.observed_biomass * 1.1,  # Slightly different data
                predicted_biomass_kg_m2=self.predicted_biomass * 1.2,
                carbon_content_ratio=0.28,
                measurement_dates=['2023-01-15', '2023-02-15', '2023-03-15', '2023-04-15', '2023-05-15'],
                measurement_metadata={'source': 'test_data'}
            )
        ]
        
        results = self.metrics_calculator.validate_model_predictions_against_real_data(validation_data)
        
        assert len(results) == 2
        assert 'British Columbia Test' in results
        assert 'California Test' in results
        
        # Check result structure
        bc_result = results['British Columbia Test']
        assert isinstance(bc_result, ValidationMetricsResult)
        assert bc_result.coordinate.name == 'British Columbia Test'
        assert 'rmse_biomass_kg_m2' in bc_result.biomass_metrics
        assert 'rmse_carbon_tc_hectare' in bc_result.carbon_metrics
        assert 'residual_statistics' in bc_result.uncertainty_analysis
    
    def test_generate_validation_summary(self):
        """Test generation of validation summary across all coordinates."""
        # Create mock validation results
        mock_result1 = ValidationMetricsResult(
            coordinate=self.bc_coordinate,
            biomass_metrics={
                'r2_biomass_correlation': 0.85,
                'rmse_biomass_kg_m2': 0.15,
                'mae_biomass_kg_m2': 0.12,
                'n_valid_points': 5
            },
            carbon_metrics={
                'r2_carbon_correlation': 0.82,
                'rmse_carbon_tc_hectare': 1.5,
                'mae_carbon_tc_hectare': 1.2
            },
            uncertainty_analysis={},
            species_specific_metrics={},
            temporal_metrics={}
        )
        
        mock_result2 = ValidationMetricsResult(
            coordinate=self.ca_coordinate,
            biomass_metrics={
                'r2_biomass_correlation': 0.78,
                'rmse_biomass_kg_m2': 0.18,
                'mae_biomass_kg_m2': 0.15,
                'n_valid_points': 5
            },
            carbon_metrics={
                'r2_carbon_correlation': 0.75,
                'rmse_carbon_tc_hectare': 1.8,
                'mae_carbon_tc_hectare': 1.5
            },
            uncertainty_analysis={},
            species_specific_metrics={},
            temporal_metrics={}
        )
        
        validation_results = {
            'British Columbia Test': mock_result1,
            'California Test': mock_result2
        }
        
        summary = self.metrics_calculator.generate_validation_summary(validation_results)
        
        assert 'validation_summary' in summary
        assert summary['validation_summary']['total_sites_validated'] == 2
        
        # Check overall performance metrics
        overall = summary['validation_summary']['overall_performance']
        assert 'biomass_metrics' in overall
        assert 'carbon_metrics' in overall
        
        # Check mean values are reasonable
        assert 0.8 < overall['biomass_metrics']['mean_r2'] < 0.9  # Average of 0.85 and 0.78
        assert 0.16 < overall['biomass_metrics']['mean_rmse_kg_m2'] < 0.17  # Average of 0.15 and 0.18
    
    def test_prediction_intervals_calculation(self):
        """Test uncertainty quantification with prediction intervals."""
        confidence_level = 0.95
        intervals = self.metrics_calculator._calculate_prediction_intervals(
            self.predicted_biomass, self.observed_biomass, confidence_level
        )
        
        assert 'lower_bound' in intervals
        assert 'upper_bound' in intervals
        assert 'confidence_level' in intervals
        assert 'residual_std' in intervals
        
        assert intervals['confidence_level'] == confidence_level
        assert intervals['lower_bound'] <= 0  # Should be negative (lower bound)
        assert intervals['upper_bound'] >= 0  # Should be positive (upper bound)
        assert intervals['residual_std'] >= 0
    
    def test_species_specific_metrics_calculation(self):
        """Test species-specific performance metrics."""
        biomass_metrics = {
            'r2_biomass_correlation': 0.85,
            'pearson_correlation': 0.88
        }
        carbon_metrics = {
            'r2_carbon_correlation': 0.82,
            'carbon_pearson_correlation': 0.84
        }
        
        species_metrics = self.metrics_calculator._calculate_species_specific_metrics(
            'Nereocystis luetkeana', biomass_metrics, carbon_metrics
        )
        
        assert species_metrics['species'] == 'Nereocystis luetkeana'
        assert species_metrics['carbon_content_ratio'] == 0.30  # For Nereocystis
        assert 'biomass_performance_score' in species_metrics
        assert 'carbon_performance_score' in species_metrics
        assert 'overall_species_score' in species_metrics
        
        # Check overall score is average of biomass and carbon R²
        expected_overall = (0.85 + 0.82) / 2
        assert abs(species_metrics['overall_species_score'] - expected_overall) < 0.01


class TestFactoryFunctions:
    """Test factory functions for enhanced metrics."""
    
    def test_create_enhanced_validation_metrics(self):
        """Test factory function for creating metrics calculator."""
        calculator = create_enhanced_validation_metrics()
        assert isinstance(calculator, EnhancedValidationMetrics)
    
    def test_validate_four_coordinate_sites_function(self):
        """Test the convenience function for validating four coordinates."""
        # Create minimal validation data
        bc_coord = ValidationCoordinate(
            name="British Columbia",
            latitude=50.1163,
            longitude=-125.2735,
            species="Nereocystis luetkeana",
            region="Pacific Northwest"
        )
        
        validation_data = [
            BiomassValidationData(
                coordinate=bc_coord,
                observed_biomass_kg_m2=np.array([1.0, 2.0]),
                predicted_biomass_kg_m2=np.array([1.1, 1.9]),
                carbon_content_ratio=0.30,
                measurement_dates=['2023-01-01', '2023-02-01'],
                measurement_metadata={}
            )
        ]
        
        results = validate_four_coordinate_sites(validation_data)
        
        assert len(results) == 1
        assert 'British Columbia' in results
        assert isinstance(results['British Columbia'], ValidationMetricsResult)
    
    def test_calculate_validation_summary_function(self):
        """Test the convenience function for calculating validation summary."""
        # Create minimal validation results
        bc_coord = ValidationCoordinate(
            name="British Columbia",
            latitude=50.1163,
            longitude=-125.2735,
            species="Nereocystis luetkeana",
            region="Pacific Northwest"
        )
        
        mock_result = ValidationMetricsResult(
            coordinate=bc_coord,
            biomass_metrics={'r2_biomass_correlation': 0.85, 'rmse_biomass_kg_m2': 0.15, 'n_valid_points': 2},
            carbon_metrics={'r2_carbon_correlation': 0.82},
            uncertainty_analysis={},
            species_specific_metrics={},
            temporal_metrics={}
        )
        
        validation_results = {'British Columbia': mock_result}
        summary = calculate_validation_summary(validation_results)
        
        assert 'validation_summary' in summary
        assert summary['validation_summary']['total_sites_validated'] == 1


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics_calculator = EnhancedValidationMetrics()
    
    def test_all_zero_observations(self):
        """Test handling when all observed values are zero."""
        predicted = np.array([0.1, 0.2, 0.3])
        observed = np.array([0.0, 0.0, 0.0])
        
        metrics = self.metrics_calculator.calculate_biomass_accuracy_metrics(predicted, observed)
        
        # Should not crash and return reasonable values
        assert metrics['n_valid_points'] == 3
        assert metrics['rmse_biomass_kg_m2'] > 0  # RMSE should be positive
        assert not np.isnan(metrics['mae_biomass_kg_m2'])
    
    def test_identical_predictions_and_observations(self):
        """Test perfect predictions (identical arrays)."""
        values = np.array([1.0, 2.0, 3.0])
        
        metrics = self.metrics_calculator.calculate_biomass_accuracy_metrics(values, values)
        
        # Perfect predictions should give R² = 1, RMSE = 0, MAE = 0
        assert abs(metrics['r2_biomass_correlation'] - 1.0) < 1e-10
        assert abs(metrics['rmse_biomass_kg_m2']) < 1e-10
        assert abs(metrics['mae_biomass_kg_m2']) < 1e-10
    
    def test_single_data_point(self):
        """Test handling of single data point."""
        predicted = np.array([1.5])
        observed = np.array([1.2])
        
        metrics = self.metrics_calculator.calculate_biomass_accuracy_metrics(predicted, observed)
        
        # Should handle single point gracefully
        assert metrics['n_valid_points'] == 1
        assert metrics['rmse_biomass_kg_m2'] == abs(predicted[0] - observed[0])
        assert metrics['mae_biomass_kg_m2'] == abs(predicted[0] - observed[0])
    
    def test_validation_with_empty_results(self):
        """Test validation summary generation with empty results."""
        empty_results = {}
        
        summary = self.metrics_calculator.generate_validation_summary(empty_results)
        
        # Should return empty summary without crashing
        assert summary == {}
    
    def test_temporal_metrics_with_invalid_dates(self):
        """Test temporal metrics with invalid date information."""
        predicted = np.array([1.0, 2.0])
        observed = np.array([1.1, 1.9])
        
        # Test with mismatched date count
        invalid_dates = ['2023-01-01']  # Only one date for two data points
        temporal_metrics = self.metrics_calculator._calculate_temporal_metrics(
            invalid_dates, predicted, observed
        )
        
        assert temporal_metrics['temporal_analysis_available'] == False
        
        # Test with empty dates
        empty_dates = []
        temporal_metrics = self.metrics_calculator._calculate_temporal_metrics(
            empty_dates, predicted, observed
        )
        
        assert temporal_metrics['temporal_analysis_available'] == False 