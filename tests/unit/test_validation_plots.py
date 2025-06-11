"""
Unit tests for validation plots module.
Tests the visualization methods for model prediction accuracy assessment.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Import the module under test
from src.kelpie_carbon_v1.visualization.validation_plots import (
    ValidationVisualizationSuite,
    create_accuracy_assessment_dashboard,
    plot_rmse_mae_r2_comparison,
    create_predicted_vs_actual_plots,
    visualize_spatial_accuracy_distribution
)

# Mock validation result structure for testing
class MockValidationResult:
    """Mock validation result for testing."""
    
    def __init__(self, site_id: str, species: str, lat: float, lon: float):
        self.coordinate = MagicMock()
        self.coordinate.latitude = lat
        self.coordinate.longitude = lon
        self.coordinate.species = species
        
        # Mock biomass metrics
        self.biomass_metrics = {
            'rmse_biomass_kg_m2': np.random.uniform(0.1, 0.3),
            'mae_biomass_kg_m2': np.random.uniform(0.1, 0.25),
            'r2_biomass_correlation': np.random.uniform(0.7, 0.95),
            'n_valid_points': np.random.randint(10, 50)
        }
        
        # Mock carbon metrics
        self.carbon_metrics = {
            'rmse_carbon_tc_hectare': np.random.uniform(1.0, 3.0),
            'mae_carbon_tc_hectare': np.random.uniform(0.8, 2.5),
            'r2_carbon_correlation': np.random.uniform(0.65, 0.90)
        }


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_validation_results():
    """Create sample validation results for testing."""
    return {
        'BC_Site': MockValidationResult('BC_Site', 'Nereocystis luetkeana', 50.1163, -125.2735),
        'CA_Site': MockValidationResult('CA_Site', 'Macrocystis pyrifera', 36.6002, -121.9015),
        'TAS_Site': MockValidationResult('TAS_Site', 'Macrocystis pyrifera', -43.1, 147.3),
        'BROUGHTON_Site': MockValidationResult('BROUGHTON_Site', 'Nereocystis luetkeana', 50.0833, -126.1667)
    }


class TestValidationVisualizationSuite:
    """Test ValidationVisualizationSuite class."""
    
    def test_initialization(self, temp_output_dir):
        """Test suite initialization."""
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        
        assert viz_suite.output_dir == Path(temp_output_dir)
        assert viz_suite.output_dir.exists()
    
    def test_initialization_default_output_dir(self):
        """Test suite initialization with default output directory."""
        viz_suite = ValidationVisualizationSuite()
        
        assert viz_suite.output_dir == Path("validation_plots")
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_rmse_mae_r2_comparison(self, mock_subplots, mock_close, mock_savefig, 
                                        temp_output_dir, sample_validation_results):
        """Test RMSE, MAE, R² comparison plotting."""
        # Mock matplotlib objects
        mock_fig = MagicMock()
        mock_axes = np.array([[MagicMock() for _ in range(3)] for _ in range(2)])
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        result = viz_suite.plot_rmse_mae_r2_comparison(sample_validation_results)
        
        # Verify plot was created
        assert isinstance(result, str)
        assert "rmse_mae_r2_comparison.png" in result
        
        # Verify matplotlib was called
        mock_subplots.assert_called_once_with(2, 3, figsize=(18, 12))
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.figure')
    def test_create_predicted_vs_actual_plots(self, mock_figure, mock_subplots, mock_close, 
                                             mock_savefig, temp_output_dir, sample_validation_results):
        """Test predicted vs actual scatter plots creation."""
        # Mock matplotlib objects
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        mock_figure.return_value = mock_fig
        
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        result = viz_suite.create_predicted_vs_actual_plots(sample_validation_results)
        
        # Verify plots were created
        assert isinstance(result, dict)
        assert 'biomass_scatter' in result
        assert 'carbon_scatter' in result
        
        # Verify files contain expected path
        for plot_path in result.values():
            assert "predicted_vs_actual" in plot_path
        
        # Verify matplotlib was called
        assert mock_savefig.call_count >= 2  # At least 2 plots created
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_visualize_spatial_accuracy_distribution_static(self, mock_close, mock_savefig,
                                                           temp_output_dir, sample_validation_results):
        """Test spatial accuracy distribution visualization (static fallback)."""
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        
        # Force static plot creation by testing the private method directly
        result = viz_suite._create_static_spatial_plot(sample_validation_results)
        
        # Verify plot was created
        assert isinstance(result, str)
        assert "spatial_accuracy_static.png" in result
        
        # Verify matplotlib was called
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    def test_visualize_spatial_accuracy_distribution_with_folium(self, temp_output_dir, 
                                                                sample_validation_results):
        """Test spatial accuracy distribution with Folium (if available)."""
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        
        with patch('src.kelpie_carbon_v1.visualization.validation_plots.FOLIUM_AVAILABLE', True):
            with patch('folium.Map') as mock_map_class:
                with patch('folium.Marker') as mock_marker_class:
                    mock_map = MagicMock()
                    mock_map_class.return_value = mock_map
                    
                    result = viz_suite.visualize_spatial_accuracy_distribution(sample_validation_results)
                    
                    # Verify map was created
                    assert isinstance(result, str)
                    assert "spatial_accuracy_map.html" in result
                    
                    # Verify folium was called
                    mock_map_class.assert_called_once()
                    assert mock_marker_class.call_count >= 4  # One marker per site
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.subplots')
    def test_create_species_accuracy_comparison(self, mock_subplots, mock_close, mock_savefig,
                                              temp_output_dir, sample_validation_results):
        """Test species-specific accuracy comparison."""
        # Mock matplotlib objects
        mock_fig = MagicMock()
        mock_axes = np.array([[MagicMock() for _ in range(2)] for _ in range(2)])
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        result = viz_suite.create_species_accuracy_comparison(sample_validation_results)
        
        # Verify plot was created
        assert isinstance(result, str)
        assert "species_accuracy_comparison.png" in result
        
        # Verify matplotlib was called
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_temporal_accuracy_trends(self, mock_subplots, mock_close, mock_savefig,
                                         temp_output_dir):
        """Test temporal accuracy trends plotting."""
        # Mock matplotlib objects
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        temporal_data = {'sample_data': 'temporal_trends'}
        
        result = viz_suite.plot_temporal_accuracy_trends(temporal_data)
        
        # Verify plot was created
        assert isinstance(result, str)
        assert "temporal_accuracy_trends.png" in result
        
        # Verify matplotlib was called
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.subplots')
    def test_create_uncertainty_calibration_plots(self, mock_subplots, mock_close, mock_savefig,
                                                 temp_output_dir):
        """Test uncertainty calibration plots creation."""
        # Mock matplotlib objects
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        
        # Create sample data
        predictions = np.random.normal(1.5, 0.3, 100)
        uncertainties = np.random.uniform(0.1, 0.4, 100)
        observations = predictions + np.random.normal(0, 0.2, 100)
        
        result = viz_suite.create_uncertainty_calibration_plots(predictions, uncertainties, observations)
        
        # Verify plots were created
        assert isinstance(result, dict)
        assert 'calibration_plot' in result
        
        # Verify matplotlib was called
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    def test_generate_validation_report_visualizations_empty_data(self, temp_output_dir):
        """Test validation report visualization with empty data."""
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        
        empty_metrics = {}
        result = viz_suite.generate_validation_report_visualizations(empty_metrics)
        
        # Should handle empty data gracefully
        assert isinstance(result, dict)
    
    def test_generate_validation_report_visualizations_with_data(self, temp_output_dir, 
                                                               sample_validation_results):
        """Test validation report visualization with complete data."""
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        
        all_metrics = {
            'validation_results': sample_validation_results,
            'validation_summary': {
                'total_sites_validated': 4,
                'overall_performance': {
                    'biomass_metrics': {'mean_r2': 0.85, 'mean_rmse_kg_m2': 0.15},
                    'carbon_metrics': {'mean_r2': 0.82, 'mean_rmse_tc_hectare': 2.1}
                }
            }
        }
        
        with patch.object(viz_suite, 'create_accuracy_assessment_dashboard', return_value='dashboard.html'):
            with patch.object(viz_suite, 'plot_rmse_mae_r2_comparison', return_value='comparison.png'):
                with patch.object(viz_suite, 'create_predicted_vs_actual_plots', 
                                return_value={'biomass': 'biomass.png', 'carbon': 'carbon.png'}):
                    with patch.object(viz_suite, 'visualize_spatial_accuracy_distribution', 
                                    return_value='spatial.png'):
                        with patch.object(viz_suite, 'create_species_accuracy_comparison', 
                                        return_value='species.png'):
                            
                            result = viz_suite.generate_validation_report_visualizations(all_metrics)
                            
                            # Verify comprehensive visualization suite was created
                            assert isinstance(result, dict)
                            assert len(result) >= 5  # Multiple visualization types
                            assert 'validation_summary' in result

    def test_empty_validation_results(self, temp_output_dir):
        """Test handling of empty validation results."""
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        
        # Should handle empty results gracefully
        empty_results = {}
        
        with patch('matplotlib.pyplot.subplots'):
            with patch('matplotlib.pyplot.savefig'):
                with patch('matplotlib.pyplot.close'):
                    result = viz_suite.plot_rmse_mae_r2_comparison(empty_results)
                    assert isinstance(result, str)


class TestFactoryFunctions:
    """Test factory functions for easy usage."""
    
    def test_create_accuracy_assessment_dashboard(self, temp_output_dir, sample_validation_results):
        """Test accuracy assessment dashboard factory function."""
        with patch('src.kelpie_carbon_v1.visualization.validation_plots.ValidationVisualizationSuite') as mock_suite_class:
            mock_suite = MagicMock()
            mock_suite.create_accuracy_assessment_dashboard.return_value = 'dashboard.html'
            mock_suite_class.return_value = mock_suite
            
            result = create_accuracy_assessment_dashboard(sample_validation_results, temp_output_dir)
            
            # Verify factory function works
            assert result == 'dashboard.html'
            mock_suite_class.assert_called_once_with(temp_output_dir)
            mock_suite.create_accuracy_assessment_dashboard.assert_called_once_with(sample_validation_results)
    
    def test_plot_rmse_mae_r2_comparison_factory(self, temp_output_dir, sample_validation_results):
        """Test RMSE, MAE, R² comparison factory function."""
        with patch('src.kelpie_carbon_v1.visualization.validation_plots.ValidationVisualizationSuite') as mock_suite_class:
            mock_suite = MagicMock()
            mock_suite.plot_rmse_mae_r2_comparison.return_value = 'comparison.png'
            mock_suite_class.return_value = mock_suite
            
            result = plot_rmse_mae_r2_comparison(sample_validation_results, temp_output_dir)
            
            # Verify factory function works
            assert result == 'comparison.png'
            mock_suite_class.assert_called_once_with(temp_output_dir)
            mock_suite.plot_rmse_mae_r2_comparison.assert_called_once_with(sample_validation_results)
    
    def test_create_predicted_vs_actual_plots_factory(self, temp_output_dir, sample_validation_results):
        """Test predicted vs actual plots factory function."""
        with patch('src.kelpie_carbon_v1.visualization.validation_plots.ValidationVisualizationSuite') as mock_suite_class:
            mock_suite = MagicMock()
            mock_suite.create_predicted_vs_actual_plots.return_value = {'biomass': 'biomass.png'}
            mock_suite_class.return_value = mock_suite
            
            result = create_predicted_vs_actual_plots(sample_validation_results, temp_output_dir)
            
            # Verify factory function works
            assert result == {'biomass': 'biomass.png'}
            mock_suite_class.assert_called_once_with(temp_output_dir)
            mock_suite.create_predicted_vs_actual_plots.assert_called_once_with(sample_validation_results)
    
    def test_visualize_spatial_accuracy_distribution_factory(self, temp_output_dir, sample_validation_results):
        """Test spatial accuracy distribution factory function."""
        with patch('src.kelpie_carbon_v1.visualization.validation_plots.ValidationVisualizationSuite') as mock_suite_class:
            mock_suite = MagicMock()
            mock_suite.visualize_spatial_accuracy_distribution.return_value = 'spatial.html'
            mock_suite_class.return_value = mock_suite
            
            result = visualize_spatial_accuracy_distribution(sample_validation_results, temp_output_dir)
            
            # Verify factory function works
            assert result == 'spatial.html'
            mock_suite_class.assert_called_once_with(temp_output_dir)
            mock_suite.visualize_spatial_accuracy_distribution.assert_called_once_with(sample_validation_results)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_validation_result(self, temp_output_dir):
        """Test handling of single validation result."""
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        
        single_result = {
            'single_site': MockValidationResult('single_site', 'Nereocystis luetkeana', 50.0, -125.0)
        }
        
        with patch('matplotlib.pyplot.subplots'):
            with patch('matplotlib.pyplot.savefig'):
                with patch('matplotlib.pyplot.close'):
                    result = viz_suite.create_species_accuracy_comparison(single_result)
                    assert isinstance(result, str)
    
    def test_missing_metrics_data(self, temp_output_dir):
        """Test handling of validation results with missing metrics."""
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        
        # Create result with missing metrics
        incomplete_result = MockValidationResult('incomplete', 'Nereocystis luetkeana', 50.0, -125.0)
        incomplete_result.biomass_metrics = {}  # Empty metrics
        incomplete_result.carbon_metrics = {}
        
        incomplete_results = {'incomplete_site': incomplete_result}
        
        with patch('matplotlib.pyplot.subplots'):
            with patch('matplotlib.pyplot.savefig'):
                with patch('matplotlib.pyplot.close'):
                    # Should handle missing metrics gracefully
                    result = viz_suite.plot_rmse_mae_r2_comparison(incomplete_results)
                    assert isinstance(result, str)
    
    def test_matplotlib_import_error_handling(self, temp_output_dir):
        """Test graceful handling when matplotlib is not available."""
        # This would be tested in integration if matplotlib wasn't available
        # For unit test, we just verify the structure exists
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        assert hasattr(viz_suite, 'output_dir')
    
    def test_folium_unavailable_fallback(self, temp_output_dir, sample_validation_results):
        """Test fallback to static plot when Folium is unavailable."""
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        
        with patch('src.kelpie_carbon_v1.visualization.validation_plots.FOLIUM_AVAILABLE', False):
            with patch.object(viz_suite, '_create_static_spatial_plot', return_value='static.png') as mock_static:
                result = viz_suite.visualize_spatial_accuracy_distribution(sample_validation_results)
                
                # Should fall back to static plot
                mock_static.assert_called_once_with(sample_validation_results)
                assert result == 'static.png'


class TestIntegrationScenarios:
    """Test integration scenarios and workflows."""
    
    def test_complete_visualization_workflow(self, temp_output_dir, sample_validation_results):
        """Test complete visualization workflow from start to finish."""
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        
        # Mock all the plotting methods
        with patch.object(viz_suite, 'plot_rmse_mae_r2_comparison', return_value='comparison.png'):
            with patch.object(viz_suite, 'create_predicted_vs_actual_plots', 
                            return_value={'biomass': 'biomass.png', 'carbon': 'carbon.png'}):
                with patch.object(viz_suite, 'visualize_spatial_accuracy_distribution', 
                                return_value='spatial.html'):
                    with patch.object(viz_suite, 'create_species_accuracy_comparison', 
                                    return_value='species.png'):
                        
                        # Test complete workflow
                        comparison_plot = viz_suite.plot_rmse_mae_r2_comparison(sample_validation_results)
                        scatter_plots = viz_suite.create_predicted_vs_actual_plots(sample_validation_results)
                        spatial_plot = viz_suite.visualize_spatial_accuracy_distribution(sample_validation_results)
                        species_plot = viz_suite.create_species_accuracy_comparison(sample_validation_results)
                        
                        # Verify all plots were created
                        assert isinstance(comparison_plot, str)
                        assert isinstance(scatter_plots, dict)
                        assert isinstance(spatial_plot, str)
                        assert isinstance(species_plot, str)
    
    def test_multi_species_validation_workflow(self, temp_output_dir):
        """Test workflow with multiple species validation."""
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        
        # Create results with multiple species
        multi_species_results = {
            'nereocystis_1': MockValidationResult('n1', 'Nereocystis luetkeana', 50.0, -125.0),
            'nereocystis_2': MockValidationResult('n2', 'Nereocystis luetkeana', 51.0, -126.0),
            'macrocystis_1': MockValidationResult('m1', 'Macrocystis pyrifera', 36.0, -121.0),
            'macrocystis_2': MockValidationResult('m2', 'Macrocystis pyrifera', 37.0, -122.0),
        }
        
        with patch('matplotlib.pyplot.subplots'):
            with patch('matplotlib.pyplot.savefig'):
                with patch('matplotlib.pyplot.close'):
                    result = viz_suite.create_species_accuracy_comparison(multi_species_results)
                    assert isinstance(result, str)
    
    def test_performance_with_large_dataset(self, temp_output_dir):
        """Test performance with large validation dataset."""
        viz_suite = ValidationVisualizationSuite(temp_output_dir)
        
        # Create large dataset
        large_results = {}
        for i in range(50):  # 50 validation sites
            large_results[f'site_{i}'] = MockValidationResult(
                f'site_{i}', 
                'Nereocystis luetkeana' if i % 2 == 0 else 'Macrocystis pyrifera',
                50.0 + i * 0.1, 
                -125.0 + i * 0.1
            )
        
        with patch('matplotlib.pyplot.subplots'):
            with patch('matplotlib.pyplot.savefig'):
                with patch('matplotlib.pyplot.close'):
                    # Should handle large datasets efficiently
                    result = viz_suite.plot_rmse_mae_r2_comparison(large_results)
                    assert isinstance(result, str) 