"""
Unit tests for satellite data optimization module.
Tests dual-satellite fusion, cloud masking, and carbon market optimization.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from typing import Dict, Any

# Import the module under test
from src.kelpie_carbon_v1.data.satellite_optimization import (
    SatelliteDataOptimization,
    SatelliteOptimizationConfig,
    ProcessingProvenance,
    create_satellite_optimization,
    optimize_dual_satellite_coverage,
    enhance_cloud_processing
)


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return SatelliteOptimizationConfig(
        temporal_fusion_window_days=5,
        cloud_mask_threshold=0.3,
        quality_weight_s2a=1.0,
        quality_weight_s2b=0.95,
        chunk_size_mb=256,
        parallel_processing=True
    )


@pytest.fixture
def mock_sentinel_dataset():
    """Create mock Sentinel-2 dataset for testing."""
    # Create realistic coordinate and time dimensions
    time = pd.date_range('2023-01-01', periods=10, freq='5D')
    x = np.linspace(-125.5, -125.0, 50)
    y = np.linspace(50.0, 50.5, 50)
    
    # Create mock spectral data
    data_vars = {}
    for band in ['blue', 'green', 'red', 'nir', 're1', 're2', 're3']:
        # Realistic reflectance values (0.0 to 1.0)
        data = np.random.uniform(0.0, 0.8, (len(time), len(y), len(x)))
        data_vars[band] = (['time', 'y', 'x'], data)
    
    coords = {
        'time': time,
        'y': y,
        'x': x
    }
    
    return xr.Dataset(data_vars, coords=coords)


@pytest.fixture
def mock_s2a_dataset(mock_sentinel_dataset):
    """Create mock Sentinel-2A dataset."""
    return mock_sentinel_dataset.copy()


@pytest.fixture
def mock_s2b_dataset(mock_sentinel_dataset):
    """Create mock Sentinel-2B dataset with slightly different timing."""
    dataset = mock_sentinel_dataset.copy()
    # Offset time by 2.5 days to simulate S2B orbit
    dataset['time'] = dataset.time + pd.Timedelta(days=2.5)
    # Add slight variation to simulate sensor differences
    for var in dataset.data_vars:
        dataset[var] = dataset[var] * 0.98  # Slight calibration difference
    return dataset


class TestSatelliteOptimizationConfig:
    """Test SatelliteOptimizationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SatelliteOptimizationConfig()
        
        assert config.temporal_fusion_window_days == 5
        assert config.cloud_mask_threshold == 0.3
        assert config.quality_weight_s2a == 1.0
        assert config.quality_weight_s2b == 0.95
        assert config.max_gap_days == 10
        assert config.interpolation_method == "linear"
        assert config.uncertainty_propagation == True
        assert config.chunk_size_mb == 512
        assert config.parallel_processing == True
        assert config.memory_optimization == True
        assert config.pixel_uncertainty_required == True
        assert config.provenance_tracking == True
        assert config.chain_of_custody == True
        assert config.landsat_integration == False
        assert config.cross_sensor_calibration == True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SatelliteOptimizationConfig(
            temporal_fusion_window_days=7,
            cloud_mask_threshold=0.2,
            quality_weight_s2a=0.9,
            quality_weight_s2b=1.0,
            parallel_processing=False
        )
        
        assert config.temporal_fusion_window_days == 7
        assert config.cloud_mask_threshold == 0.2
        assert config.quality_weight_s2a == 0.9
        assert config.quality_weight_s2b == 1.0
        assert config.parallel_processing == False


class TestProcessingProvenance:
    """Test ProcessingProvenance dataclass."""
    
    def test_provenance_creation(self):
        """Test processing provenance record creation."""
        provenance = ProcessingProvenance(
            processing_timestamp=datetime.now(),
            input_data_sources=['Sentinel-2A', 'Sentinel-2B'],
            processing_steps=[{'step': 'fusion', 'method': 'weighted'}],
            quality_flags={'temporal_improvement': 45.0},
            uncertainty_estimates={'fusion_uncertainty': 0.05},
            software_version='kelpie_carbon_v1.0.0',
            processing_parameters={'config': 'test'}
        )
        
        assert len(provenance.input_data_sources) == 2
        assert 'Sentinel-2A' in provenance.input_data_sources
        assert 'Sentinel-2B' in provenance.input_data_sources
        assert len(provenance.processing_steps) == 1
        assert 'temporal_improvement' in provenance.quality_flags
        assert 'fusion_uncertainty' in provenance.uncertainty_estimates
        assert provenance.software_version == 'kelpie_carbon_v1.0.0'


class TestSatelliteDataOptimization:
    """Test SatelliteDataOptimization class."""
    
    def test_initialization_default_config(self):
        """Test initialization with default configuration."""
        optimizer = SatelliteDataOptimization()
        
        assert isinstance(optimizer.config, SatelliteOptimizationConfig)
        assert optimizer.config.temporal_fusion_window_days == 5
        assert optimizer.processing_history == []
    
    def test_initialization_custom_config(self, sample_config):
        """Test initialization with custom configuration."""
        optimizer = SatelliteDataOptimization(sample_config)
        
        assert optimizer.config == sample_config
        assert optimizer.config.chunk_size_mb == 256
        assert optimizer.processing_history == []
    
    def test_implement_dual_sentinel_fusion(self, sample_config, mock_sentinel_dataset):
        """Test dual Sentinel-2A/2B fusion implementation."""
        optimizer = SatelliteDataOptimization(sample_config)
        
        # Create second dataset with offset timing
        s2b_dataset = mock_sentinel_dataset.copy()
        s2b_dataset['time'] = s2b_dataset.time + pd.Timedelta(days=2.5)
        
        result = optimizer.implement_dual_sentinel_fusion(mock_sentinel_dataset, s2b_dataset)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'fused_dataset' in result
        assert 'fusion_quality' in result
        assert 'temporal_improvement' in result
        assert 'processing_provenance' in result
        
        # Verify temporal improvement
        assert isinstance(result['temporal_improvement'], float)
        
        # Verify provenance
        provenance = result['processing_provenance']
        assert isinstance(provenance, ProcessingProvenance)
    
    def test_create_enhanced_cloud_masking(self, sample_config, mock_sentinel_dataset):
        """Test enhanced cloud masking implementation."""
        optimizer = SatelliteDataOptimization(sample_config)
        
        result = optimizer.create_enhanced_cloud_masking(mock_sentinel_dataset)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'cloud_masked_dataset' in result
        assert 'cloud_mask' in result
        assert 'quality_metrics' in result
        
        # Verify cloud masked dataset
        masked_dataset = result['cloud_masked_dataset']
        assert isinstance(masked_dataset, xr.Dataset)
    
    def test_implement_carbon_market_optimization(self, sample_config):
        """Test carbon market optimization framework."""
        optimizer = SatelliteDataOptimization(sample_config)
        
        result = optimizer.implement_carbon_market_optimization()
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'compliance_standards' in result
        assert 'quality_flag_system' in result
        assert 'chain_of_custody' in result
        
        # Verify compliance standards
        compliance = result['compliance_standards']
        assert compliance['verra_vcs'] == True
        assert compliance['gold_standard'] == True


class TestFactoryFunctions:
    """Test factory functions for easy usage."""
    
    def test_create_satellite_optimization_default(self):
        """Test satellite optimization creation with default config."""
        optimizer = create_satellite_optimization()
        
        assert isinstance(optimizer, SatelliteDataOptimization)
        assert isinstance(optimizer.config, SatelliteOptimizationConfig)
    
    def test_optimize_dual_satellite_coverage(self, mock_sentinel_dataset):
        """Test dual satellite coverage optimization factory function."""
        s2b_dataset = mock_sentinel_dataset.copy()
        s2b_dataset['time'] = s2b_dataset.time + pd.Timedelta(days=2.5)
        
        result = optimize_dual_satellite_coverage(mock_sentinel_dataset, s2b_dataset)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'fused_dataset' in result
        assert 'temporal_improvement' in result


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataset_handling(self, sample_config):
        """Test handling of empty datasets."""
        optimizer = SatelliteDataOptimization(sample_config)
        
        # Create empty dataset
        empty_dataset = xr.Dataset()
        
        # Should handle empty datasets gracefully
        try:
            result = optimizer.create_enhanced_cloud_masking(empty_dataset)
            assert isinstance(result, dict)
        except (KeyError, AttributeError):
            # Expected for empty dataset - this is acceptable
            pass
    
    def test_single_time_step_dataset(self, sample_config):
        """Test handling of dataset with single time step."""
        optimizer = SatelliteDataOptimization(sample_config)
        
        # Create single time step dataset
        time = pd.date_range('2023-01-01', periods=1)
        x = np.linspace(-125.5, -125.0, 10)
        y = np.linspace(50.0, 50.5, 10)
        
        data_vars = {
            'red': (['time', 'y', 'x'], np.random.uniform(0.0, 0.8, (1, len(y), len(x)))),
            'nir': (['time', 'y', 'x'], np.random.uniform(0.0, 0.8, (1, len(y), len(x)))),
            'blue': (['time', 'y', 'x'], np.random.uniform(0.0, 0.8, (1, len(y), len(x))))
        }
        
        single_time_dataset = xr.Dataset(data_vars, coords={'time': time, 'y': y, 'x': x})
        
        # Should handle single time step
        result = optimizer._multi_method_cloud_detection(single_time_dataset)
        assert isinstance(result, dict)
    
    def test_missing_spectral_bands(self, sample_config):
        """Test handling of datasets with missing spectral bands."""
        optimizer = SatelliteDataOptimization(sample_config)
        
        # Create dataset with only some bands
        time = pd.date_range('2023-01-01', periods=5, freq='5D')
        x = np.linspace(-125.5, -125.0, 20)
        y = np.linspace(50.0, 50.5, 20)
        
        # Only red and blue bands (missing nir)
        data_vars = {
            'red': (['time', 'y', 'x'], np.random.uniform(0.0, 0.8, (len(time), len(y), len(x)))),
            'blue': (['time', 'y', 'x'], np.random.uniform(0.0, 0.8, (len(time), len(y), len(x))))
        }
        
        partial_dataset = xr.Dataset(data_vars, coords={'time': time, 'y': y, 'x': x})
        
        # Should handle missing bands gracefully
        result = optimizer._multi_method_cloud_detection(partial_dataset)
        assert isinstance(result, dict)
    
    def test_nan_values_in_data(self, sample_config, mock_sentinel_dataset):
        """Test handling of NaN values in satellite data."""
        optimizer = SatelliteDataOptimization(sample_config)
        
        # Introduce NaN values
        dataset_with_nans = mock_sentinel_dataset.copy()
        # Set 10% of red band values to NaN
        red_data = dataset_with_nans.red.values
        nan_mask = np.random.choice([True, False], size=red_data.shape, p=[0.1, 0.9])
        red_data[nan_mask] = np.nan
        dataset_with_nans['red'] = (['time', 'y', 'x'], red_data)
        
        # Should handle NaN values
        result = optimizer._multi_method_cloud_detection(dataset_with_nans)
        assert isinstance(result, dict)
    
    def test_extreme_spectral_values(self, sample_config):
        """Test handling of extreme spectral values."""
        optimizer = SatelliteDataOptimization(sample_config)
        
        # Create dataset with extreme values
        time = pd.date_range('2023-01-01', periods=3, freq='5D')
        x = np.linspace(-125.5, -125.0, 10)
        y = np.linspace(50.0, 50.5, 10)
        
        data_vars = {
            'red': (['time', 'y', 'x'], np.full((len(time), len(y), len(x)), 1.0)),  # Maximum
            'nir': (['time', 'y', 'x'], np.zeros((len(time), len(y), len(x)))),      # Minimum
            'blue': (['time', 'y', 'x'], np.random.uniform(0.0, 1.0, (len(time), len(y), len(x))))
        }
        
        extreme_dataset = xr.Dataset(data_vars, coords={'time': time, 'y': y, 'x': x})
        
        # Should handle extreme values
        result = optimizer._multi_method_cloud_detection(extreme_dataset)
        assert isinstance(result, dict)
    
    def test_mismatched_dataset_coordinates(self, sample_config):
        """Test handling of datasets with mismatched coordinates."""
        optimizer = SatelliteDataOptimization(sample_config)
        
        # Create datasets with different coordinate systems
        time1 = pd.date_range('2023-01-01', periods=5, freq='5D')
        time2 = pd.date_range('2023-01-15', periods=3, freq='10D')  # Different timing
        
        x1 = np.linspace(-125.5, -125.0, 20)
        x2 = np.linspace(-125.3, -124.8, 15)  # Different spatial extent
        
        y1 = np.linspace(50.0, 50.5, 20)
        y2 = np.linspace(50.1, 50.4, 15)
        
        # Create mismatched datasets
        dataset1 = xr.Dataset({
            'red': (['time', 'y', 'x'], np.random.uniform(0.0, 0.8, (len(time1), len(y1), len(x1))))
        }, coords={'time': time1, 'y': y1, 'x': x1})
        
        dataset2 = xr.Dataset({
            'red': (['time', 'y', 'x'], np.random.uniform(0.0, 0.8, (len(time2), len(y2), len(x2))))
        }, coords={'time': time2, 'y': y2, 'x': x2})
        
        # Should handle coordinate alignment
        result = optimizer._align_temporal_coordinates(dataset1, dataset2)
        assert isinstance(result, dict)
        assert 's2a' in result
        assert 's2b' in result


class TestIntegrationScenarios:
    """Test integration scenarios and workflows."""
    
    def test_complete_optimization_workflow(self, sample_config, mock_s2a_dataset, mock_s2b_dataset):
        """Test complete satellite optimization workflow."""
        optimizer = SatelliteDataOptimization(sample_config)
        
        # Test complete dual-satellite workflow
        fusion_result = optimizer.implement_dual_sentinel_fusion(mock_s2a_dataset, mock_s2b_dataset)
        
        # Test cloud masking on fused result
        cloud_result = optimizer.create_enhanced_cloud_masking(fusion_result['fused_dataset'])
        
        # Test carbon market optimization
        carbon_result = optimizer.implement_carbon_market_optimization()
        
        # Verify all components work together
        assert isinstance(fusion_result, dict)
        assert isinstance(cloud_result, dict)
        assert isinstance(carbon_result, dict)
        
        # Verify data flows through pipeline
        assert 'fused_dataset' in fusion_result
        assert 'cloud_masked_dataset' in cloud_result
        assert 'compliance_standards' in carbon_result
    
    def test_performance_with_realistic_data_size(self, sample_config):
        """Test performance with realistic data sizes."""
        optimizer = SatelliteDataOptimization(sample_config)
        
        # Create larger, more realistic datasets
        time = pd.date_range('2023-01-01', periods=20, freq='5D')
        x = np.linspace(-126.0, -124.0, 100)  # ~200km at this latitude
        y = np.linspace(49.5, 50.5, 100)     # ~100km
        
        large_dataset = xr.Dataset({
            'red': (['time', 'y', 'x'], np.random.uniform(0.0, 0.8, (len(time), len(y), len(x)))),
            'nir': (['time', 'y', 'x'], np.random.uniform(0.0, 0.8, (len(time), len(y), len(x)))),
            'blue': (['time', 'y', 'x'], np.random.uniform(0.0, 0.8, (len(time), len(y), len(x))))
        }, coords={'time': time, 'y': y, 'x': x})
        
        # Should handle larger datasets efficiently
        result = optimizer._multi_method_cloud_detection(large_dataset)
        assert isinstance(result, dict)
    
    def test_multi_temporal_fusion_scenario(self, sample_config):
        """Test multi-temporal fusion scenario."""
        optimizer = SatelliteDataOptimization(sample_config)
        
        # Create datasets with multiple temporal overlaps
        base_time = pd.date_range('2023-01-01', periods=15, freq='5D')
        x = np.linspace(-125.5, -125.0, 30)
        y = np.linspace(50.0, 50.5, 30)
        
        # S2A every 10 days
        s2a_times = base_time[::2]
        s2a_dataset = xr.Dataset({
            'red': (['time', 'y', 'x'], np.random.uniform(0.0, 0.8, (len(s2a_times), len(y), len(x))))
        }, coords={'time': s2a_times, 'y': y, 'x': x})
        
        # S2B offset by 5 days
        s2b_times = base_time[1::2]
        s2b_dataset = xr.Dataset({
            'red': (['time', 'y', 'x'], np.random.uniform(0.0, 0.8, (len(s2b_times), len(y), len(x))))
        }, coords={'time': s2b_times, 'y': y, 'x': x})
        
        # Test temporal alignment
        result = optimizer._align_temporal_coordinates(s2a_dataset, s2b_dataset)
        
        # Should create unified temporal grid
        assert isinstance(result, dict)
        unified_grid = result['unified_time_grid']
        assert len(unified_grid) >= max(len(s2a_times), len(s2b_times))
    
    def test_error_recovery_and_robustness(self, sample_config):
        """Test error recovery and system robustness."""
        optimizer = SatelliteDataOptimization(sample_config)
        
        # Test with various problematic inputs
        problematic_inputs = [
            None,  # None input
            xr.Dataset(),  # Empty dataset
        ]
        
        for problematic_input in problematic_inputs:
            try:
                if problematic_input is not None:
                    result = optimizer._multi_method_cloud_detection(problematic_input)
                    # If no exception, verify reasonable response
                    assert isinstance(result, dict)
            except (KeyError, AttributeError, ValueError):
                # Expected exceptions for problematic inputs
                pass 