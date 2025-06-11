"""Tests for optimization module - Task A2.8: Comprehensive testing.

This module tests the threshold optimization functionality including
adaptive thresholding, environmental condition handling, and real-time optimization.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.kelpie_carbon_v1.optimization import (
    ThresholdOptimizer,
    optimize_detection_pipeline,
    get_optimized_config_for_site
)


class TestThresholdOptimizer:
    """Test the ThresholdOptimizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = ThresholdOptimizer()
        
        # Create sample validation results
        self.sample_validation_results = {
            "validation_timestamp": "2025-06-10T09:24:34.523091",
            "total_sites": 3,
            "successful_validations": 3,
            "configuration": {
                "max_cloud_cover": 30.0,
                "buffer_km": 2.0,
                "date_range_days": 30,
                "min_detection_threshold": 0.01,
                "max_processing_time": 120
            },
            "skema_configuration": {
                "apply_waf": True,
                "combine_with_ndre": True,
                "detection_combination": "union",
                "apply_morphology": True,
                "min_kelp_cluster_size": 5,
                "ndre_threshold": 0.0,
                "require_water_context": False
            },
            "results": [
                {
                    "site_name": "Broughton Archipelago",
                    "coordinates": {"lat": 50.0833, "lng": -126.1667},
                    "species": "Nereocystis luetkeana",
                    "expected_detection_rate": 0.15,
                    "actual_detection_rate": 0.97745,
                    "cloud_cover": 0.0,
                    "acquisition_date": "2025-06-10",
                    "processing_time": 1.278597,
                    "success": True,
                    "error_message": None
                },
                {
                    "site_name": "Saanich Inlet",
                    "coordinates": {"lat": 48.583, "lng": -123.5},
                    "species": "Mixed Nereocystis + Macrocystis",
                    "expected_detection_rate": 0.12,
                    "actual_detection_rate": 0.97785,
                    "cloud_cover": 0.0,
                    "acquisition_date": "2025-06-10",
                    "processing_time": 1.186234,
                    "success": True,
                    "error_message": None
                },
                {
                    "site_name": "Monterey Bay",
                    "coordinates": {"lat": 36.8, "lng": -121.9},
                    "species": "Macrocystis pyrifera",
                    "expected_detection_rate": 0.10,
                    "actual_detection_rate": 0.9764,
                    "cloud_cover": 0.0,
                    "acquisition_date": "2025-06-10",
                    "processing_time": 0.940151,
                    "success": True,
                    "error_message": None
                }
            ]
        }

    def test_load_validation_results(self):
        """Test loading validation results from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_validation_results, f)
            temp_file = f.name
        
        try:
            results = self.optimizer.load_validation_results(temp_file)
            
            assert results['total_sites'] == 3
            assert results['successful_validations'] == 3
            assert len(results['results']) == 3
            assert len(self.optimizer.validation_history) == 1
            
        finally:
            Path(temp_file).unlink()

    def test_analyze_detection_rates(self):
        """Test detection rate analysis."""
        analysis = self.optimizer.analyze_detection_rates(self.sample_validation_results)
        
        assert analysis['sites_analyzed'] == 3
        assert analysis['mean_detection_rate'] > 0.97  # High detection rate
        assert analysis['mean_expected_rate'] < 0.15   # Low expected rate
        assert analysis['over_detection_ratio'] > 7.0  # Severe over-detection
        assert analysis['accuracy_score'] < 0.2        # Poor accuracy

    def test_calculate_optimal_thresholds_over_detection(self):
        """Test threshold calculation for over-detection scenario."""
        optimal = self.optimizer.calculate_optimal_thresholds(self.sample_validation_results)
        
        # Should increase thresholds for severe over-detection
        assert optimal['ndre_threshold'] >= 0.05
        assert optimal['kelp_fai_threshold'] >= 0.02
        assert optimal['min_detection_threshold'] >= 0.02

    def test_create_adaptive_config_kelp_farm(self):
        """Test adaptive configuration for kelp farm sites."""
        config = self.optimizer.create_adaptive_config(
            'kelp_farm',
            {'cloud_cover': 0.15, 'turbidity': 'medium'}
        )
        
        assert 'ndre_threshold' in config
        assert 'kelp_fai_threshold' in config
        assert 'min_detection_threshold' in config
        assert config['apply_morphology'] is True
        assert config['require_water_context'] is True
        assert config['min_kelp_cluster_size'] >= 5

    def test_create_adaptive_config_open_ocean(self):
        """Test adaptive configuration for open ocean sites."""
        config = self.optimizer.create_adaptive_config(
            'open_ocean',
            {'cloud_cover': 0.20, 'turbidity': 'low'}
        )
        
        # Open ocean should have higher thresholds
        assert config['ndre_threshold'] > 0.10
        assert config['kelp_fai_threshold'] > 0.04
        assert config['min_kelp_cluster_size'] >= 10

    def test_create_adaptive_config_high_cloud_cover(self):
        """Test adaptive configuration adjustment for high cloud cover."""
        base_config = self.optimizer.create_adaptive_config(
            'kelp_farm',
            {'cloud_cover': 0.10, 'turbidity': 'low'}
        )
        
        cloudy_config = self.optimizer.create_adaptive_config(
            'kelp_farm',
            {'cloud_cover': 0.40, 'turbidity': 'low'}
        )
        
        # High cloud cover should increase thresholds
        assert cloudy_config['ndre_threshold'] > base_config['ndre_threshold']
        assert cloudy_config['kelp_fai_threshold'] > base_config['kelp_fai_threshold']

    def test_create_adaptive_config_turbidity_effects(self):
        """Test adaptive configuration adjustment for different turbidity levels."""
        low_turbidity = self.optimizer.create_adaptive_config(
            'coastal',
            {'cloud_cover': 0.15, 'turbidity': 'low'}
        )
        
        high_turbidity = self.optimizer.create_adaptive_config(
            'coastal',
            {'cloud_cover': 0.15, 'turbidity': 'high'}
        )
        
        # High turbidity should affect thresholds and cluster size
        assert high_turbidity['kelp_fai_threshold'] > low_turbidity['kelp_fai_threshold']
        assert high_turbidity['min_kelp_cluster_size'] > low_turbidity['min_kelp_cluster_size']

    def test_optimize_for_real_time(self):
        """Test real-time optimization configuration."""
        config = self.optimizer.optimize_for_real_time(15.0)
        
        assert config['apply_waf'] is True
        assert config['waf_fast_mode'] is True
        assert config['apply_morphology'] is False  # Skip for speed
        assert config['detection_combination'] == 'intersection'  # Faster
        assert config['min_kelp_cluster_size'] <= 5  # Smaller for speed
        assert 'max_processing_resolution' in config

    def test_generate_recommendations_critical(self):
        """Test recommendation generation for critical over-detection."""
        analysis = {
            'over_detection_ratio': 8.0,
            'accuracy_score': 0.1,
            'mean_detection_rate': 0.98,
            'mean_expected_rate': 0.12
        }
        
        recommendations = self.optimizer._generate_recommendations(analysis)
        
        assert any('CRITICAL' in rec for rec in recommendations)
        assert any('threshold' in rec.lower() for rec in recommendations)
        assert any('comprehensive' in rec.lower() for rec in recommendations)

    def test_generate_recommendations_good_accuracy(self):
        """Test recommendation generation for good accuracy."""
        analysis = {
            'over_detection_ratio': 1.1,
            'accuracy_score': 0.85,
            'mean_detection_rate': 0.13,
            'mean_expected_rate': 0.12
        }
        
        recommendations = self.optimizer._generate_recommendations(analysis)
        
        assert any('Good accuracy' in rec for rec in recommendations)

    def test_save_optimization_results(self):
        """Test saving optimization results to JSON file."""
        self.optimizer.optimization_results = {
            'test_data': 'test_value',
            'analysis': {'accuracy': 0.8}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test_optimization.json'
            self.optimizer.save_optimization_results(str(output_path))
            
            assert output_path.exists()
            
            with open(output_path) as f:
                saved_results = json.load(f)
            
            assert saved_results['test_data'] == 'test_value'
            assert 'timestamp' in saved_results
            assert 'optimization_type' in saved_results

    def test_run_comprehensive_optimization(self):
        """Test comprehensive optimization workflow."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_validation_results, f)
            temp_file = f.name
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                results = self.optimizer.run_comprehensive_optimization(
                    temp_file, temp_dir
                )
                
                assert 'current_analysis' in results
                assert 'optimized_scenarios' in results
                assert 'recommendations' in results
                
                # Check that all scenarios are generated
                scenarios = results['optimized_scenarios']
                assert 'optimal_accuracy' in scenarios
                assert 'kelp_farm_tuned' in scenarios
                assert 'open_ocean_tuned' in scenarios
                assert 'coastal_tuned' in scenarios
                assert 'real_time_optimized' in scenarios
                
            finally:
                Path(temp_file).unlink()


class TestOptimizationUtilityFunctions:
    """Test utility functions for optimization."""

    def test_optimize_detection_pipeline(self):
        """Test main optimization pipeline function."""
        sample_results = {
            "total_sites": 2,
            "successful_validations": 2,
            "configuration": {"min_detection_threshold": 0.01},
            "skema_configuration": {"ndre_threshold": 0.0},
            "results": [
                {
                    "expected_detection_rate": 0.15,
                    "actual_detection_rate": 0.50,
                    "success": True
                },
                {
                    "expected_detection_rate": 0.12,
                    "actual_detection_rate": 0.45,
                    "success": True
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_results, f)
            temp_file = f.name
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                results = optimize_detection_pipeline(temp_file, temp_dir)
                
                assert 'current_analysis' in results
                assert 'optimized_scenarios' in results
                assert results['current_analysis']['over_detection_ratio'] > 1.0
                
            finally:
                Path(temp_file).unlink()

    def test_get_optimized_config_for_site(self):
        """Test getting optimized configuration for specific site."""
        config = get_optimized_config_for_site(
            'kelp_farm',
            {'cloud_cover': 0.20, 'turbidity': 'medium'}
        )
        
        assert 'ndre_threshold' in config
        assert 'kelp_fai_threshold' in config
        assert 'apply_morphology' in config
        assert isinstance(config['ndre_threshold'], float)
        assert config['ndre_threshold'] > 0.0


class TestOptimizationEdgeCases:
    """Test edge cases and error handling in optimization."""

    def test_empty_validation_results(self):
        """Test handling of empty validation results."""
        optimizer = ThresholdOptimizer()
        empty_results = {
            "total_sites": 0,
            "successful_validations": 0,
            "configuration": {"min_detection_threshold": 0.01},
            "skema_configuration": {"ndre_threshold": 0.0},
            "results": []
        }
        
        analysis = optimizer.analyze_detection_rates(empty_results)
        
        assert analysis['sites_analyzed'] == 0
        assert analysis['mean_detection_rate'] == 0.0
        assert analysis['mean_expected_rate'] == 0.0

    def test_under_detection_scenario(self):
        """Test optimization for under-detection scenario."""
        optimizer = ThresholdOptimizer()
        under_detection_results = {
            "configuration": {"min_detection_threshold": 0.05},
            "skema_configuration": {"ndre_threshold": 0.15},
            "results": [
                {
                    "expected_detection_rate": 0.15,
                    "actual_detection_rate": 0.05,  # Under-detecting
                    "success": True
                }
            ]
        }
        
        optimal = optimizer.calculate_optimal_thresholds(under_detection_results)
        
        # Should decrease thresholds for under-detection
        assert optimal['ndre_threshold'] < 0.15
        assert optimal['min_detection_threshold'] < 0.05

    def test_invalid_site_type(self):
        """Test handling of invalid site type."""
        optimizer = ThresholdOptimizer()
        
        # Should default to coastal configuration
        config = optimizer.create_adaptive_config(
            'invalid_site_type',
            {'cloud_cover': 0.15, 'turbidity': 'medium'}
        )
        
        assert 'ndre_threshold' in config
        assert config['require_water_context'] is False  # Coastal default

    def test_missing_environmental_conditions(self):
        """Test handling of missing environmental condition parameters."""
        optimizer = ThresholdOptimizer()
        
        # Should use defaults for missing parameters
        config = optimizer.create_adaptive_config(
            'kelp_farm',
            {}  # Empty conditions
        )
        
        assert 'ndre_threshold' in config
        assert 'kelp_fai_threshold' in config

    def test_extreme_processing_time_target(self):
        """Test real-time optimization with extreme processing time targets."""
        optimizer = ThresholdOptimizer()
        
        # Very fast target should create very optimized config
        fast_config = optimizer.optimize_for_real_time(5.0)
        
        assert fast_config['apply_morphology'] is False
        assert fast_config['min_kelp_cluster_size'] <= 5
        
        # Very slow target should still be reasonable
        slow_config = optimizer.optimize_for_real_time(300.0)
        
        assert 'ndre_threshold' in slow_config


class TestOptimizationPerformance:
    """Test performance aspects of optimization."""

    def test_optimization_performance_benchmarks(self):
        """Test that optimization completes within reasonable time."""
        import time
        
        optimizer = ThresholdOptimizer()
        
        # Create larger validation dataset
        large_results = {
            "total_sites": 10,
            "successful_validations": 10,
            "configuration": {"min_detection_threshold": 0.01},
            "skema_configuration": {"ndre_threshold": 0.0},
            "results": [
                {
                    "expected_detection_rate": 0.15,
                    "actual_detection_rate": 0.50,
                    "success": True
                } for _ in range(10)
            ]
        }
        
        start_time = time.time()
        analysis = optimizer.analyze_detection_rates(large_results)
        optimal = optimizer.calculate_optimal_thresholds(large_results)
        end_time = time.time()
        
        # Should complete optimization quickly
        assert end_time - start_time < 1.0  # Less than 1 second
        assert analysis['sites_analyzed'] == 10

    def test_memory_efficiency(self):
        """Test memory efficiency of optimization operations."""
        optimizer = ThresholdOptimizer()
        
        # Test with multiple validation runs
        for i in range(5):
            results = {
                "total_sites": 3,
                "successful_validations": 3,
                "configuration": {"min_detection_threshold": 0.01},
                "skema_configuration": {"ndre_threshold": 0.0},
                "results": [
                    {
                        "expected_detection_rate": 0.15,
                        "actual_detection_rate": 0.50,
                        "success": True
                    } for _ in range(3)
                ]
            }
            
            analysis = optimizer.analyze_detection_rates(results)
            optimizer.validation_history.append(results)
        
        # Should maintain reasonable memory usage
        assert len(optimizer.validation_history) == 5
        assert isinstance(analysis, dict)


@pytest.mark.integration
class TestOptimizationIntegration:
    """Integration tests for optimization with real components."""

    def test_integration_with_validation_pipeline(self):
        """Test optimization integration with validation pipeline."""
        # This would test with actual validation results if available
        # For now, test with realistic mock data
        
        realistic_results = {
            "validation_timestamp": "2025-06-10T09:24:34.523091",
            "total_sites": 3,
            "successful_validations": 3,
            "configuration": {
                "max_cloud_cover": 30.0,
                "min_detection_threshold": 0.01
            },
            "skema_configuration": {
                "ndre_threshold": 0.0,
                "apply_waf": True,
                "combine_with_ndre": True
            },
            "results": [
                {
                    "site_name": "Test Kelp Farm",
                    "expected_detection_rate": 0.15,
                    "actual_detection_rate": 0.90,
                    "processing_time": 1.5,
                    "success": True
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(realistic_results, f)
            temp_file = f.name
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                results = optimize_detection_pipeline(temp_file, temp_dir)
                
                # Verify complete optimization pipeline
                assert 'current_analysis' in results
                assert 'optimized_scenarios' in results
                assert 'recommendations' in results
                
                # Check that recommendations are appropriate
                analysis = results['current_analysis']
                recommendations = results['recommendations']
                
                if analysis['over_detection_ratio'] > 2.0:
                    assert any('over-detection' in rec.lower() for rec in recommendations)
                
            finally:
                Path(temp_file).unlink()

    def test_optimization_scenario_validation(self):
        """Test that optimized scenarios produce valid configurations."""
        optimizer = ThresholdOptimizer()
        
        scenarios = [
            ('kelp_farm', {'cloud_cover': 0.1, 'turbidity': 'low'}),
            ('open_ocean', {'cloud_cover': 0.2, 'turbidity': 'medium'}),
            ('coastal', {'cloud_cover': 0.3, 'turbidity': 'high'}),
        ]
        
        for site_type, conditions in scenarios:
            config = optimizer.create_adaptive_config(site_type, conditions)
            
            # Validate configuration structure and values
            assert 0.0 <= config['ndre_threshold'] <= 1.0
            assert 0.0 <= config['kelp_fai_threshold'] <= 1.0
            assert 0.0 <= config['min_detection_threshold'] <= 1.0
            assert config['min_kelp_cluster_size'] >= 1
            assert isinstance(config['apply_morphology'], bool)
            assert isinstance(config['require_water_context'], bool) 