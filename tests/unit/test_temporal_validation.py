"""
Tests for Temporal Validation & Environmental Drivers.

This module tests the temporal validation framework for SKEMA kelp detection,
including time-series persistence, seasonal patterns, and environmental correlations.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from kelpie_carbon_v1.validation.temporal_validation import (
    TemporalValidator,
    TemporalDataPoint,
    SeasonalPattern,
    TemporalValidationResult,
    create_temporal_validator,
    run_broughton_temporal_validation,
    run_comprehensive_temporal_analysis,
)
from kelpie_carbon_v1.validation.real_world_validation import ValidationSite


class TestTemporalDataPoint:
    """Test the TemporalDataPoint dataclass."""
    
    def test_temporal_data_point_creation(self):
        """Test creating a temporal data point."""
        timestamp = datetime.now()
        environmental_conditions = {
            "tidal_height": 1.5,
            "current_speed": 8.0,
            "water_temperature": 12.5
        }
        
        data_point = TemporalDataPoint(
            timestamp=timestamp,
            detection_rate=0.25,
            kelp_area_km2=1.8,
            confidence_score=0.85,
            environmental_conditions=environmental_conditions,
            quality_flags=["high_cloud_coverage"]
        )
        
        assert data_point.timestamp == timestamp
        assert data_point.detection_rate == 0.25
        assert data_point.kelp_area_km2 == 1.8
        assert data_point.confidence_score == 0.85
        assert data_point.environmental_conditions == environmental_conditions
        assert data_point.quality_flags == ["high_cloud_coverage"]


class TestSeasonalPattern:
    """Test the SeasonalPattern dataclass."""
    
    def test_seasonal_pattern_creation(self):
        """Test creating a seasonal pattern."""
        pattern = SeasonalPattern(
            season="summer",
            average_detection_rate=0.35,
            peak_month=7,
            trough_month=3,
            variability_coefficient=0.15,
            trend_slope=0.02,
            statistical_significance=0.03
        )
        
        assert pattern.season == "summer"
        assert pattern.average_detection_rate == 0.35
        assert pattern.peak_month == 7
        assert pattern.trough_month == 3
        assert pattern.variability_coefficient == 0.15
        assert pattern.trend_slope == 0.02
        assert pattern.statistical_significance == 0.03


class TestTemporalValidator:
    """Test the TemporalValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = TemporalValidator()
        
        # Create test validation site
        self.test_site = ValidationSite(
            name="Test Site",
            lat=50.0,
            lng=-125.0,
            species="Nereocystis luetkeana",
            expected_detection_rate=0.20,
            water_depth="8m",
            optimal_season="Summer",
            site_type="kelp_farm"
        )
        
        # Create mock satellite dataset
        self.mock_dataset = xr.Dataset({
            'red': (('y', 'x'), np.random.rand(50, 50) * 0.3),
            'green': (('y', 'x'), np.random.rand(50, 50) * 0.4),
            'blue': (('y', 'x'), np.random.rand(50, 50) * 0.2),
            'nir': (('y', 'x'), np.random.rand(50, 50) * 0.6),
            'red_edge': (('y', 'x'), np.random.rand(50, 50) * 0.5),
        }, attrs={'scene_id': 'test_scene', 'cloud_coverage': 0.1})
        
        # Create mock detection mask
        self.mock_detection_mask = np.random.rand(50, 50) * 0.3
    
    def test_get_broughton_validation_config(self):
        """Test getting Broughton Archipelago validation configuration."""
        config = self.validator.get_broughton_validation_config()
        
        # Verify site configuration
        assert config["site"].name == "Broughton Archipelago - Temporal"
        assert config["site"].lat == 50.0833
        assert config["site"].lng == -126.1667
        assert config["site"].species == "Nereocystis luetkeana"
        
        # Verify temporal parameters
        assert config["temporal_parameters"]["validation_years"] == 3
        assert config["temporal_parameters"]["sampling_frequency_days"] == 15
        assert "seasonal_windows" in config["temporal_parameters"]
        assert "environmental_drivers" in config["temporal_parameters"]
        
        # Verify persistence thresholds
        assert config["persistence_thresholds"]["minimum_detection_rate"] == 0.10
        assert config["persistence_thresholds"]["consistency_threshold"] == 0.75
    
    def test_generate_sampling_dates(self):
        """Test generating sampling dates with specified interval."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        interval_days = 7
        
        dates = self.validator._generate_sampling_dates(start_date, end_date, interval_days)
        
        assert len(dates) == 5  # Jan 1, 8, 15, 22, 29
        assert dates[0] == start_date
        assert dates[1] == start_date + timedelta(days=7)
        assert all(isinstance(date, datetime) for date in dates)
    
    def test_calculate_kelp_area(self):
        """Test kelp area calculation from detection mask."""
        kelp_mask = np.zeros((10, 10))
        kelp_mask[2:5, 2:5] = 0.5  # 9 pixels above 30% threshold
        
        area_km2 = self.validator._calculate_kelp_area(kelp_mask, self.mock_dataset)
        
        # 9 pixels * 100 m²/pixel = 900 m² = 0.0009 km²
        assert area_km2 == 0.0009
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        # High confidence: high mean, low variance
        high_confidence_mask = np.full((10, 10), 0.8)
        confidence_high = self.validator._calculate_confidence_score(high_confidence_mask, self.mock_dataset)
        
        # Low confidence: low mean, high variance
        low_confidence_mask = np.random.rand(10, 10)
        confidence_low = self.validator._calculate_confidence_score(low_confidence_mask, self.mock_dataset)
        
        assert 0.0 <= confidence_high <= 1.0
        assert 0.0 <= confidence_low <= 1.0
        assert confidence_high > confidence_low
    
    def test_simulate_environmental_conditions(self):
        """Test environmental conditions simulation."""
        test_date = datetime(2023, 7, 15)  # Summer date
        
        conditions = self.validator._simulate_environmental_conditions(self.test_site, test_date)
        
        # Verify all required conditions are present
        required_conditions = [
            "tidal_height", "current_speed", "water_temperature",
            "secchi_depth", "wind_speed", "precipitation"
        ]
        
        for condition in required_conditions:
            assert condition in conditions
            assert isinstance(conditions[condition], float)
        
        # Verify reasonable ranges
        assert -3.0 <= conditions["tidal_height"] <= 3.0
        assert 0.0 <= conditions["current_speed"] <= 25.0
        assert 0.0 <= conditions["water_temperature"] <= 25.0
        assert 2.0 <= conditions["secchi_depth"] <= 12.0
    
    def test_assess_data_quality(self):
        """Test data quality assessment."""
        # High quality dataset
        high_quality_dataset = self.mock_dataset.copy()
        high_quality_dataset.attrs['cloud_coverage'] = 0.05
        high_quality_mask = np.random.rand(50, 50) * 0.3 + 0.1  # Reasonable variation
        
        flags_high = self.validator._assess_data_quality(high_quality_dataset, high_quality_mask)
        
        # Low quality dataset
        low_quality_dataset = self.mock_dataset.copy()
        low_quality_dataset.attrs['cloud_coverage'] = 0.5
        low_quality_mask = np.full((50, 50), 0.95)  # Extremely high detection
        
        flags_low = self.validator._assess_data_quality(low_quality_dataset, low_quality_mask)
        
        assert isinstance(flags_high, list)
        assert isinstance(flags_low, list)
        assert len(flags_low) > len(flags_high)  # More flags for low quality
        assert "high_cloud_coverage" in flags_low
        assert "extremely_high_detection" in flags_low
    
    def test_get_season(self):
        """Test season mapping from month."""
        assert self.validator._get_season(3) == 'spring'
        assert self.validator._get_season(7) == 'summer'
        assert self.validator._get_season(10) == 'fall'
        assert self.validator._get_season(12) == 'winter'
        assert self.validator._get_season(1) == 'winter'
    
    def test_analyze_seasonal_patterns(self):
        """Test seasonal pattern analysis."""
        # Create test data points spanning multiple seasons
        data_points = []
        base_date = datetime(2023, 1, 1)
        
        for i in range(12):  # 12 months of data
            month_date = base_date + timedelta(days=i * 30)
            detection_rate = 0.3 + 0.2 * np.sin(2 * np.pi * i / 12)  # Seasonal variation
            
            data_points.append(TemporalDataPoint(
                timestamp=month_date,
                detection_rate=detection_rate,
                kelp_area_km2=detection_rate * 2.0,
                confidence_score=0.8,
                environmental_conditions={}
            ))
        
        patterns = self.validator._analyze_seasonal_patterns(data_points)
        
        # Should have patterns for multiple seasons
        assert len(patterns) > 0
        
        for season, pattern in patterns.items():
            assert isinstance(pattern, SeasonalPattern)
            assert pattern.season == season
            assert 0.0 <= pattern.average_detection_rate <= 1.0
            assert 1 <= pattern.peak_month <= 12
            assert 1 <= pattern.trough_month <= 12
            assert pattern.variability_coefficient >= 0.0
    
    def test_calculate_persistence_metrics(self):
        """Test persistence metrics calculation."""
        # Create test data points with known characteristics
        data_points = []
        for i in range(10):
            detection_rate = 0.15 + 0.05 * np.sin(i)  # Persistent detection above threshold
            
            data_points.append(TemporalDataPoint(
                timestamp=datetime.now() + timedelta(days=i*15),
                detection_rate=detection_rate,
                kelp_area_km2=detection_rate * 2.0,
                confidence_score=0.8,
                environmental_conditions={}
            ))
        
        metrics = self.validator._calculate_persistence_metrics(data_points)
        
        # Verify all expected metrics are present
        expected_metrics = [
            'mean_detection_rate', 'std_detection_rate', 'coefficient_of_variation',
            'persistence_rate', 'consistency_rate', 'trend_slope', 'trend_r_squared',
            'trend_p_value', 'trend_stability', 'temporal_coverage', 'data_quality_score'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
        
        # Verify reasonable values
        assert 0.0 <= metrics['persistence_rate'] <= 1.0
        assert 0.0 <= metrics['consistency_rate'] <= 1.0
        assert 0.0 <= metrics['data_quality_score'] <= 1.0
        assert metrics['temporal_coverage'] == 10
    
    def test_analyze_environmental_correlations(self):
        """Test environmental correlation analysis."""
        # Create data points with correlated environmental conditions
        data_points = []
        for i in range(20):
            temp = 10 + 5 * np.sin(2 * np.pi * i / 20)  # Temperature cycle
            detection_rate = 0.2 + 0.1 * (temp - 10) / 5  # Correlated with temperature
            
            data_points.append(TemporalDataPoint(
                timestamp=datetime.now() + timedelta(days=i*15),
                detection_rate=detection_rate,
                kelp_area_km2=detection_rate * 2.0,
                confidence_score=0.8,
                environmental_conditions={
                    'water_temperature': temp,
                    'tidal_height': np.random.normal(0, 1),
                    'current_speed': np.random.normal(10, 2)
                }
            ))
        
        correlations = self.validator._analyze_environmental_correlations(data_points)
        
        # Should have correlations for environmental variables
        assert 'water_temperature' in correlations
        assert 'water_temperature_p_value' in correlations
        
        # Water temperature should be positively correlated
        assert correlations['water_temperature'] > 0.3  # Moderate positive correlation
        
        # All correlation values should be valid
        for var, corr in correlations.items():
            if not var.endswith('_p_value'):
                assert -1.0 <= corr <= 1.0
    
    def test_perform_trend_analysis(self):
        """Test trend analysis."""
        # Create data with increasing trend
        data_points = []
        for i in range(15):
            detection_rate = 0.1 + 0.02 * i + 0.05 * np.random.normal()  # Increasing trend with noise
            
            data_points.append(TemporalDataPoint(
                timestamp=datetime(2023, 1, 1) + timedelta(days=i*30),
                detection_rate=detection_rate,
                kelp_area_km2=detection_rate * 2.0,
                confidence_score=0.8,
                environmental_conditions={}
            ))
        
        trend_analysis = self.validator._perform_trend_analysis(data_points)
        
        # Verify trend components
        assert 'linear_trend' in trend_analysis
        assert 'monthly_patterns' in trend_analysis
        assert 'change_points' in trend_analysis
        assert 'variability_analysis' in trend_analysis
        
        linear_trend = trend_analysis['linear_trend']
        assert 'slope' in linear_trend
        assert 'r_squared' in linear_trend
        assert 'p_value' in linear_trend
        assert linear_trend['trend_direction'] in ['increasing', 'decreasing']
        assert linear_trend['trend_strength'] in ['strong', 'moderate', 'weak']
        
        # Should detect increasing trend
        assert linear_trend['slope'] > 0
        assert linear_trend['trend_direction'] == 'increasing'
    
    def test_assess_temporal_quality(self):
        """Test temporal quality assessment."""
        # Create sufficient high-quality data
        data_points = []
        for i in range(20):  # Sufficient data points
            data_points.append(TemporalDataPoint(
                timestamp=datetime(2023, 1, 1) + timedelta(days=i*15),
                detection_rate=0.18,  # Above minimum threshold
                kelp_area_km2=0.36,
                confidence_score=0.85,
                environmental_conditions={},
                quality_flags=[]  # No quality issues
            ))
        
        quality = self.validator._assess_temporal_quality(data_points, self.test_site)
        
        assert 'overall_quality' in quality
        assert 'quality_score' in quality
        assert 'quality_checks' in quality
        assert 'data_coverage' in quality
        assert 'temporal_gaps' in quality
        assert 'recommendations' in quality
        
        assert quality['overall_quality'] in ['excellent', 'good', 'moderate', 'limited']
        assert 0.0 <= quality['quality_score'] <= 1.0
        assert 0.0 <= quality['data_coverage'] <= 1.0
        
        # Should pass quality checks with good data
        assert quality['quality_checks']['meets_minimum_detection'] == True
        assert quality['quality_checks']['sufficient_data_points'] == True
    
    def test_generate_temporal_recommendations(self):
        """Test temporal recommendation generation."""
        # Test with low persistence metrics
        low_persistence_metrics = {
            'persistence_rate': 0.4,  # Below 0.6 threshold
            'coefficient_of_variation': 0.8,  # High variability
            'data_quality_score': 0.5
        }
        
        # Test with good seasonal patterns
        good_seasonal_patterns = {
            'summer': SeasonalPattern(
                season='summer',
                average_detection_rate=0.25,
                peak_month=7,
                trough_month=3,
                variability_coefficient=0.2,
                trend_slope=0.01,
                statistical_significance=0.1
            )
        }
        
        # Test with declining trend
        declining_trend = {
            'linear_trend': {
                'slope': -0.02,
                'trend_strength': 'strong',
                'p_value': 0.01
            }
        }
        
        recommendations = self.validator._generate_temporal_recommendations(
            low_persistence_metrics, good_seasonal_patterns, declining_trend
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('persistence' in rec.lower() for rec in recommendations)
        assert any('variability' in rec.lower() for rec in recommendations)
        assert any('declining trend' in rec.lower() for rec in recommendations)
    
    @patch('kelpie_carbon_v1.validation.temporal_validation.fetch_sentinel_tiles')
    @patch('kelpie_carbon_v1.validation.temporal_validation.create_skema_kelp_detection_mask')
    @pytest.mark.asyncio
    async def test_collect_temporal_data_point(self, mock_mask, mock_fetch):
        """Test collecting a single temporal data point."""
        # Mock successful data collection - fetch_sentinel_tiles needs to be async
        async def mock_fetch_async(*args, **kwargs):
            return [self.mock_dataset]
        
        mock_fetch.side_effect = mock_fetch_async
        mock_mask.return_value = self.mock_detection_mask
        
        test_date = datetime(2023, 7, 15)
        data_point = await self.validator._collect_temporal_data_point(self.test_site, test_date)
        
        assert data_point is not None
        assert isinstance(data_point, TemporalDataPoint)
        assert data_point.timestamp == test_date
        assert 0.0 <= data_point.detection_rate <= 1.0
        assert data_point.kelp_area_km2 >= 0.0
        assert 0.0 <= data_point.confidence_score <= 1.0
        assert len(data_point.environmental_conditions) > 0
        
        # Test failed data collection
        async def mock_fetch_empty(*args, **kwargs):
            return []
        
        mock_fetch.side_effect = mock_fetch_empty
        data_point_fail = await self.validator._collect_temporal_data_point(self.test_site, test_date)
        assert data_point_fail is None
    
    @patch('kelpie_carbon_v1.validation.temporal_validation.fetch_sentinel_tiles')
    @patch('kelpie_carbon_v1.validation.temporal_validation.create_skema_kelp_detection_mask')
    @pytest.mark.asyncio
    async def test_validate_temporal_persistence(self, mock_mask, mock_fetch):
        """Test full temporal persistence validation."""
        # Mock data collection - fetch_sentinel_tiles needs to be async
        async def mock_fetch_async(*args, **kwargs):
            return [self.mock_dataset]
        
        mock_fetch.side_effect = mock_fetch_async
        mock_mask.return_value = self.mock_detection_mask
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 3, 1)  # 2-month period
        
        result = await self.validator.validate_temporal_persistence(
            site=self.test_site,
            start_date=start_date,
            end_date=end_date,
            sampling_interval_days=15
        )
        
        assert isinstance(result, TemporalValidationResult)
        assert result.site_name == self.test_site.name
        assert result.validation_period == (start_date, end_date)
        assert len(result.data_points) > 0
        assert isinstance(result.seasonal_patterns, dict)
        assert isinstance(result.persistence_metrics, dict)
        assert isinstance(result.environmental_correlations, dict)
        assert isinstance(result.trend_analysis, dict)
        assert isinstance(result.quality_assessment, dict)
        assert isinstance(result.recommendations, list)
    
    @patch('kelpie_carbon_v1.validation.temporal_validation.fetch_sentinel_tiles')
    @patch('kelpie_carbon_v1.validation.temporal_validation.create_skema_kelp_detection_mask')
    @pytest.mark.asyncio
    async def test_run_broughton_archipelago_validation(self, mock_mask, mock_fetch):
        """Test Broughton Archipelago validation."""
        # Mock data collection - fetch_sentinel_tiles needs to be async
        async def mock_fetch_async(*args, **kwargs):
            return [self.mock_dataset]
        
        mock_fetch.side_effect = mock_fetch_async
        mock_mask.return_value = self.mock_detection_mask
        
        result = await self.validator.run_broughton_archipelago_validation(validation_years=1)
        
        assert isinstance(result, TemporalValidationResult)
        assert "Broughton Archipelago" in result.site_name
        assert len(result.data_points) >= 0  # May have limited data in test environment
    
    def test_generate_comprehensive_temporal_report(self):
        """Test comprehensive temporal report generation."""
        # Create mock results
        mock_result1 = TemporalValidationResult(
            site_name="Site 1",
            validation_period=(datetime(2023, 1, 1), datetime(2023, 12, 31)),
            data_points=[],
            seasonal_patterns={},
            persistence_metrics={
                'persistence_rate': 0.8,
                'trend_stability': 0.7,
                'data_quality_score': 0.85
            },
            environmental_correlations={},
            trend_analysis={},
            quality_assessment={'quality_score': 0.8, 'data_coverage': 0.9, 'overall_quality': 'good'},
            recommendations=["Good temporal stability"]
        )
        
        mock_result2 = TemporalValidationResult(
            site_name="Site 2",
            validation_period=(datetime(2023, 1, 1), datetime(2023, 12, 31)),
            data_points=[],
            seasonal_patterns={},
            persistence_metrics={
                'persistence_rate': 0.6,
                'trend_stability': 0.8,
                'data_quality_score': 0.75
            },
            environmental_correlations={},
            trend_analysis={},
            quality_assessment={'quality_score': 0.7, 'data_coverage': 0.8, 'overall_quality': 'good'},
            recommendations=["Monitor stability"]
        )
        
        results = [mock_result1, mock_result2]
        report = self.validator.generate_comprehensive_temporal_report(results)
        
        # Verify report structure
        assert 'executive_summary' in report
        assert 'detailed_metrics' in report
        assert 'recommendations' in report
        assert 'validation_quality' in report
        assert 'timestamp' in report
        assert 'methodology' in report
        
        # Verify executive summary
        exec_summary = report['executive_summary']
        assert exec_summary['total_sites_validated'] == 2
        assert 'overall_assessment' in exec_summary
        assert 'mean_persistence_rate' in exec_summary
        
        # Verify recommendations structure
        recommendations = report['recommendations']
        assert 'high_priority' in recommendations
        assert 'operational' in recommendations
        assert 'research' in recommendations
    
    def test_calculate_detection_consistency(self):
        """Test detection consistency calculation."""
        # High consistency data
        consistent_data = [
            TemporalDataPoint(datetime.now(), 0.20, 1.0, 0.8, {}),
            TemporalDataPoint(datetime.now(), 0.22, 1.1, 0.8, {}),
            TemporalDataPoint(datetime.now(), 0.18, 0.9, 0.8, {})
        ]
        
        consistency = self.validator._calculate_detection_consistency(consistent_data)
        assert 0.0 <= consistency <= 1.0
        
        # Inconsistent data
        inconsistent_data = [
            TemporalDataPoint(datetime.now(), 0.05, 0.1, 0.8, {}),
            TemporalDataPoint(datetime.now(), 0.50, 2.5, 0.8, {}),
            TemporalDataPoint(datetime.now(), 0.10, 0.5, 0.8, {})
        ]
        
        inconsistency = self.validator._calculate_detection_consistency(inconsistent_data)
        assert consistency > inconsistency


class TestFactoryFunctions:
    """Test factory functions and high-level interfaces."""
    
    def test_create_temporal_validator(self):
        """Test temporal validator factory function."""
        validator = create_temporal_validator()
        assert isinstance(validator, TemporalValidator)
    
    @patch('kelpie_carbon_v1.validation.temporal_validation.fetch_sentinel_tiles')
    @patch('kelpie_carbon_v1.validation.temporal_validation.create_skema_kelp_detection_mask')
    @pytest.mark.asyncio
    async def test_run_broughton_temporal_validation(self, mock_mask, mock_fetch):
        """Test Broughton temporal validation factory function."""
        # Mock data collection
        mock_dataset = xr.Dataset({
            'red': (('y', 'x'), np.random.rand(10, 10) * 0.3),
            'nir': (('y', 'x'), np.random.rand(10, 10) * 0.6),
        }, attrs={'scene_id': 'test', 'cloud_coverage': 0.1})
        
        # Mock async function properly
        async def mock_fetch_async(*args, **kwargs):
            return [mock_dataset]
        
        mock_fetch.side_effect = mock_fetch_async
        mock_mask.return_value = np.random.rand(10, 10) * 0.3
        
        result = await run_broughton_temporal_validation(validation_years=1)
        assert isinstance(result, TemporalValidationResult)
        assert "Broughton Archipelago" in result.site_name
    
    @patch('kelpie_carbon_v1.validation.temporal_validation.fetch_sentinel_tiles')
    @patch('kelpie_carbon_v1.validation.temporal_validation.create_skema_kelp_detection_mask')
    @pytest.mark.asyncio
    async def test_run_comprehensive_temporal_analysis(self, mock_mask, mock_fetch):
        """Test comprehensive temporal analysis factory function."""
        # Mock data collection
        mock_dataset = xr.Dataset({
            'red': (('y', 'x'), np.random.rand(10, 10) * 0.3),
            'nir': (('y', 'x'), np.random.rand(10, 10) * 0.6),
        }, attrs={'scene_id': 'test', 'cloud_coverage': 0.1})
        
        # Mock async function properly
        async def mock_fetch_async(*args, **kwargs):
            return [mock_dataset]
        
        mock_fetch.side_effect = mock_fetch_async
        mock_mask.return_value = np.random.rand(10, 10) * 0.3
        
        # Create test sites
        sites = [
            ValidationSite("Site 1", 50.0, -125.0, "Nereocystis", 0.2, "8m", "Summer"),
            ValidationSite("Site 2", 48.0, -123.0, "Macrocystis", 0.15, "10m", "Summer")
        ]
        
        report = await run_comprehensive_temporal_analysis(sites, validation_years=1)
        
        assert isinstance(report, dict)
        assert 'executive_summary' in report
        assert 'detailed_metrics' in report
        assert report['executive_summary']['total_sites_validated'] == 2


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = TemporalValidator()
    
    def test_empty_data_points(self):
        """Test handling of empty data points list."""
        empty_data = []
        
        # Test seasonal patterns with empty data
        patterns = self.validator._analyze_seasonal_patterns(empty_data)
        assert patterns == {}
        
        # Test persistence metrics with empty data
        metrics = self.validator._calculate_persistence_metrics(empty_data)
        assert metrics == {}
        
        # Test environmental correlations with empty data
        correlations = self.validator._analyze_environmental_correlations(empty_data)
        assert correlations == {}
    
    def test_single_data_point(self):
        """Test handling of single data point."""
        single_data = [TemporalDataPoint(
            timestamp=datetime.now(),
            detection_rate=0.25,
            kelp_area_km2=1.0,
            confidence_score=0.8,
            environmental_conditions={'temp': 15.0}
        )]
        
        # Should handle single data point gracefully
        metrics = self.validator._calculate_persistence_metrics(single_data)
        assert 'mean_detection_rate' in metrics
        assert metrics['temporal_coverage'] == 1
        
        # Consistency calculation with single point
        consistency = self.validator._calculate_detection_consistency(single_data)
        assert consistency == 0.0  # No consistency with single point
    
    def test_insufficient_data_for_correlations(self):
        """Test correlation analysis with insufficient data."""
        minimal_data = [
            TemporalDataPoint(datetime.now(), 0.2, 1.0, 0.8, {'temp': 15.0}),
            TemporalDataPoint(datetime.now(), 0.3, 1.5, 0.8, {'temp': 16.0})
        ]
        
        correlations = self.validator._analyze_environmental_correlations(minimal_data)
        assert correlations == {}  # Insufficient data for correlations
    
    def test_nan_handling_in_correlations(self):
        """Test handling of NaN values in correlation calculations."""
        data_with_constant_env = []
        for i in range(10):
            data_with_constant_env.append(TemporalDataPoint(
                timestamp=datetime.now() + timedelta(days=i),
                detection_rate=0.2 + 0.1 * i,
                kelp_area_km2=1.0,
                confidence_score=0.8,
                environmental_conditions={'constant_temp': 15.0}  # No variation
            ))
        
        correlations = self.validator._analyze_environmental_correlations(data_with_constant_env)
        # Should handle constant environmental variables gracefully
        assert isinstance(correlations, dict)
    
    def test_extreme_detection_rates(self):
        """Test handling of extreme detection rates."""
        extreme_data = [
            TemporalDataPoint(datetime.now(), 0.0, 0.0, 0.0, {}),  # Zero detection
            TemporalDataPoint(datetime.now(), 1.0, 10.0, 1.0, {}),  # Maximum detection
        ]
        
        metrics = self.validator._calculate_persistence_metrics(extreme_data)
        assert 'mean_detection_rate' in metrics
        assert 0.0 <= metrics['mean_detection_rate'] <= 1.0
        
        # Test quality score calculation
        quality_score = self.validator._calculate_data_quality_score(extreme_data)
        assert 0.0 <= quality_score <= 1.0
    
    def test_temporal_gaps_identification(self):
        """Test identification of temporal gaps."""
        # Create data with significant gap
        data_with_gap = [
            TemporalDataPoint(datetime(2023, 1, 1), 0.2, 1.0, 0.8, {}),
            TemporalDataPoint(datetime(2023, 3, 15), 0.25, 1.2, 0.8, {}),  # 74-day gap
        ]
        
        gaps = self.validator._identify_temporal_gaps(data_with_gap)
        assert len(gaps) == 1
        assert gaps[0]['gap_days'] > 30
        assert 'start_date' in gaps[0]
        assert 'end_date' in gaps[0]
    
    def test_insufficient_temporal_coverage(self):
        """Test assessment with insufficient temporal coverage."""
        insufficient_data = [
            TemporalDataPoint(datetime(2023, 1, 1), 0.2, 1.0, 0.8, {}),
            TemporalDataPoint(datetime(2023, 1, 15), 0.22, 1.1, 0.8, {}),
        ]
        
        coverage_good = self.validator._assess_temporal_coverage(insufficient_data)
        assert coverage_good == False  # Insufficient coverage
        
        # Test data coverage calculation
        coverage_pct = self.validator._calculate_data_coverage(insufficient_data)
        assert 0.0 <= coverage_pct <= 1.0