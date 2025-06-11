"""
Unit tests for Historical Baseline Analysis module.

This module tests the comprehensive historical baseline analysis capabilities,
including historical data digitization, change detection algorithms, and temporal
trend analysis following UVic methodology.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import json
from pathlib import Path
import tempfile

from src.kelpie_carbon_v1.validation.historical_baseline_analysis import (
    HistoricalSite,
    HistoricalDataset,
    ChangeDetectionAnalyzer,
    TemporalTrendAnalyzer,
    HistoricalBaselineAnalysis,
    create_uvic_historical_sites,
    create_sample_historical_dataset
)

class TestHistoricalSite:
    """Test HistoricalSite data structure."""
    
    def test_valid_site_creation(self):
        """Test creating a valid historical site."""
        site = HistoricalSite(
            name="Test Site",
            latitude=50.0,
            longitude=-125.0,
            region="British Columbia",
            historical_period=(1858, 1956),
            data_sources=["Test Source"],
            species=["Nereocystis luetkeana"]
        )
        
        assert site.name == "Test Site"
        assert site.latitude == 50.0
        assert site.longitude == -125.0
        assert site.region == "British Columbia"
        assert site.historical_period == (1858, 1956)
        assert site.species == ["Nereocystis luetkeana"]
        assert site.digitization_quality == "high"
    
    def test_invalid_latitude(self):
        """Test validation of latitude bounds."""
        with pytest.raises(ValueError, match="Invalid latitude"):
            HistoricalSite(
                name="Test",
                latitude=95.0,
                longitude=-125.0,
                region="Test",
                historical_period=(1850, 1950),
                data_sources=["Test"],
                species=["Test"]
            )
    
    def test_invalid_longitude(self):
        """Test validation of longitude bounds."""
        with pytest.raises(ValueError, match="Invalid longitude"):
            HistoricalSite(
                name="Test",
                latitude=50.0,
                longitude=185.0,
                region="Test",
                historical_period=(1850, 1950),
                data_sources=["Test"],
                species=["Test"]
            )
    
    def test_invalid_historical_period(self):
        """Test validation of historical period."""
        with pytest.raises(ValueError, match="Start year must be <= end year"):
            HistoricalSite(
                name="Test",
                latitude=50.0,
                longitude=-125.0,
                region="Test",
                historical_period=(1950, 1850),
                data_sources=["Test"],
                species=["Test"]
            )
    
    def test_invalid_quality(self):
        """Test validation of digitization quality."""
        with pytest.raises(ValueError, match="Quality must be"):
            HistoricalSite(
                name="Test",
                latitude=50.0,
                longitude=-125.0,
                region="Test",
                historical_period=(1850, 1950),
                data_sources=["Test"],
                species=["Test"],
                digitization_quality="invalid"
            )

class TestHistoricalDataset:
    """Test HistoricalDataset data structure."""
    
    def create_test_site(self):
        """Create a test site for dataset testing."""
        return HistoricalSite(
            name="Test Site",
            latitude=50.0,
            longitude=-125.0,
            region="Test Region",
            historical_period=(1850, 1950),
            data_sources=["Test Source"],
            species=["Test Species"]
        )
    
    def test_valid_dataset_creation(self):
        """Test creating a valid historical dataset."""
        site = self.create_test_site()
        temporal_data = {
            1850: {"extent": 100.0, "confidence": 0.8},
            1900: {"extent": 90.0, "confidence": 0.9},
            1950: {"extent": 80.0, "confidence": 0.85}
        }
        confidence_intervals = {
            1850: (80.0, 120.0),
            1900: (75.0, 105.0),
            1950: (65.0, 95.0)
        }
        
        dataset = HistoricalDataset(
            site=site,
            temporal_data=temporal_data,
            baseline_extent=97.5,
            confidence_intervals=confidence_intervals,
            data_quality_metrics={"temporal_coverage": 0.9}
        )
        
        assert dataset.site.name == "Test Site"
        assert len(dataset.temporal_data) == 3
        assert dataset.baseline_extent == 97.5
        assert len(dataset.confidence_intervals) == 3
    
    def test_empty_temporal_data(self):
        """Test validation of empty temporal data."""
        site = self.create_test_site()
        
        with pytest.raises(ValueError, match="Temporal data cannot be empty"):
            HistoricalDataset(
                site=site,
                temporal_data={},
                baseline_extent=100.0,
                confidence_intervals={},
                data_quality_metrics={}
            )
    
    def test_negative_baseline_extent(self):
        """Test validation of negative baseline extent."""
        site = self.create_test_site()
        temporal_data = {1850: {"extent": 100.0, "confidence": 0.8}}
        
        with pytest.raises(ValueError, match="Baseline extent must be >= 0"):
            HistoricalDataset(
                site=site,
                temporal_data=temporal_data,
                baseline_extent=-10.0,
                confidence_intervals={},
                data_quality_metrics={}
            )

class TestChangeDetectionAnalyzer:
    """Test ChangeDetectionAnalyzer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ChangeDetectionAnalyzer()
        
        self.historical_data = {
            1850: 100.0, 1860: 95.0, 1870: 90.0, 1880: 85.0, 1890: 80.0
        }
        self.current_data = {2020: 75.0, 2021: 70.0, 2022: 65.0}
    
    def test_detect_significant_changes_mann_kendall(self):
        """Test change detection using Mann-Kendall test."""
        result = self.analyzer.detect_significant_changes(
            self.historical_data, 
            self.current_data,
            method="mann_kendall"
        )
        
        assert "historical_mean" in result
        assert "current_mean" in result
        assert "relative_change_percent" in result
        assert "is_significant" in result
        assert result["statistical_test"] == "mann_kendall"
        
        assert result["historical_mean"] == 90.0
        assert result["current_mean"] == 70.0
        assert result["relative_change_percent"] < 0
    
    def test_detect_significant_changes_t_test(self):
        """Test change detection using t-test."""
        result = self.analyzer.detect_significant_changes(
            self.historical_data,
            self.current_data,
            method="t_test"
        )
        
        assert result["statistical_test"] == "t_test"
        assert "test_statistic" in result
        assert "p_value" in result
        assert "change_magnitude" in result
    
    def test_detect_significant_changes_wilcoxon(self):
        """Test change detection using Wilcoxon test."""
        result = self.analyzer.detect_significant_changes(
            self.historical_data,
            self.current_data,
            method="wilcoxon"
        )
        
        assert result["statistical_test"] == "wilcoxon"
        assert "test_statistic" in result
        assert "p_value" in result
    
    def test_invalid_method(self):
        """Test error handling for invalid method."""
        result = self.analyzer.detect_significant_changes(
            self.historical_data,
            self.current_data,
            method="invalid_method"
        )
        
        assert "error" in result
        assert "Unknown method" in result["error"]
    
    def test_mann_kendall_test_insufficient_data(self):
        """Test Mann-Kendall test with insufficient data."""
        statistic, p_value = self.analyzer._mann_kendall_test([1, 2])
        assert statistic == 0.0
        assert p_value == 1.0
    
    def test_mann_kendall_test_increasing_trend(self):
        """Test Mann-Kendall test with increasing trend."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        statistic, p_value = self.analyzer._mann_kendall_test(data)
        assert statistic > 0
        assert p_value < 0.05
    
    def test_mann_kendall_test_decreasing_trend(self):
        """Test Mann-Kendall test with decreasing trend."""
        data = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        statistic, p_value = self.analyzer._mann_kendall_test(data)
        assert statistic < 0
        assert p_value < 0.05
    
    def test_analyze_change_patterns(self):
        """Test comprehensive change pattern analysis."""
        site = HistoricalSite(
            name="Test Site",
            latitude=50.0,
            longitude=-125.0,
            region="Test",
            historical_period=(1850, 1900),
            data_sources=["Test"],
            species=["Test"]
        )
        
        temporal_data = {
            year: {"extent": 100 - (year - 1850) * 0.5, "confidence": 0.8}
            for year in range(1850, 1901, 10)
        }
        
        dataset = HistoricalDataset(
            site=site,
            temporal_data=temporal_data,
            baseline_extent=100.0,
            confidence_intervals={},
            data_quality_metrics={}
        )
        
        result = self.analyzer.analyze_change_patterns(dataset, 75.0, 2024)
        
        assert "trend_analysis" in result
        assert "decadal_averages" in result
        assert "change_points" in result
        assert "variability_metrics" in result
        assert "baseline_comparison" in result
        
        assert result["trend_analysis"]["trend_direction"] == "decreasing"
        assert result["baseline_comparison"]["current_vs_baseline"] < 0

class TestTemporalTrendAnalyzer:
    """Test TemporalTrendAnalyzer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TemporalTrendAnalyzer()
        
        self.site = HistoricalSite(
            name="Test Site",
            latitude=50.0,
            longitude=-125.0,
            region="Test",
            historical_period=(1850, 1950),
            data_sources=["Test"],
            species=["Test"]
        )
        
        self.temporal_data = {
            year: {"extent": 100 - (year - 1850) * 0.3, "confidence": 0.8 + (year - 1850) * 0.001}
            for year in range(1850, 1951, 10)
        }
        
        self.dataset = HistoricalDataset(
            site=self.site,
            temporal_data=self.temporal_data,
            baseline_extent=100.0,
            confidence_intervals={},
            data_quality_metrics={}
        )
    
    def test_analyze_temporal_trends(self):
        """Test comprehensive temporal trend analysis."""
        result = self.analyzer.analyze_temporal_trends(self.dataset)
        
        assert "trend_metrics" in result
        assert "seasonal_patterns" in result
        assert "cyclical_patterns" in result
        assert "forecast" in result
        assert "risk_assessment" in result
        assert "data_quality" in result
        
        trend_metrics = result["trend_metrics"]
        assert "linear_trend" in trend_metrics
        assert trend_metrics["linear_trend"]["slope"] < 0
    
    def test_analyze_temporal_trends_no_forecast(self):
        """Test temporal trend analysis without forecast."""
        result = self.analyzer.analyze_temporal_trends(self.dataset, include_forecast=False)
        
        assert "forecast" in result
        assert result["forecast"] == {}
    
    def test_calculate_trend_metrics_insufficient_data(self):
        """Test trend metrics with insufficient data."""
        result = self.analyzer._calculate_trend_metrics([1850, 1860], [100, 95])
        assert "error" in result
        assert "Insufficient data" in result["error"]
    
    def test_calculate_trend_metrics_sufficient_data(self):
        """Test trend metrics with sufficient data."""
        years = list(range(1850, 1901, 10))
        extents = [100 - (year - 1850) * 0.2 for year in years]
        
        result = self.analyzer._calculate_trend_metrics(years, extents)
        
        assert "linear_trend" in result
        assert "polynomial_trend" in result
        assert "rate_of_change" in result
        
        assert result["linear_trend"]["slope"] < 0
        assert result["linear_trend"]["r_squared"] > 0.8
    
    def test_analyze_seasonal_patterns_insufficient_data(self):
        """Test seasonal pattern analysis with insufficient data."""
        temporal_data = {year: {"extent": 100.0, "confidence": 0.8} for year in range(1850, 1855)}
        
        result = self.analyzer._analyze_seasonal_patterns(temporal_data)
        assert "note" in result
        assert "Insufficient data" in result["note"]
    
    def test_detect_cyclical_patterns_insufficient_data(self):
        """Test cyclical pattern detection with insufficient data."""
        years = [1850, 1860, 1870]
        extents = [100, 95, 90]
        
        result = self.analyzer._detect_cyclical_patterns(years, extents)
        assert "note" in result
        assert "Insufficient data" in result["note"]
    
    def test_detect_cyclical_patterns_sufficient_data(self):
        """Test cyclical pattern detection with sufficient data."""
        years = list(range(1850, 1901, 2))
        extents = [100 + 10 * np.sin(2 * np.pi * (year - 1850) / 10) for year in years]
        
        result = self.analyzer._detect_cyclical_patterns(years, extents)
        
        assert "autocorrelation" in result
        assert "significant_cycles" in result
        assert "max_autocorr" in result
        assert len(result["autocorrelation"]) > 0
    
    def test_generate_forecast_insufficient_data(self):
        """Test forecast generation with insufficient data."""
        years = [1850, 1860]
        extents = [100, 95]
        
        result = self.analyzer._generate_forecast(years, extents)
        assert "note" in result
        assert "Insufficient data" in result["note"]
    
    def test_generate_forecast_sufficient_data(self):
        """Test forecast generation with sufficient data."""
        years = list(range(1850, 1901, 10))
        extents = [100 - (year - 1850) * 0.2 for year in years]
        
        result = self.analyzer._generate_forecast(years, extents)
        
        assert "method" in result
        assert "forecast_years" in result
        assert "predicted_extents" in result
        assert "confidence_intervals_95" in result
        assert result["method"] == "linear_regression"
        assert len(result["forecast_years"]) == self.analyzer.forecast_years
    
    def test_assess_trend_risks_high_risk(self):
        """Test risk assessment for high-risk trends."""
        trend_results = {
            "linear_trend": {"slope": -1.0, "p_value": 0.01},
            "rate_of_change": {"rate_std": 2.0}
        }
        forecast_results = {
            "predicted_extents": [-5, -10, -15, -20, -25]
        }
        
        result = self.analyzer._assess_trend_risks(trend_results, forecast_results)
        
        assert result["overall_risk_level"] == "high"
        assert len(result["risk_factors"]) > 0
        assert "Significant declining trend" in result["risk_factors"]
        assert "Forecast predicts potential extinction" in result["risk_factors"]
    
    def test_assess_trend_risks_low_risk(self):
        """Test risk assessment for low-risk trends."""
        trend_results = {
            "linear_trend": {"slope": 0.1, "p_value": 0.5},
            "rate_of_change": {"rate_std": 0.1}
        }
        forecast_results = {
            "predicted_extents": [105, 110, 115, 120, 125]
        }
        
        result = self.analyzer._assess_trend_risks(trend_results, forecast_results)
        
        assert result["overall_risk_level"] == "low"
        assert len(result["recommendations"]) > 0
        assert "Continue regular monitoring" in result["recommendations"]

class TestHistoricalBaselineAnalysis:
    """Test main HistoricalBaselineAnalysis framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = HistoricalBaselineAnalysis()
    
    def test_create_historical_site(self):
        """Test creating and registering a historical site."""
        site = self.analyzer.create_historical_site(
            name="Test Site",
            latitude=50.0,
            longitude=-125.0,
            region="Test Region",
            historical_period=(1850, 1950),
            data_sources=["Test Source"],
            species=["Test Species"]
        )
        
        assert site.name == "Test Site"
        assert "Test Site" in self.analyzer.historical_sites
        assert self.analyzer.historical_sites["Test Site"] == site
    
    def test_digitize_historical_data(self):
        """Test historical data digitization."""
        self.analyzer.create_historical_site(
            name="Test Site",
            latitude=50.0,
            longitude=-125.0,
            region="Test Region",
            historical_period=(1850, 1950),
            data_sources=["Test Source"],
            species=["Test Species"]
        )
        
        chart_data = {
            1850: {"extent": "100.0", "confidence": 0.9, "source": "Chart 1"},
            1860: {"extent": "95.0", "confidence": 0.8, "source": "Chart 2"},
            1870: {"extent": "90.0", "confidence": 0.85, "source": "Chart 3"}
        }
        
        dataset = self.analyzer.digitize_historical_data("Test Site", chart_data)
        
        assert dataset.site.name == "Test Site"
        assert len(dataset.temporal_data) == 3
        assert dataset.baseline_extent == 97.5
        assert "Test Site" in self.analyzer.historical_datasets
    
    def test_digitize_historical_data_site_not_found(self):
        """Test error handling for missing site."""
        with pytest.raises(ValueError, match="Site .* not found"):
            self.analyzer.digitize_historical_data("Missing Site", {})
    
    def test_apply_quality_control(self):
        """Test quality control procedures."""
        chart_data = {
            1850: {"extent": "100.0", "confidence": 0.9, "source": "Good"},
            1860: {"extent": "invalid", "confidence": 0.8, "source": "Bad"},
            1870: {"extent": "90.0", "confidence": 0.3, "source": "Low confidence"},
            1880: {"extent": "85.0", "confidence": 0.8, "source": "Good"}
        }
        
        qc_params = {"min_confidence": 0.5}
        result = self.analyzer._apply_quality_control(chart_data, qc_params)
        
        assert len(result) == 2
        assert 1850 in result
        assert 1880 in result
        assert 1860 not in result
        assert 1870 not in result
    
    def test_calculate_data_quality_metrics(self):
        """Test data quality metrics calculation."""
        site = HistoricalSite(
            name="Test Site",
            latitude=50.0,
            longitude=-125.0,
            region="Test",
            historical_period=(1850, 1890),
            data_sources=["Test"],
            species=["Test"]
        )
        
        processed_data = {
            1850: {"extent": 100.0, "confidence": 0.9},
            1860: {"extent": 95.0, "confidence": 0.8},
            1870: {"extent": 90.0, "confidence": 0.85},
            1880: {"extent": 85.0, "confidence": 0.9}
        }
        
        metrics = self.analyzer._calculate_data_quality_metrics(processed_data, site)
        
        assert "temporal_coverage" in metrics
        assert "mean_confidence" in metrics
        assert "min_confidence" in metrics
        assert "max_temporal_gap" in metrics
        assert "data_completeness" in metrics
        
        expected_years = 1890 - 1850 + 1
        actual_years = 4
        assert metrics["temporal_coverage"] == actual_years / expected_years
        assert metrics["mean_confidence"] == 0.8625
        assert metrics["min_confidence"] == 0.8
        assert metrics["max_temporal_gap"] == 10
    
    def test_perform_comprehensive_analysis(self):
        """Test comprehensive analysis workflow."""
        self.analyzer.create_historical_site(
            name="Test Site",
            latitude=50.0,
            longitude=-125.0,
            region="Test Region",
            historical_period=(1850, 1950),
            data_sources=["Test Source"],
            species=["Test Species"]
        )
        
        chart_data = {
            1850: {"extent": 100.0, "confidence": 0.9},
            1870: {"extent": 95.0, "confidence": 0.8},
            1890: {"extent": 90.0, "confidence": 0.85},
            1910: {"extent": 85.0, "confidence": 0.9},
            1930: {"extent": 80.0, "confidence": 0.8},
            1950: {"extent": 75.0, "confidence": 0.85}
        }
        
        self.analyzer.digitize_historical_data("Test Site", chart_data)
        
        result = self.analyzer.perform_comprehensive_analysis(
            "Test Site", 
            current_extent=70.0,
            current_year=2024
        )
        
        assert "site_information" in result
        assert "change_detection" in result
        assert "change_patterns" in result
        assert "temporal_trends" in result
        assert "data_quality" in result
        assert "analysis_metadata" in result
        
        site_info = result["site_information"]
        assert site_info["name"] == "Test Site"
        assert site_info["region"] == "Test Region"
    
    def test_perform_comprehensive_analysis_missing_dataset(self):
        """Test error handling for missing dataset."""
        with pytest.raises(ValueError, match="No historical dataset found"):
            self.analyzer.perform_comprehensive_analysis("Missing Site", 70.0)
    
    def test_generate_comparison_report(self):
        """Test comparative analysis report generation."""
        sites = ["Site A", "Site B"]
        
        for i, site_name in enumerate(sites):
            self.analyzer.create_historical_site(
                name=site_name,
                latitude=50.0 + i,
                longitude=-125.0 - i,
                region=f"Region {i+1}",
                historical_period=(1850, 1950),
                data_sources=[f"Source {i+1}"],
                species=[f"Species {i+1}"]
            )
            
            chart_data = {
                year: {"extent": 100.0 - i * 10 - (year - 1850) * 0.1, "confidence": 0.8}
                for year in range(1850, 1951, 20)
            }
            
            self.analyzer.digitize_historical_data(site_name, chart_data)
        
        report = self.analyzer.generate_comparison_report(sites)
        
        assert "summary_statistics" in report
        assert "site_comparisons" in report
        assert "report_metadata" in report
        
        summary = report["summary_statistics"]
        assert summary["total_sites"] == 2
        assert summary["regional_diversity"] == 2
        
        assert "Site A" in report["site_comparisons"]
        assert "Site B" in report["site_comparisons"]
    
    def test_generate_comparison_report_missing_sites(self):
        """Test error handling for missing sites in comparison."""
        with pytest.raises(ValueError, match="Missing datasets for sites"):
            self.analyzer.generate_comparison_report(["Missing Site"])
    
    def test_generate_comparison_report_json_format(self):
        """Test JSON format output for comparison report."""
        self.analyzer.create_historical_site(
            name="Test Site",
            latitude=50.0,
            longitude=-125.0,
            region="Test",
            historical_period=(1850, 1950),
            data_sources=["Test"],
            species=["Test"]
        )
        
        chart_data = {1850: {"extent": 100.0, "confidence": 0.8}}
        self.analyzer.digitize_historical_data("Test Site", chart_data)
        
        report = self.analyzer.generate_comparison_report(["Test Site"], output_format="json")
        
        assert isinstance(report, str)
        parsed = json.loads(report)
        assert "summary_statistics" in parsed
    
    def test_generate_comparison_report_markdown_format(self):
        """Test Markdown format output for comparison report."""
        self.analyzer.create_historical_site(
            name="Test Site",
            latitude=50.0,
            longitude=-125.0,
            region="Test",
            historical_period=(1850, 1950),
            data_sources=["Test"],
            species=["Test"]
        )
        
        chart_data = {1850: {"extent": 100.0, "confidence": 0.8}}
        self.analyzer.digitize_historical_data("Test Site", chart_data)
        
        report = self.analyzer.generate_comparison_report(["Test Site"], output_format="markdown")
        
        assert isinstance(report, str)
        assert "# Historical Baseline Analysis" in report
        assert "## Summary Statistics" in report
        assert "### Test Site" in report
    
    def test_export_results(self):
        """Test exporting results to file."""
        self.analyzer.create_historical_site(
            name="Test Site",
            latitude=50.0,
            longitude=-125.0,
            region="Test",
            historical_period=(1850, 1950),
            data_sources=["Test"],
            species=["Test"]
        )
        
        chart_data = {1850: {"extent": 100.0, "confidence": 0.8}}
        self.analyzer.digitize_historical_data("Test Site", chart_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_export"
            self.analyzer.export_results("Test Site", output_path)
            
            json_file = output_path.with_suffix('.json')
            assert json_file.exists()
            
            with open(json_file) as f:
                data = json.load(f)
            
            assert "site_metadata" in data
            assert "temporal_data" in data
            assert "baseline_extent" in data
    
    def test_export_results_with_visualizations(self):
        """Test exporting results with visualizations."""
        self.analyzer.create_historical_site(
            name="Test Site",
            latitude=50.0,
            longitude=-125.0,
            region="Test",
            historical_period=(1850, 1950),
            data_sources=["Test"],
            species=["Test"]
        )
        
        chart_data = {
            year: {"extent": 100.0 - (year - 1850) * 0.1, "confidence": 0.8}
            for year in range(1850, 1951, 20)
        }
        self.analyzer.digitize_historical_data("Test Site", chart_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_export"
            
            with patch('matplotlib.pyplot.savefig') as mock_savefig, \
                 patch('matplotlib.pyplot.close') as mock_close:
                
                self.analyzer.export_results("Test Site", output_path, include_visualizations=True)
                
                mock_savefig.assert_called_once()
                mock_close.assert_called_once()
    
    def test_export_results_missing_site(self):
        """Test error handling for missing site in export."""
        with pytest.raises(ValueError, match="No dataset found for site"):
            self.analyzer.export_results("Missing Site", "output.json")

class TestFactoryFunctions:
    """Test factory functions for creating historical sites and datasets."""
    
    def test_create_uvic_historical_sites(self):
        """Test creation of UVic historical sites."""
        sites = create_uvic_historical_sites()
        
        assert isinstance(sites, dict)
        assert len(sites) >= 3
        
        assert "broughton" in sites
        assert "saanich" in sites
        assert "monterey" in sites
        
        broughton = sites["broughton"]
        assert broughton.name == "Broughton Archipelago"
        assert broughton.region == "British Columbia"
        assert "Nereocystis luetkeana" in broughton.species
        assert broughton.digitization_quality == "high"
    
    def test_create_sample_historical_dataset(self):
        """Test creation of sample historical dataset."""
        dataset = create_sample_historical_dataset()
        
        assert isinstance(dataset, HistoricalDataset)
        assert dataset.site.name == "Sample Broughton Site"
        assert len(dataset.temporal_data) > 10
        assert dataset.baseline_extent > 0
        
        for year, data in dataset.temporal_data.items():
            assert "extent" in data
            assert "confidence" in data
            assert "source" in data
            assert "notes" in data
            assert data["extent"] >= 0
            assert 0 <= data["confidence"] <= 1

class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    def test_complete_workflow(self):
        """Test complete analysis workflow from start to finish."""
        analyzer = HistoricalBaselineAnalysis()
        
        site = analyzer.create_historical_site(
            name="Integration Test Site",
            latitude=50.0,
            longitude=-125.0,
            region="Test Region",
            historical_period=(1850, 1950),
            data_sources=["Test Charts"],
            species=["Nereocystis luetkeana"]
        )
        
        chart_data = {}
        np.random.seed(42)
        for year in range(1850, 1951, 5):
            base_decline = (year - 1850) * 0.3
            noise = np.random.normal(0, 5)
            extent = max(0, 150 - base_decline + noise)
            
            chart_data[year] = {
                "extent": extent,
                "confidence": np.random.uniform(0.7, 0.9),
                "source": f"Chart_{year}",
                "notes": f"Historical data for {year}"
            }
        
        dataset = analyzer.digitize_historical_data("Integration Test Site", chart_data)
        
        analysis = analyzer.perform_comprehensive_analysis(
            "Integration Test Site",
            current_extent=80.0,
            current_year=2024,
            analysis_options={"include_forecast": True}
        )
        
        assert analysis["site_information"]["name"] == "Integration Test Site"
        assert "change_detection" in analysis
        assert "temporal_trends" in analysis
        
        change_detection = analysis["change_detection"]
        assert change_detection["relative_change_percent"] < 0
        
        report = analyzer.generate_comparison_report(["Integration Test Site"])
        assert report["summary_statistics"]["total_sites"] == 1
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "integration_test"
            analyzer.export_results("Integration Test Site", output_path)
            
            json_file = output_path.with_suffix('.json')
            assert json_file.exists()
    
    def test_error_recovery(self):
        """Test error recovery and graceful degradation."""
        analyzer = HistoricalBaselineAnalysis()
        
        analyzer.create_historical_site(
            name="Error Test Site",
            latitude=50.0,
            longitude=-125.0,
            region="Test",
            historical_period=(1850, 1950),
            data_sources=["Test"],
            species=["Test"]
        )
        
        problematic_data = {
            1850: {"extent": "invalid", "confidence": 0.8},
            1860: {"extent": 100.0, "confidence": 0.2},
            1870: {"extent": 1000.0, "confidence": 0.9},
            1880: {"extent": 90.0, "confidence": 0.8},
            1890: {"extent": 85.0, "confidence": 0.9}
        }
        
        dataset = analyzer.digitize_historical_data("Error Test Site", problematic_data)
        
        assert len(dataset.temporal_data) >= 2
        
        analysis = analyzer.perform_comprehensive_analysis(
            "Error Test Site",
            current_extent=80.0
        )
        
        assert "error" not in analysis
    
    def test_minimal_data_analysis(self):
        """Test analysis with minimal data."""
        analyzer = HistoricalBaselineAnalysis()
        
        analyzer.create_historical_site(
            name="Minimal Data Site",
            latitude=50.0,
            longitude=-125.0,
            region="Test",
            historical_period=(1950, 1960),
            data_sources=["Test"],
            species=["Test"]
        )
        
        minimal_data = {
            1950: {"extent": 100.0, "confidence": 0.8},
            1960: {"extent": 95.0, "confidence": 0.8}
        }
        
        dataset = analyzer.digitize_historical_data("Minimal Data Site", minimal_data)
        
        analysis = analyzer.perform_comprehensive_analysis(
            "Minimal Data Site",
            current_extent=90.0
        )
        
        assert "temporal_trends" in analysis
        trends = analysis["temporal_trends"]
        
        assert "data_quality" in trends
        assert trends["data_quality"]["temporal_coverage_years"] == 2

if __name__ == "__main__":
    analyzer = HistoricalBaselineAnalysis()
    
    site = analyzer.create_historical_site(
        name="Test Site",
        latitude=50.0,
        longitude=-125.0,
        region="Test Region",
        historical_period=(1850, 1950),
        data_sources=["Test Source"],
        species=["Test Species"]
    )
    
    print(f"Created site: {site.name}")
    print("Historical baseline analysis tests ready to run!") 