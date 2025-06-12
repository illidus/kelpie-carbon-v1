"""
Comprehensive metrics validation tests.

Replaces multiple validation test files with parameterized tests.
"""

import pytest
import numpy as np
from unittest.mock import Mock


@pytest.mark.parametrize(
    "metric_type, input_data, expected_error",
    [
        # Detection metrics
        ("detection_accuracy", [], ValueError),
        ("detection_accuracy", None, (ValueError, TypeError)),
        ("detection_accuracy", [0.5, 0.8, 0.9], None),
        
        # Temporal metrics  
        ("temporal_trend", {}, ValueError),
        ("temporal_trend", {"2020": 100, "2021": 110}, None),
        
        # Composite metrics
        ("composite_score", [-1.0], ValueError),
        ("composite_score", [1.5], ValueError), 
        ("composite_score", [0.0, 0.5, 1.0], None),
        
        # Performance metrics
        ("performance", {"accuracy": -0.1}, ValueError),
        ("performance", {"accuracy": 1.1}, ValueError),
        ("performance", {"accuracy": 0.85}, None),
    ],
)
def test_metrics_validation(metric_type, input_data, expected_error):
    """Test various metrics validation scenarios."""
    try:
        from src.kelpie_carbon.analytics.analytics_framework import MetricCalculator
        
        calculator = MetricCalculator()
        
        if expected_error:
            with pytest.raises(expected_error):
                if metric_type == "detection_accuracy":
                    calculator.calculate_detection_metrics(
                        predicted=input_data,
                        ground_truth=[0.7, 0.8, 0.9] if input_data else []
                    )
                elif metric_type == "composite_score":
                    calculator.calculate_composite_score({"metrics": input_data})
        else:
            # Should not raise error
            if metric_type == "detection_accuracy" and input_data:
                result = calculator.calculate_detection_metrics(
                    predicted=input_data,
                    ground_truth=[0.6, 0.7, 0.8]
                )
                assert "accuracy" in result
                
    except ImportError:
        pytest.skip("Analytics framework not available")


@pytest.mark.parametrize(
    "analysis_type, site_data, time_data, should_succeed",
    [
        # Valid analysis scenarios
        ("validation", {"lat": 48.5, "lon": -123.5}, {"start": "2023-01-01", "end": "2023-12-31"}, True),
        ("temporal", {"lat": 50.0, "lon": -125.0}, {"start": "2022-01-01", "end": "2023-12-31"}, True),
        
        # Invalid site data
        ("validation", {"lat": 95.0, "lon": -123.5}, {"start": "2023-01-01", "end": "2023-12-31"}, False),
        ("temporal", {"lat": 48.5, "lon": 185.0}, {"start": "2023-01-01", "end": "2023-12-31"}, False),
        
        # Invalid time data
        ("validation", {"lat": 48.5, "lon": -123.5}, {"start": "2023-12-31", "end": "2023-01-01"}, False),
    ],
)
def test_analysis_creation_scenarios(analysis_type, site_data, time_data, should_succeed):
    """Test analysis creation with various input scenarios."""
    try:
        from src.kelpie_carbon.analytics.analytics_framework import AnalysisRequest, AnalysisType
        from datetime import datetime
        
        analysis_types = [AnalysisType.VALIDATION if analysis_type == "validation" else AnalysisType.TEMPORAL]
        coordinates = (site_data["lat"], site_data["lon"])
        time_range = (datetime.fromisoformat(time_data["start"]), datetime.fromisoformat(time_data["end"]))
        
        if should_succeed:
            request = AnalysisRequest(
                analysis_types=analysis_types,
                site_coordinates=coordinates,
                time_range=time_range
            )
            assert request.analysis_types == analysis_types
            assert request.site_coordinates == coordinates
        else:
            with pytest.raises((ValueError, TypeError)):
                AnalysisRequest(
                    analysis_types=analysis_types,
                    site_coordinates=coordinates,
                    time_range=time_range
                )
                
    except ImportError:
        pytest.skip("Analytics framework not available")
