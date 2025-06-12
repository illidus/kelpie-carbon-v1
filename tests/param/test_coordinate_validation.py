"""
Consolidated coordinate validation tests.

Parameterized tests for all coordinate validation scenarios.
"""

import pytest


@pytest.mark.parametrize(
    "latitude, longitude, expected_error, error_message",
    [
        # Invalid latitude tests
        (91.0, -123.5, ValueError, "Invalid latitude"),
        (-91.0, -123.5, ValueError, "Invalid latitude"),
        (95.0, -125.0, ValueError, "Invalid latitude"),
        (-95.0, -125.0, ValueError, "Invalid latitude"),
        
        # Invalid longitude tests  
        (48.5, 181.0, ValueError, "Invalid longitude"),
        (48.5, -181.0, ValueError, "Invalid longitude"),
        (50.0, 185.0, ValueError, "Invalid longitude"),
        (50.0, -185.0, ValueError, "Invalid longitude"),
        
        # Edge cases
        (90.0, 180.0, None, None),  # Valid edge case
        (-90.0, -180.0, None, None),  # Valid edge case
    ],
)
def test_coordinate_validation(latitude, longitude, expected_error, error_message):
    """Test coordinate validation with various invalid inputs."""
    from src.kelpie_carbon.validation.historical_baseline_analysis import HistoricalSite
    
    if expected_error:
        with pytest.raises(expected_error, match=error_message):
            HistoricalSite(
                name="Test Site",
                latitude=latitude,
                longitude=longitude,
                region="Test Region",
                historical_period=(1850, 1950),
                data_sources=["Test Source"],
                species=["Test Species"]
            )
    else:
        # Should not raise an error
        site = HistoricalSite(
            name="Test Site",
            latitude=latitude,
            longitude=longitude,
            region="Test Region", 
            historical_period=(1850, 1950),
            data_sources=["Test Source"],
            species=["Test Species"]
        )
        assert site.latitude == latitude
        assert site.longitude == longitude


@pytest.mark.parametrize(
    "analysis_type, coordinates, time_range, should_raise",
    [
        # Valid coordinates
        ("validation", (48.5, -123.5), ("2023-01-01", "2023-12-31"), False),
        ("temporal", (50.0, -125.0), ("2023-01-01", "2023-12-31"), False),
        
        # Invalid coordinates in analysis requests
        ("validation", (91.0, -123.5), ("2023-01-01", "2023-12-31"), True),
        ("temporal", (48.5, -181.0), ("2023-01-01", "2023-12-31"), True),
    ],
)
def test_analysis_coordinate_validation(analysis_type, coordinates, time_range, should_raise):
    """Test coordinate validation in analysis requests."""
    try:
        from src.kelpie_carbon.analytics.analytics_framework import AnalysisRequest, AnalysisType
        from datetime import datetime
        
        analysis_types = [AnalysisType.VALIDATION if analysis_type == "validation" else AnalysisType.TEMPORAL]
        time_range_parsed = (datetime.fromisoformat(time_range[0]), datetime.fromisoformat(time_range[1]))
        
        if should_raise:
            with pytest.raises(ValueError):
                AnalysisRequest(
                    analysis_types=analysis_types,
                    site_coordinates=coordinates,
                    time_range=time_range_parsed
                )
        else:
            request = AnalysisRequest(
                analysis_types=analysis_types,
                site_coordinates=coordinates,
                time_range=time_range_parsed
            )
            assert request.site_coordinates == coordinates
            
    except ImportError:
        pytest.skip("Analytics framework not available")
