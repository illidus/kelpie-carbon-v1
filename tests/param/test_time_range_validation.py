"""
Consolidated time range validation tests.

Parameterized tests for all time range validation scenarios.
"""

from datetime import datetime

import pytest


@pytest.mark.parametrize(
    "start_year, end_year, expected_error, error_message",
    [
        # Invalid historical periods
        (1950, 1850, ValueError, "Start year must be <= end year"),
        (2000, 1990, ValueError, "Start year must be <= end year"),
        (1900, 1850, ValueError, "Start year must be <= end year"),
        # Valid historical periods
        (1850, 1950, None, None),
        (1900, 1950, None, None),
        (1850, 2000, None, None),
        (2020, 2023, None, None),
    ],
)
def test_historical_period_validation(
    start_year, end_year, expected_error, error_message
):
    """Test historical period validation."""
    try:
        from src.kelpie_carbon.validation.historical_baseline_analysis import (
            HistoricalSite,
        )

        if expected_error:
            with pytest.raises(expected_error, match=error_message):
                HistoricalSite(
                    name="Test Site",
                    latitude=50.0,
                    longitude=-125.0,
                    region="Test Region",
                    historical_period=(start_year, end_year),
                    data_sources=["Test Source"],
                    species=["Test Species"],
                )
        else:
            site = HistoricalSite(
                name="Test Site",
                latitude=50.0,
                longitude=-125.0,
                region="Test Region",
                historical_period=(start_year, end_year),
                data_sources=["Test Source"],
                species=["Test Species"],
            )
            assert site.historical_period == (start_year, end_year)

    except ImportError:
        pytest.skip("Historical baseline analysis not available")


@pytest.mark.parametrize(
    "start_date, end_date, expected_error",
    [
        # Invalid time ranges (end before start)
        ("2023-12-31", "2023-01-01", ValueError),
        ("2023-06-01", "2023-01-01", ValueError),
        ("2024-01-01", "2023-01-01", ValueError),
        # Valid time ranges
        ("2023-01-01", "2023-12-31", None),
        ("2023-01-01", "2023-06-01", None),
        ("2022-01-01", "2023-12-31", None),
    ],
)
def test_analysis_time_range_validation(start_date, end_date, expected_error):
    """Test time range validation in analysis requests."""
    try:
        from src.kelpie_carbon.analytics.analytics_framework import (
            AnalysisRequest,
            AnalysisType,
        )

        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        if expected_error:
            with pytest.raises(expected_error):
                AnalysisRequest(
                    analysis_types=[AnalysisType.VALIDATION],
                    site_coordinates=(48.5, -123.5),
                    time_range=(start_dt, end_dt),
                )
        else:
            request = AnalysisRequest(
                analysis_types=[AnalysisType.VALIDATION],
                site_coordinates=(48.5, -123.5),
                time_range=(start_dt, end_dt),
            )
            assert request.time_range == (start_dt, end_dt)

    except ImportError:
        pytest.skip("Analytics framework not available")
