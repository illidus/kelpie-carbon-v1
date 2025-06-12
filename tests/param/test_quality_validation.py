"""
Consolidated quality validation tests.

Parameterized tests for all quality validation scenarios.
"""

import pytest


@pytest.mark.parametrize(
    "quality_value, expected_error, error_message",
    [
        # Invalid quality values
        ("invalid", ValueError, "Quality must be"),
        ("bad_quality", ValueError, "Quality must be"),
        ("unknown", ValueError, "Quality must be"),
        (None, ValueError, "Quality must be"),
        (123, ValueError, "Quality must be"),
        # Valid quality values
        ("high", None, None),
        ("medium", None, None),
        ("low", None, None),
    ],
)
def test_digitization_quality_validation(quality_value, expected_error, error_message):
    """Test digitization quality validation."""
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
                    historical_period=(1850, 1950),
                    data_sources=["Test Source"],
                    species=["Test Species"],
                    digitization_quality=quality_value,
                )
        else:
            site = HistoricalSite(
                name="Test Site",
                latitude=50.0,
                longitude=-125.0,
                region="Test Region",
                historical_period=(1850, 1950),
                data_sources=["Test Source"],
                species=["Test Species"],
                digitization_quality=quality_value,
            )
            assert site.digitization_quality == quality_value

    except ImportError:
        pytest.skip("Historical baseline analysis not available")
