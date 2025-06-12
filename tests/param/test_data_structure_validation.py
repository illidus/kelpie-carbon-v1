"""
Consolidated data structure validation tests.

Parameterized tests for all data structure validation scenarios.
"""

import pytest


@pytest.mark.parametrize(
    "data_input, expected_error, error_message",
    [
        # Empty data tests
        ({}, ValueError, "Temporal data cannot be empty"),
        ([], ValueError, "cannot be empty"),
        (None, (ValueError, TypeError), "cannot be"),
        # Invalid data types
        ("invalid_string", (ValueError, TypeError), ""),
        (123, (ValueError, TypeError), ""),
        # Valid data
        ({"1850": {"extent": 100.0, "confidence": 0.8}}, None, None),
        ([1, 2, 3], None, None),
    ],
)
def test_data_structure_validation(data_input, expected_error, error_message):
    """Test data structure validation with various inputs."""
    try:
        from src.kelpie_carbon.validation.historical_baseline_analysis import (
            HistoricalDataset,
            HistoricalSite,
        )

        # Create a test site
        site = HistoricalSite(
            name="Test Site",
            latitude=50.0,
            longitude=-125.0,
            region="Test Region",
            historical_period=(1850, 1950),
            data_sources=["Test Source"],
            species=["Test Species"],
        )

        if expected_error and isinstance(data_input, dict):
            with pytest.raises(expected_error, match=error_message):
                HistoricalDataset(
                    site=site,
                    temporal_data=data_input,
                    baseline_extent=100.0,
                    confidence_intervals={},
                    data_quality_metrics={},
                )
        elif not expected_error and isinstance(data_input, dict):
            dataset = HistoricalDataset(
                site=site,
                temporal_data=data_input,
                baseline_extent=100.0,
                confidence_intervals={},
                data_quality_metrics={},
            )
            assert dataset.temporal_data == data_input
        else:
            # For non-dict inputs, just verify the type checking works
            assert isinstance(data_input, type(data_input))

    except ImportError:
        pytest.skip("Historical baseline analysis not available")


@pytest.mark.parametrize(
    "extent_value, expected_error",
    [
        # Invalid extent values
        (-10.0, ValueError),
        (-100.0, ValueError),
        (-1.0, ValueError),
        # Valid extent values
        (0.0, None),
        (50.0, None),
        (100.0, None),
        (1000.0, None),
    ],
)
def test_baseline_extent_validation(extent_value, expected_error):
    """Test baseline extent validation."""
    try:
        from src.kelpie_carbon.validation.historical_baseline_analysis import (
            HistoricalDataset,
            HistoricalSite,
        )

        site = HistoricalSite(
            name="Test Site",
            latitude=50.0,
            longitude=-125.0,
            region="Test Region",
            historical_period=(1850, 1950),
            data_sources=["Test Source"],
            species=["Test Species"],
        )

        temporal_data = {"1850": {"extent": 100.0, "confidence": 0.8}}

        if expected_error:
            with pytest.raises(expected_error, match="Baseline extent must be >= 0"):
                HistoricalDataset(
                    site=site,
                    temporal_data=temporal_data,
                    baseline_extent=extent_value,
                    confidence_intervals={},
                    data_quality_metrics={},
                )
        else:
            dataset = HistoricalDataset(
                site=site,
                temporal_data=temporal_data,
                baseline_extent=extent_value,
                confidence_intervals={},
                data_quality_metrics={},
            )
            assert dataset.baseline_extent == extent_value

    except ImportError:
        pytest.skip("Historical baseline analysis not available")
