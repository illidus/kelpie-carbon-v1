"""Tests for fetch module."""
import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch, MagicMock
from kelpie_carbon_v1.core.fetch import (
    fetch_sentinel_tiles,
    _create_mock_sentinel_data
)


def test_fetch_sentinel_tiles_with_mock_data():
    """Test fetch_sentinel_tiles returns mock data when no credentials."""
    # Use valid coordinates in BC, Canada
    lat, lng = 49.2827, -123.1207
    start_date, end_date = "2023-01-01", "2023-01-31"

    result = fetch_sentinel_tiles(lat, lng, start_date, end_date)

    # Check structure
    assert isinstance(result, dict)
    assert "data" in result
    assert "source" in result
    assert "bands" in result

    # Check data is xarray Dataset
    assert isinstance(result["data"], xr.Dataset)

    # Check required bands
    expected_bands = ["red", "red_edge", "nir", "swir1", "cloud_mask"]
    for band in expected_bands:
        assert band in result["data"].data_vars

    # Check data shapes are consistent
    data_vars = list(result["data"].data_vars.values())
    first_shape = data_vars[0].shape
    for var in data_vars:
        assert var.shape == first_shape

    # Check coordinate dimensions
    assert "x" in result["data"].coords
    assert "y" in result["data"].coords


def test_fetch_sentinel_tiles_invalid_coordinates():
    """Test fetch_sentinel_tiles with invalid coordinates."""
    with pytest.raises(ValueError, match="Latitude must be between"):
        fetch_sentinel_tiles(91.0, 0.0, "2023-01-01", "2023-01-31")

    with pytest.raises(ValueError, match="Longitude must be between"):
        fetch_sentinel_tiles(0.0, 181.0, "2023-01-01", "2023-01-31")


def test_fetch_sentinel_tiles_invalid_dates():
    """Test fetch_sentinel_tiles with invalid date formats."""
    with pytest.raises(ValueError, match="Invalid date format"):
        fetch_sentinel_tiles(49.0, -123.0, "2023/01/01", "2023-01-31")

    with pytest.raises(ValueError, match="Invalid date format"):
        fetch_sentinel_tiles(49.0, -123.0, "2023-01-01", "invalid-date")
