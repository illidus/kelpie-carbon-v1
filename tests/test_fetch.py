"""Tests for fetch module."""
import pytest

from kelpie_carbon_v1.fetch import fetch_sentinel_tiles


def test_fetch_sentinel_tiles_raises_not_implemented():
    """Test that fetch_sentinel_tiles raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        fetch_sentinel_tiles("test_aoi", "2023-01-01")
