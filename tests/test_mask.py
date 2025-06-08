"""Tests for mask module."""
import pytest

from kelpie_carbon_v1.mask import apply_mask


def test_apply_mask_raises_not_implemented():
    """Test that apply_mask raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        apply_mask("test_raster")
