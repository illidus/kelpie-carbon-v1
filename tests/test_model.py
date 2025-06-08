"""Tests for model module."""
import pytest

from kelpie_carbon_v1.model import predict_biomass


def test_predict_biomass_raises_not_implemented():
    """Test that predict_biomass raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        predict_biomass("test_indices_array")
