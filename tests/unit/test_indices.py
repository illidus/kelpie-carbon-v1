"""Tests for indices module."""

import numpy as np
import xarray as xr

from kelpie_carbon_v1.core.indices import (
    calculate_indices_from_dataset,
    floating_algae_index,
)


def test_fai_positive_when_nir_gt_re():
    """Test that FAI is positive when NIR > red edge."""
    re = np.array([0.1, 0.2])
    nir = np.array([0.3, 0.4])
    result = floating_algae_index(re, nir)
    assert np.all(result > 0)


def test_calculate_indices_from_dataset():
    """Test calculation of indices from xarray dataset."""
    # Create mock satellite dataset
    data = xr.Dataset(
        {
            "red": (["y", "x"], np.random.rand(10, 10)),
            "red_edge": (["y", "x"], np.random.rand(10, 10)),
            "nir": (["y", "x"], np.random.rand(10, 10) + 0.2),  # Higher NIR
            "swir1": (["y", "x"], np.random.rand(10, 10)),
        },
        coords={"x": np.linspace(-123.2, -123.1, 10), "y": np.linspace(49.2, 49.3, 10)},
    )

    indices = calculate_indices_from_dataset(data)

    # Check that indices were calculated
    assert "fai" in indices
    assert "ndre" in indices
    assert "kelp_index" in indices

    # Check dimensions match
    assert indices["fai"].shape == data["red"].shape
    assert indices["ndre"].shape == data["red"].shape

    # Check coordinates preserved
    assert "x" in indices.coords
    assert "y" in indices.coords
