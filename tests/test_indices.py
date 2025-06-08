"""Tests for indices module."""
import numpy as np

from kelpie_carbon_v1.indices import floating_algae_index


def test_fai_positive_when_nir_gt_re():
    """Test that FAI is positive when NIR > red edge."""
    re = np.array([0.1, 0.2])
    nir = np.array([0.3, 0.4])
    result = floating_algae_index(re, nir)
    assert np.all(result > 0)
