"""Spectral indices for kelp biomass estimation."""
import numpy as np


def floating_algae_index(red_edge: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute the Floating Algae Index (FAI).

    For now, implement as: FAI = nir - red_edge
    """
    return nir - red_edge
