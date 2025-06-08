"""Spectral index calculations for kelp detection."""
from typing import Union

import numpy as np
import xarray as xr


def floating_algae_index(
    red_edge: Union[np.ndarray, xr.DataArray], nir: Union[np.ndarray, xr.DataArray]
) -> Union[np.ndarray, xr.DataArray]:
    """Calculate Floating Algae Index (FAI).

    FAI is useful for detecting kelp and other floating vegetation.

    Args:
        red_edge: Red edge band values (Sentinel-2 B05)
        nir: Near-infrared band values (Sentinel-2 B08)

    Returns:
        FAI values as same type as input
    """
    return nir - red_edge


def calculate_indices_from_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Calculate all spectral indices from satellite dataset.

    Args:
        dataset: xarray Dataset with satellite bands

    Returns:
        Dataset with computed indices
    """
    indices = xr.Dataset()

    # Floating Algae Index (FAI)
    if "red_edge" in dataset and "nir" in dataset:
        indices["fai"] = floating_algae_index(dataset["red_edge"], dataset["nir"])

    # Normalized Difference Red-Edge (NDRE)
    if "red_edge" in dataset and "nir" in dataset:
        indices["ndre"] = (dataset["nir"] - dataset["red_edge"]) / (
            dataset["nir"] + dataset["red_edge"]
        )

    # Simple Kelp Index (combining spectral properties)
    if all(band in dataset for band in ["red", "red_edge", "nir"]):
        # Kelp typically has low red, moderate red-edge, high NIR
        indices["kelp_index"] = (dataset["nir"] - dataset["red"]) / (
            dataset["red_edge"] + 0.001
        )

    # Add coordinates from original dataset
    indices.coords.update(dataset.coords)

    return indices
