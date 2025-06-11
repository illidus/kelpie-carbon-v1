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

    # Normalized Difference Red-Edge (NDRE) - SKEMA Enhanced Formula
    # Uses proper NDRE formula: (Red_Edge - Red) / (Red_Edge + Red)
    # Prioritizes 740nm band (red_edge_2) for optimal submerged kelp detection
    if "red" in dataset:
        if "red_edge_2" in dataset:
            # Use optimal 740nm red-edge band
            indices["ndre"] = (dataset["red_edge_2"] - dataset["red"]) / (
                dataset["red_edge_2"] + dataset["red"]
            )
        elif "red_edge" in dataset:
            # Fallback to 705nm red-edge band
            indices["ndre"] = (dataset["red_edge"] - dataset["red"]) / (
                dataset["red_edge"] + dataset["red"]
            )

    # Traditional NDVI for comparison with NDRE
    if "red" in dataset and "nir" in dataset:
        indices["ndvi"] = (dataset["nir"] - dataset["red"]) / (
            dataset["nir"] + dataset["red"]
        )

    # Red Edge NDVI (traditional formula)
    if "red_edge" in dataset and "nir" in dataset:
        indices["red_edge_ndvi"] = (dataset["nir"] - dataset["red_edge"]) / (
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
