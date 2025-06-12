# Indices Module

This module calculates spectral indices from satellite data for kelp detection and analysis.

## Functions

### `floating_algae_index(red_edge, nir)`

Calculates the Floating Algae Index (FAI) for kelp detection.

**Status**: ✅ Implemented

**Parameters**:
- `red_edge`: Red edge band values (Sentinel-2 B05) - numpy array or xarray DataArray
- `nir`: Near-infrared band values (Sentinel-2 B08) - numpy array or xarray DataArray

**Returns**:
- FAI values as same type as input (numpy array or xarray DataArray)

**Formula**: `FAI = NIR - Red Edge`

### `calculate_indices_from_dataset(dataset)`

Calculates multiple spectral indices from an xarray Dataset containing satellite bands.

**Status**: ✅ Implemented

**Parameters**:
- `dataset`: xarray Dataset with satellite bands (red, red_edge, nir, etc.)

**Returns**:
- xarray Dataset containing calculated indices

**Calculated Indices**:
- `fai`: Floating Algae Index (NIR - Red Edge)
- `ndre`: Normalized Difference Red-Edge ((NIR - Red Edge) / (NIR + Red Edge))
- `kelp_index`: Simple Kelp Index ((NIR - Red) / (Red Edge + 0.001))

**Implementation Notes**:
- Supports both numpy arrays and xarray DataArrays
- Preserves coordinate information from input dataset
- Handles missing bands gracefully
- Optimized for kelp forest detection in coastal waters
