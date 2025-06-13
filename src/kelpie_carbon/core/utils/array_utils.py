"""Array manipulation utilities for Kelpie Carbon v1."""

import numpy as np
from scipy import interpolate


def normalize_array(arr: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize array values using various methods.

    Args:
        arr: Input array to normalize
        method: Normalization method ('minmax', 'zscore', 'robust')

    Returns:
        Normalized array

    Raises:
        ValueError: If method is not supported

    """
    if method == "minmax":
        arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
        if arr_max == arr_min:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)

    elif method == "zscore":
        arr_mean, arr_std = np.nanmean(arr), np.nanstd(arr)
        if arr_std == 0:
            return np.zeros_like(arr)
        return (arr - arr_mean) / arr_std

    elif method == "robust":
        median = np.nanmedian(arr)
        mad = np.nanmedian(np.abs(arr - median))
        if mad == 0:
            return np.zeros_like(arr)
        return (arr - median) / (1.4826 * mad)

    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def clip_array_percentiles(
    arr: np.ndarray, lower_percentile: float = 2.0, upper_percentile: float = 98.0
) -> np.ndarray:
    """Clip array values to specified percentiles.

    Args:
        arr: Input array
        lower_percentile: Lower percentile for clipping
        upper_percentile: Upper percentile for clipping

    Returns:
        Clipped array

    """
    lower_val = np.nanpercentile(arr, lower_percentile)
    upper_val = np.nanpercentile(arr, upper_percentile)
    return np.clip(arr, lower_val, upper_val)


def calculate_statistics(arr: np.ndarray) -> dict:
    """Calculate comprehensive statistics for an array.

    Args:
        arr: Input array

    Returns:
        Dictionary containing various statistics

    """
    return {
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "std": float(np.nanstd(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "count": int(np.sum(~np.isnan(arr))),
        "nan_count": int(np.sum(np.isnan(arr))),
        "percentile_25": float(np.nanpercentile(arr, 25)),
        "percentile_75": float(np.nanpercentile(arr, 75)),
        "percentile_90": float(np.nanpercentile(arr, 90)),
        "percentile_95": float(np.nanpercentile(arr, 95)),
    }


def safe_divide(
    numerator: np.ndarray, denominator: np.ndarray, fill_value: float = 0.0
) -> np.ndarray:
    """Safely divide arrays, handling division by zero.

    Args:
        numerator: Numerator array
        denominator: Denominator array
        fill_value: Value to use when denominator is zero

    Returns:
        Result of division with safe handling

    """
    # Create output array
    result = np.full_like(numerator, fill_value, dtype=np.float64)

    # Find valid (non-zero) denominators
    valid_mask = (denominator != 0) & ~np.isnan(denominator) & ~np.isnan(numerator)

    # Perform division only where valid
    result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    return result


def interpolate_missing_values(
    arr: np.ndarray, method: str = "linear", axis: int | None = None
) -> np.ndarray:
    """Interpolate missing (NaN) values in array.

    Args:
        arr: Input array with potential NaN values
        method: Interpolation method ('linear', 'cubic', 'nearest')
        axis: Axis along which to interpolate (None for 1D)

    Returns:
        Array with interpolated values

    Raises:
        ValueError: If method is not supported

    """
    if not np.any(np.isnan(arr)):
        return arr.copy()

    if arr.ndim == 1 or axis is not None:
        # 1D interpolation or along specific axis
        result = arr.copy()

        if axis is None:
            # 1D case
            valid_mask = ~np.isnan(arr)
            if not np.any(valid_mask):
                return arr  # All NaN, cannot interpolate

            if np.sum(valid_mask) < 2:
                # Fill with single valid value
                result[~valid_mask] = arr[valid_mask][0]
                return result

            valid_indices = np.where(valid_mask)[0]
            valid_values = arr[valid_mask]

            if method == "linear":
                f = interpolate.interp1d(
                    valid_indices,
                    valid_values,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
            elif method == "cubic":
                if len(valid_values) >= 4:
                    f = interpolate.interp1d(
                        valid_indices,
                        valid_values,
                        kind="cubic",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                else:
                    # Fall back to linear for insufficient points
                    f = interpolate.interp1d(
                        valid_indices,
                        valid_values,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
            elif method == "nearest":
                f = interpolate.interp1d(
                    valid_indices,
                    valid_values,
                    kind="nearest",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
            else:
                raise ValueError(f"Unsupported interpolation method: {method}")

            all_indices = np.arange(len(arr))
            result = f(all_indices)

        else:
            # Interpolation along specific axis
            result = np.apply_along_axis(
                lambda x: interpolate_missing_values(x, method=method), axis, arr
            )

    else:
        # 2D interpolation
        if method not in ["linear", "cubic", "nearest"]:
            raise ValueError(f"Unsupported 2D interpolation method: {method}")

        result = arr.copy()
        nan_mask = np.isnan(arr)

        if not np.any(nan_mask):
            return result

        # Get coordinates of valid points
        valid_mask = ~nan_mask
        if not np.any(valid_mask):
            return arr  # All NaN

        rows, cols = np.mgrid[0 : arr.shape[0], 0 : arr.shape[1]]
        valid_points = np.column_stack((rows[valid_mask], cols[valid_mask]))
        valid_values = arr[valid_mask]

        # Points to interpolate
        nan_points = np.column_stack((rows[nan_mask], cols[nan_mask]))

        if len(valid_values) < 3:
            # Not enough points for 2D interpolation
            result[nan_mask] = np.nanmean(valid_values)
        else:
            # Use griddata for 2D interpolation
            from scipy.interpolate import griddata

            interpolated = griddata(
                valid_points,
                valid_values,
                nan_points,
                method=(
                    method if method != "cubic" else "linear"
                ),  # griddata doesn't support cubic
            )
            result[nan_mask] = interpolated

    return result
