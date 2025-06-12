"""Validation utilities for Kelpie Carbon v1."""

import re
from datetime import datetime
from typing import Any

import xarray as xr


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


def validate_coordinates(
    lat: float,
    lng: float,
    lat_bounds: tuple[float, float] = (-90.0, 90.0),
    lng_bounds: tuple[float, float] = (-180.0, 180.0),
) -> bool:
    """Validate latitude and longitude coordinates.

    Args:
        lat: Latitude value
        lng: Longitude value
        lat_bounds: Valid latitude range (min, max)
        lng_bounds: Valid longitude range (min, max)

    Returns:
        True if coordinates are valid

    Raises:
        ValidationError: If coordinates are invalid
    """
    if not isinstance(lat, (int, float)):
        raise ValidationError(f"Latitude must be numeric, got {type(lat)}")

    if not isinstance(lng, (int, float)):
        raise ValidationError(f"Longitude must be numeric, got {type(lng)}")

    lat_min, lat_max = lat_bounds
    lng_min, lng_max = lng_bounds

    if not (lat_min <= lat <= lat_max):
        raise ValidationError(f"Latitude {lat} out of bounds [{lat_min}, {lat_max}]")

    if not (lng_min <= lng <= lng_max):
        raise ValidationError(f"Longitude {lng} out of bounds [{lng_min}, {lng_max}]")

    return True


def validate_date_range(
    start_date: str | datetime, end_date: str | datetime, max_days: int | None = None
) -> tuple[datetime, datetime]:
    """Validate and parse date range.

    Args:
        start_date: Start date string (YYYY-MM-DD) or datetime
        end_date: End date string (YYYY-MM-DD) or datetime
        max_days: Maximum allowed days between dates

    Returns:
        Tuple of parsed start and end datetime objects

    Raises:
        ValidationError: If dates are invalid or out of range
    """
    # Parse dates if they're strings
    if isinstance(start_date, str):
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise ValidationError(
                f"Invalid start date format: {start_date}. Expected YYYY-MM-DD"
            )
    else:
        start_dt = start_date

    if isinstance(end_date, str):
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValidationError(
                f"Invalid end date format: {end_date}. Expected YYYY-MM-DD"
            )
    else:
        end_dt = end_date

    # Validate date order
    if start_dt >= end_dt:
        raise ValidationError(
            f"Start date {start_dt.date()} must be before end date {end_dt.date()}"
        )

    # Check maximum duration if specified
    if max_days is not None:
        duration = (end_dt - start_dt).days
        if duration > max_days:
            raise ValidationError(
                f"Date range ({duration} days) exceeds maximum allowed ({max_days} days)"
            )

    # Check that dates are not in the future
    now = datetime.now()
    if start_dt > now:
        raise ValidationError(f"Start date {start_dt.date()} cannot be in the future")

    if end_dt > now:
        raise ValidationError(f"End date {end_dt.date()} cannot be in the future")

    return start_dt, end_dt


def validate_dataset_bands(
    dataset: xr.Dataset,
    required_bands: list[str],
    optional_bands: list[str] | None = None,
) -> dict[str, bool]:
    """Validate that dataset contains required spectral bands.

    Args:
        dataset: xarray Dataset to validate
        required_bands: List of required band names
        optional_bands: List of optional band names

    Returns:
        Dictionary mapping band names to availability

    Raises:
        ValidationError: If required bands are missing
    """
    available_bands = list(dataset.data_vars.keys())
    band_status = {}
    missing_required = []

    # Check required bands
    for band in required_bands:
        is_available = band in available_bands
        band_status[band] = is_available
        if not is_available:
            missing_required.append(band)

    # Check optional bands
    if optional_bands:
        for band in optional_bands:
            band_status[band] = band in available_bands

    # Raise error if required bands are missing
    if missing_required:
        raise ValidationError(
            f"Missing required bands: {missing_required}. "
            f"Available bands: {available_bands}"
        )

    return band_status


def validate_config_structure(
    config: dict[str, Any], schema: dict[str, Any], path: str = ""
) -> bool:
    """Validate configuration structure against a schema.

    Args:
        config: Configuration dictionary to validate
        schema: Schema dictionary defining expected structure
        path: Current path in nested structure (for error messages)

    Returns:
        True if configuration is valid

    Raises:
        ValidationError: If configuration doesn't match schema
    """
    for key, expected_type in schema.items():
        current_path = f"{path}.{key}" if path else key

        if key not in config:
            raise ValidationError(f"Missing required key: {current_path}")

        value = config[key]

        # Handle nested dictionaries
        if isinstance(expected_type, dict):
            if not isinstance(value, dict):
                raise ValidationError(
                    f"{current_path} must be a dictionary, got {type(value)}"
                )
            validate_config_structure(value, expected_type, current_path)

        # Handle type checking
        elif isinstance(expected_type, type):
            if not isinstance(value, expected_type):
                raise ValidationError(
                    f"{current_path} must be {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

        # Handle multiple allowed types
        elif isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                type_names = [t.__name__ for t in expected_type]
                raise ValidationError(
                    f"{current_path} must be one of {type_names}, "
                    f"got {type(value).__name__}"
                )

        # Handle callable validators
        elif callable(expected_type):
            try:
                if not expected_type(value):
                    raise ValidationError(f"{current_path} failed validation check")
            except Exception as e:
                raise ValidationError(f"{current_path} validation failed: {e}")

    return True


def validate_email(email: str) -> bool:
    """Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        True if email is valid

    Raises:
        ValidationError: If email format is invalid
    """
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    if not isinstance(email, str):
        raise ValidationError(f"Email must be a string, got {type(email)}")

    if not re.match(email_pattern, email):
        raise ValidationError(f"Invalid email format: {email}")

    return True


def validate_url(url: str, allowed_schemes: list[str] | None = None) -> bool:
    """Validate URL format.

    Args:
        url: URL to validate
        allowed_schemes: List of allowed URL schemes (default: ['http', 'https'])

    Returns:
        True if URL is valid

    Raises:
        ValidationError: If URL format is invalid
    """
    if allowed_schemes is None:
        allowed_schemes = ["http", "https"]

    if not isinstance(url, str):
        raise ValidationError(f"URL must be a string, got {type(url)}")

    # Basic URL pattern
    url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"

    if not re.match(url_pattern, url):
        raise ValidationError(f"Invalid URL format: {url}")

    # Check scheme
    scheme = url.split("://")[0].lower()
    if scheme not in allowed_schemes:
        raise ValidationError(
            f"URL scheme '{scheme}' not in allowed schemes: {allowed_schemes}"
        )

    return True


def validate_file_path(
    file_path: str,
    must_exist: bool = False,
    allowed_extensions: list[str] | None = None,
) -> bool:
    """Validate file path format and existence.

    Args:
        file_path: File path to validate
        must_exist: Whether the file must exist
        allowed_extensions: List of allowed file extensions

    Returns:
        True if file path is valid

    Raises:
        ValidationError: If file path is invalid
    """
    from pathlib import Path

    if not isinstance(file_path, str):
        raise ValidationError(f"File path must be a string, got {type(file_path)}")

    if not file_path.strip():
        raise ValidationError("File path cannot be empty")

    path = Path(file_path)

    # Check if file must exist
    if must_exist and not path.exists():
        raise ValidationError(f"File does not exist: {file_path}")

    # Check file extension
    if allowed_extensions:
        extension = path.suffix.lower()
        if extension not in allowed_extensions:
            raise ValidationError(
                f"File extension '{extension}' not in allowed extensions: {allowed_extensions}"
            )

    # Check for potentially dangerous paths
    try:
        path.resolve()
    except (OSError, ValueError) as e:
        raise ValidationError(f"Invalid file path: {e}")

    return True


def validate_numeric_range(
    value: float,
    min_value: float | None = None,
    max_value: float | None = None,
    name: str = "value",
) -> bool:
    """Validate that a numeric value is within specified range.

    Args:
        value: Numeric value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        name: Name of the value for error messages

    Returns:
        True if value is valid

    Raises:
        ValidationError: If value is out of range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(value)}")

    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} ({value}) must be >= {min_value}")

    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} ({value}) must be <= {max_value}")

    return True
