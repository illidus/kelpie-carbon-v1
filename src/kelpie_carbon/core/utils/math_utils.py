"""Mathematical utilities for Kelpie Carbon v1."""

import math

import numpy as np


def calculate_area_from_pixels(pixel_count: int, pixel_size_meters: float) -> float:
    """Calculate area in square meters from pixel count.

    Args:
        pixel_count: Number of pixels
        pixel_size_meters: Size of each pixel in meters

    Returns:
        Area in square meters
    """
    return pixel_count * (pixel_size_meters**2)


def convert_coordinates(
    lat: float, lng: float, from_crs: str = "EPSG:4326", to_crs: str = "EPSG:3857"
) -> tuple[float, float]:
    """Convert coordinates between coordinate reference systems.

    Args:
        lat: Latitude
        lng: Longitude
        from_crs: Source CRS (default: WGS84)
        to_crs: Target CRS (default: Web Mercator)

    Returns:
        Tuple of converted coordinates (x, y)

    Note:
        This is a simplified implementation. For production use,
        consider using proper GIS libraries like pyproj.
    """
    if from_crs == "EPSG:4326" and to_crs == "EPSG:3857":
        # WGS84 to Web Mercator conversion
        x = lng * 20037508.34 / 180
        y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
        y = y * 20037508.34 / 180
        return x, y

    elif from_crs == "EPSG:3857" and to_crs == "EPSG:4326":
        # Web Mercator to WGS84 conversion
        lng = lat * 180 / 20037508.34
        lat = math.atan(math.exp(lng * math.pi / 180)) * 360 / math.pi - 90
        return lat, lng

    else:
        # For other conversions, return as-is (should use pyproj in production)
        return lat, lng


def calculate_distance(
    lat1: float, lng1: float, lat2: float, lng2: float, method: str = "haversine"
) -> float:
    """Calculate distance between two points on Earth.

    Args:
        lat1, lng1: First point coordinates
        lat2, lng2: Second point coordinates
        method: Distance calculation method ('haversine', 'euclidean')

    Returns:
        Distance in meters
    """
    if method == "haversine":
        return _haversine_distance(lat1, lng1, lat2, lng2)
    elif method == "euclidean":
        return _euclidean_distance(lat1, lng1, lat2, lng2)
    else:
        raise ValueError(f"Unsupported distance method: {method}")


def _haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate Haversine distance between two points."""
    # Earth's radius in meters
    R = 6371000

    # Convert to radians
    lat1_rad = math.radians(lat1)
    lng1_rad = math.radians(lng1)
    lat2_rad = math.radians(lat2)
    lng2_rad = math.radians(lng2)

    # Differences
    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad

    # Haversine formula
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def _euclidean_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate approximate Euclidean distance between two points."""
    # Approximate conversion: 1 degree = 111,319 meters
    lat_meters = (lat2 - lat1) * 111319
    lng_meters = (lng2 - lng1) * 111319 * math.cos(math.radians((lat1 + lat2) / 2))

    return math.sqrt(lat_meters**2 + lng_meters**2)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Generate a 2D Gaussian kernel.

    Args:
        size: Kernel size (should be odd)
        sigma: Standard deviation

    Returns:
        2D Gaussian kernel array
    """
    if size % 2 == 0:
        size += 1  # Ensure odd size

    # Create coordinate grids
    center = size // 2
    x, y = np.mgrid[0:size, 0:size]
    x = x - center
    y = y - center

    # Calculate Gaussian values
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalize to sum to 1
    return g / g.sum()
