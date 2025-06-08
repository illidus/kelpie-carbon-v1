"""Satellite data fetching module."""
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)


def configure_sentinelhub() -> SHConfig:
    """Configure SentinelHub with credentials from environment."""
    config = SHConfig()

    # Try to get credentials from environment variables
    client_id = os.getenv("SENTINELHUB_CLIENT_ID")
    client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET")

    if client_id and client_secret:
        config.sh_client_id = client_id
        config.sh_client_secret = client_secret
    else:
        # For development/testing, use a mock configuration
        config.sh_client_id = "mock_client_id"
        config.sh_client_secret = "mock_client_secret"  # nosec

    return config


def create_evalscript() -> str:
    """Create evalscript for Sentinel-2 bands needed for kelp analysis."""
    return """
    //VERSION=3
    function setup() {
        return {
            input: ["B04", "B05", "B08", "B11", "CLM"],
            output: { bands: 5 }
        };
    }

    function evaluatePixel(sample) {
        // Return Red, Red Edge, NIR, SWIR1, and Cloud mask
        return [sample.B04, sample.B05, sample.B08, sample.B11, sample.CLM];
    }
    """


def fetch_sentinel_tiles(
    lat: float, lng: float, start_date: str, end_date: str, buffer_km: float = 1.0
) -> Dict:
    """Fetch Sentinel-2 tiles for given coordinates and date range.

    Args:
        lat: Latitude of center point
        lng: Longitude of center point
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        buffer_km: Buffer around point in kilometers

    Returns:
        Dictionary containing satellite data and metadata

    Raises:
        ValueError: If coordinates or dates are invalid
        RuntimeError: If data fetch fails
    """
    # Validate inputs
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
    if not (-180 <= lng <= 180):
        raise ValueError(f"Longitude must be between -180 and 180, got {lng}")

    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

    # Create bounding box around the point
    # Rough conversion: 1 degree ≈ 111 km at equator
    buffer_deg = buffer_km / 111.0

    bbox = BBox(
        bbox=[lng - buffer_deg, lat - buffer_deg, lng + buffer_deg, lat + buffer_deg],
        crs=CRS.WGS84,
    )

    # Calculate optimal resolution
    bbox_size = bbox_to_dimensions(bbox, resolution=10)  # 10m resolution

    # Configure SentinelHub
    config = configure_sentinelhub()

    # Check if we have real credentials
    if config.sh_client_id == "mock_client_id":
        # Return mock data for development
        return _create_mock_sentinel_data(lat, lng, start_date, end_date, bbox_size)

    # Create request for real data
    evalscript = create_evalscript()

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(start_date, end_date),
                mosaicking_order="leastCC",  # Least cloud coverage
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=bbox_size,
        config=config,
    )

    try:
        # Execute request
        data = request.get_data()

        if not data or len(data) == 0:
            raise RuntimeError(
                "No satellite data returned for the specified area and time"
            )

        # Convert to xarray Dataset
        image_array = data[0]  # First (and only) response

        # Create coordinate arrays
        lons = np.linspace(bbox.min_x, bbox.max_x, image_array.shape[1])
        lats = np.linspace(bbox.max_y, bbox.min_y, image_array.shape[0])

        # Create xarray dataset
        dataset = xr.Dataset(
            {
                "red": (["y", "x"], image_array[:, :, 0]),
                "red_edge": (["y", "x"], image_array[:, :, 1]),
                "nir": (["y", "x"], image_array[:, :, 2]),
                "swir1": (["y", "x"], image_array[:, :, 3]),
                "cloud_mask": (["y", "x"], image_array[:, :, 4]),
            },
            coords={"x": lons, "y": lats},
        )

        return {
            "data": dataset,
            "bbox": bbox,
            "acquisition_date": end_date,  # Use end date as representative
            "resolution": 10,  # meters
            "source": "Sentinel-2 L2A",
            "bands": ["red", "red_edge", "nir", "swir1", "cloud_mask"],
        }

    except Exception as e:
        raise RuntimeError(f"Failed to fetch satellite data: {str(e)}")


def _create_mock_sentinel_data(
    lat: float, lng: float, start_date: str, end_date: str, size: Tuple[int, int]
) -> Dict:
    """Create mock Sentinel-2 data for development/testing."""
    height, width = size

    # Generate realistic-looking fake satellite data
    np.random.seed(42)  # Reproducible fake data

    # Create mock spectral data (scaled to typical Sentinel-2 values)
    red = np.random.normal(0.1, 0.05, (height, width)).clip(0, 1)
    red_edge = np.random.normal(0.15, 0.05, (height, width)).clip(0, 1)
    nir = np.random.normal(0.3, 0.1, (height, width)).clip(0, 1)
    swir1 = np.random.normal(0.2, 0.05, (height, width)).clip(0, 1)
    cloud_mask = np.random.choice([0, 1], size=(height, width), p=[0.8, 0.2])

    # Add some spatial correlation (kelp-like patterns)
    from scipy import ndimage

    red = ndimage.gaussian_filter(red, sigma=2)
    red_edge = ndimage.gaussian_filter(red_edge, sigma=2)
    nir = ndimage.gaussian_filter(nir, sigma=2)

    # Create coordinate arrays
    buffer_deg = 1.0 / 111.0  # 1km buffer in degrees
    lons = np.linspace(lng - buffer_deg, lng + buffer_deg, width)
    lats = np.linspace(lat + buffer_deg, lat - buffer_deg, height)

    # Create xarray dataset
    dataset = xr.Dataset(
        {
            "red": (["y", "x"], red),
            "red_edge": (["y", "x"], red_edge),
            "nir": (["y", "x"], nir),
            "swir1": (["y", "x"], swir1),
            "cloud_mask": (["y", "x"], cloud_mask),
        },
        coords={"x": lons, "y": lats},
    )

    return {
        "data": dataset,
        "bbox": f"POINT({lng} {lat}) +/- 1km",
        "acquisition_date": end_date,
        "resolution": 10,
        "source": "Mock Sentinel-2 L2A (Development)",
        "bands": ["red", "red_edge", "nir", "swir1", "cloud_mask"],
    }
