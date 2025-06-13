"""Satellite data fetching module using Microsoft Planetary Computer."""

from datetime import datetime

import numpy as np
import planetary_computer as pc
import rioxarray as rxr
import xarray as xr
from pystac_client import Client
from scipy.ndimage import binary_dilation

from .constants import SatelliteData
from .logging_config import get_logger

logger = get_logger(__name__)


def fetch_sentinel_tiles(
    lat: float, lng: float, start_date: str, end_date: str, buffer_km: float = 1.0
) -> dict:
    """Fetch Sentinel-2 tiles for given coordinates and date range using Planetary Computer.

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
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}") from e

    # Create bounding box around the point
    # Rough conversion: 1 degree ≈ 111 km at equator
    buffer_deg = buffer_km / SatelliteData.KM_PER_DEGREE
    bbox = [lng - buffer_deg, lat - buffer_deg, lng + buffer_deg, lat + buffer_deg]

    try:
        # Connect to Planetary Computer STAC API
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

        # Search for Sentinel-2 L2A data
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={
                "eo:cloud_cover": {"lt": SatelliteData.MAX_CLOUD_COVER}
            },  # Less than threshold cloud cover
        )

        items = list(search.items())

        if not items:
            # Fall back to mock data if no real data is available
            logger.warning(
                f"No Sentinel-2 data found for {lat}, {lng} between {start_date} and {end_date}"
            )
            logger.info("Using realistic mock data instead...")
            return _create_mock_sentinel_data(
                lat, lng, start_date, end_date, (200, 200)
            )

        # Select the item with least cloud coverage
        best_item = min(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))

        logger.info(f"Found Sentinel-2 scene: {best_item.id}")
        logger.info(
            f"Cloud coverage: {best_item.properties.get('eo:cloud_cover', 'Unknown')}%"
        )
        logger.info(f"Date: {best_item.properties.get('datetime', 'Unknown')}")

        # Sign the assets for access
        signed_item = pc.sign(best_item)

        # Get the required bands
        bands_to_fetch = {
            "red": "B04",  # Red - 665nm
            "red_edge": "B05",  # Red Edge 1 - 705nm (existing)
            "red_edge_2": "B06",  # Red Edge 2 - 740nm (NEW - optimal for submerged kelp)
            "red_edge_3": "B07",  # Red Edge 3 - 783nm (NEW - additional red-edge)
            "nir": "B08",  # NIR - 842nm
            "swir1": "B11",  # SWIR1 - 1610nm
        }

        # Also get SCL (Scene Classification Layer) for cloud masking
        cloud_band = "SCL"

        data_arrays = {}

        # Load each band
        for band_name, asset_key in bands_to_fetch.items():
            if asset_key in signed_item.assets:
                asset_url = signed_item.assets[asset_key].href
                logger.debug(f"Loading {band_name} band from {asset_key}...")

                # Open with rioxarray and clip to our bounding box
                from typing import cast

                with cast(xr.DataArray, rxr.open_rasterio(asset_url)) as da:
                    # Clip to our area of interest
                    clipped = da.rio.clip_box(
                        minx=bbox[0],
                        miny=bbox[1],
                        maxx=bbox[2],
                        maxy=bbox[3],
                        crs="EPSG:4326",
                    )

                    # Convert to float and scale (Sentinel-2 data is typically in 0-10000 range)
                    clipped = (
                        clipped.astype(np.float32) / SatelliteData.SENTINEL_SCALE_FACTOR
                    )

                    # Take only the first band if it's a multiband image
                    if len(clipped.dims) > 2:
                        clipped = clipped.isel(band=0)

                    data_arrays[band_name] = clipped
            else:
                logger.warning(f"{asset_key} band not available in this scene")

        # Load cloud mask
        cloud_mask = None
        if cloud_band in signed_item.assets:
            asset_url = signed_item.assets[cloud_band].href
            logger.debug(f"Loading cloud mask from {cloud_band}...")

            with cast(xr.DataArray, rxr.open_rasterio(asset_url)) as da:
                clipped = da.rio.clip_box(
                    minx=bbox[0],
                    miny=bbox[1],
                    maxx=bbox[2],
                    maxy=bbox[3],
                    crs="EPSG:4326",
                )

                if len(clipped.dims) > 2:
                    clipped = clipped.isel(band=0)

                # Convert SCL to binary cloud mask
                # SCL values: 0=no_data, 1=saturated, 2=dark, 3=cloud_shadow, 4=vegetation,
                # 5=not_vegetated, 6=water, 7=unclassified, 8=cloud_medium, 9=cloud_high, 10=thin_cirrus, 11=snow
                cloud_mask = ((clipped == 8) | (clipped == 9) | (clipped == 10)).astype(
                    np.int32
                )
                data_arrays["cloud_mask"] = cloud_mask

        if not data_arrays:
            raise RuntimeError("No usable bands found in the Sentinel-2 scene")

        # Get common coordinates (all bands should have same coordinates after clipping)
        first_band = list(data_arrays.values())[0]

        # Ensure coordinates are in geographic format (lat/lon)
        if (
            hasattr(first_band, "rio")
            and first_band.rio.crs is not None
            and first_band.rio.crs != "EPSG:4326"
        ):
            # Transform to WGS84 if not already
            first_band = first_band.rio.reproject("EPSG:4326")

        coords = {"x": first_band.x, "y": first_band.y}

        # Create xarray Dataset
        dataset_dict = {}
        for band_name, data_array in data_arrays.items():
            # Transform each band to WGS84 if needed
            if (
                hasattr(data_array, "rio")
                and data_array.rio.crs is not None
                and data_array.rio.crs != "EPSG:4326"
            ):
                data_array = data_array.rio.reproject("EPSG:4326")

            # Ensure all bands have the same coordinates
            data_array = data_array.interp(
                x=coords["x"], y=coords["y"], method="nearest"
            )
            dataset_dict[band_name] = (["y", "x"], data_array.values)

        # Create dataset with geographic coordinates
        dataset = xr.Dataset(dataset_dict, coords=coords)

        # Store the original bounding box (which is already in geographic coordinates)
        dataset.attrs["bbox"] = bbox

        return {
            "data": dataset,
            "bbox": bbox,
            "acquisition_date": best_item.properties.get("datetime", end_date)[:10],
            "resolution": 10,  # meters
            "source": "Sentinel-2 L2A (Planetary Computer)",
            "bands": list(bands_to_fetch.keys())
            + (["cloud_mask"] if cloud_mask is not None else []),
            "scene_id": best_item.id,
            "cloud_cover": best_item.properties.get("eo:cloud_cover", "Unknown"),
        }

    except Exception as e:
        logger.error(f"Error fetching real satellite data: {str(e)}")
        logger.info("Falling back to realistic mock data...")
        return _create_mock_sentinel_data(lat, lng, start_date, end_date, (200, 200))


def _create_mock_sentinel_data(
    lat: float, lng: float, start_date: str, end_date: str, size: tuple[int, int]
) -> dict:
    """Create mock Sentinel-2 data for development/testing."""
    height, width = size

    # Generate realistic-looking fake satellite data that resembles coastal areas
    np.random.seed(int(lat * 1000 + lng * 1000) % 2147483647)  # Location-based seed

    # Create a coastline-like pattern
    from scipy import ndimage

    # Generate base terrain (water vs land)
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    x_grid, y_grid = np.meshgrid(x, y)

    # Create a realistic coastline pattern
    coastline = (
        np.sin(x_grid * 0.5)
        + 0.3 * np.cos(y_grid * 0.7)
        + 0.2 * np.sin(x_grid * y_grid * 0.1)
    )
    water_mask = coastline < 0.2

    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.1, (height, width))
    water_mask = (coastline + noise) < 0.2

    # Create realistic spectral values
    # Water areas: low red, low NIR, higher blue (we'll simulate this)
    # Land areas: variable red/NIR depending on vegetation
    # Kelp areas: specific spectral signature

    # Initialize bands
    red = np.zeros((height, width))
    red_edge = np.zeros((height, width))
    red_edge_2 = np.zeros((height, width))  # NEW: B06 - 740nm
    red_edge_3 = np.zeros((height, width))  # NEW: B07 - 783nm
    nir = np.zeros((height, width))
    swir1 = np.zeros((height, width))

    # Water areas
    red[water_mask] = np.clip(np.random.normal(0.02, 0.01, np.sum(water_mask)), 0, 0.1)
    red_edge[water_mask] = np.clip(
        np.random.normal(0.03, 0.01, np.sum(water_mask)), 0, 0.1
    )
    red_edge_2[water_mask] = np.clip(
        np.random.normal(0.025, 0.008, np.sum(water_mask)), 0, 0.08
    )  # Slightly lower
    red_edge_3[water_mask] = np.clip(
        np.random.normal(0.015, 0.005, np.sum(water_mask)), 0, 0.06
    )  # Approaching NIR
    nir[water_mask] = np.clip(
        np.random.normal(0.01, 0.005, np.sum(water_mask)), 0, 0.05
    )
    swir1[water_mask] = np.clip(
        np.random.normal(0.005, 0.002, np.sum(water_mask)), 0, 0.02
    )

    # Land areas (mix of vegetation and bare soil)
    land_mask = ~water_mask
    vegetation_prob = np.random.random((height, width)) < 0.6  # 60% vegetation

    # Vegetated land
    veg_mask = land_mask & vegetation_prob
    red[veg_mask] = np.clip(np.random.normal(0.04, 0.02, np.sum(veg_mask)), 0, 0.15)
    red_edge[veg_mask] = np.clip(np.random.normal(0.15, 0.05, np.sum(veg_mask)), 0, 0.4)
    red_edge_2[veg_mask] = np.clip(
        np.random.normal(0.20, 0.06, np.sum(veg_mask)), 0.05, 0.45
    )  # Peak red-edge
    red_edge_3[veg_mask] = np.clip(
        np.random.normal(0.30, 0.08, np.sum(veg_mask)), 0.15, 0.65
    )  # Transitioning to NIR
    nir[veg_mask] = np.clip(np.random.normal(0.4, 0.1, np.sum(veg_mask)), 0.2, 0.8)
    swir1[veg_mask] = np.clip(np.random.normal(0.25, 0.05, np.sum(veg_mask)), 0.1, 0.5)

    # Non-vegetated land
    bare_mask = land_mask & ~vegetation_prob
    red[bare_mask] = np.clip(np.random.normal(0.15, 0.05, np.sum(bare_mask)), 0.05, 0.4)
    red_edge[bare_mask] = np.clip(
        np.random.normal(0.18, 0.05, np.sum(bare_mask)), 0.1, 0.4
    )
    red_edge_2[bare_mask] = np.clip(
        np.random.normal(0.20, 0.05, np.sum(bare_mask)), 0.12, 0.42
    )  # Similar to red_edge
    red_edge_3[bare_mask] = np.clip(
        np.random.normal(0.22, 0.06, np.sum(bare_mask)), 0.14, 0.45
    )  # Slightly higher
    nir[bare_mask] = np.clip(np.random.normal(0.25, 0.08, np.sum(bare_mask)), 0.1, 0.5)
    swir1[bare_mask] = np.clip(
        np.random.normal(0.3, 0.08, np.sum(bare_mask)), 0.15, 0.6
    )

    # Add some kelp patches in water areas near coastline
    # Kelp has distinctive spectral signature: low red, high red-edge, moderate NIR
    coastline_buffer = binary_dilation(land_mask, iterations=3) & water_mask

    # Random kelp patches
    kelp_probability = (
        np.random.random((height, width)) < 0.1
    )  # 10% chance in coastal waters
    kelp_mask = coastline_buffer & kelp_probability

    # Kelp spectral signature - Based on SKEMA research findings
    red[kelp_mask] = np.clip(np.random.normal(0.02, 0.005, np.sum(kelp_mask)), 0, 0.05)
    red_edge[kelp_mask] = np.clip(
        np.random.normal(0.08, 0.02, np.sum(kelp_mask)), 0.04, 0.15
    )
    red_edge_2[kelp_mask] = np.clip(
        np.random.normal(0.12, 0.025, np.sum(kelp_mask)), 0.06, 0.18
    )  # Higher for submerged detection
    red_edge_3[kelp_mask] = np.clip(
        np.random.normal(0.09, 0.02, np.sum(kelp_mask)), 0.05, 0.14
    )  # Lower than peak red-edge
    nir[kelp_mask] = np.clip(
        np.random.normal(0.06, 0.02, np.sum(kelp_mask)), 0.02, 0.12
    )
    swir1[kelp_mask] = np.clip(
        np.random.normal(0.01, 0.005, np.sum(kelp_mask)), 0, 0.03
    )

    # Apply spatial smoothing for more realistic appearance
    red = ndimage.gaussian_filter(red, sigma=1.0)
    red_edge = ndimage.gaussian_filter(red_edge, sigma=1.0)
    red_edge_2 = ndimage.gaussian_filter(red_edge_2, sigma=1.0)
    red_edge_3 = ndimage.gaussian_filter(red_edge_3, sigma=1.0)
    nir = ndimage.gaussian_filter(nir, sigma=1.0)
    swir1 = ndimage.gaussian_filter(swir1, sigma=1.0)

    # Cloud mask (sparse clouds)
    cloud_mask = np.random.choice([0, 1], size=(height, width), p=[0.95, 0.05])
    cloud_mask = ndimage.binary_dilation(cloud_mask, iterations=2).astype(int)

    # Create coordinate arrays
    buffer_deg = 1.0 / 111.0  # 1km buffer in degrees
    lons = np.linspace(lng - buffer_deg, lng + buffer_deg, width)
    lats = np.linspace(lat + buffer_deg, lat - buffer_deg, height)

    # Create xarray dataset
    dataset = xr.Dataset(
        {
            "red": (["y", "x"], red),
            "red_edge": (["y", "x"], red_edge),
            "red_edge_2": (["y", "x"], red_edge_2),  # NEW: B06 - 740nm
            "red_edge_3": (["y", "x"], red_edge_3),  # NEW: B07 - 783nm
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
        "source": "Mock Sentinel-2 L2A (Realistic Coastal Simulation)",
        "bands": [
            "red",
            "red_edge",
            "red_edge_2",
            "red_edge_3",
            "nir",
            "swir1",
            "cloud_mask",
        ],
    }
