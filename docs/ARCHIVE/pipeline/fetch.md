# Fetch Module

This module handles satellite data fetching for the Kelpie Carbon pipeline using real Sentinel-2 data.

## Functions

### `fetch_sentinel_tiles(lat, lng, start_date, end_date, buffer_km=1.0)`

Downloads Sentinel-2 satellite imagery for the specified coordinates and date range.

**Status**: ✅ Implemented with SentinelHub integration

**Parameters**:
- `lat`: Latitude of center point (-90 to 90)
- `lng`: Longitude of center point (-180 to 180)
- `start_date`: Start date in YYYY-MM-DD format
- `end_date`: End date in YYYY-MM-DD format
- `buffer_km`: Buffer around point in kilometers (default: 1.0)

**Returns**:
- Dictionary containing:
  - `data`: xarray Dataset with satellite bands
  - `bbox`: Bounding box information
  - `acquisition_date`: Date of imagery
  - `resolution`: Spatial resolution in meters
  - `source`: Data source identifier
  - `bands`: List of available bands

**Bands Retrieved**:
- `red`: Red band (B04) - 665nm
- `red_edge`: Red edge band (B05) - 705nm
- `nir`: Near-infrared band (B08) - 842nm
- `swir1`: Short-wave infrared band (B11) - 1610nm
- `cloud_mask`: Cloud mask (CLM)

**Authentication**:
- Uses environment variables `SENTINELHUB_CLIENT_ID` and `SENTINELHUB_CLIENT_SECRET`
- Falls back to mock data for development when credentials not available

**Implementation Notes**:
- Integrates with SentinelHub API for real Sentinel-2 L2A data
- Handles coordinate validation and date format checking
- Generates realistic mock data for development/testing
- Uses least cloud coverage mosaicking for optimal imagery
- 10m spatial resolution for detailed kelp analysis
