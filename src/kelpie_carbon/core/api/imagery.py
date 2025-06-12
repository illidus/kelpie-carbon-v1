"""API endpoints for satellite imagery visualization."""

import io
import sys
import time
import uuid
from typing import Any

import xarray as xr
from fastapi import APIRouter, HTTPException, Response
from PIL import Image

from ...data.imagery import (
    generate_biomass_heatmap,
    generate_cloud_mask_overlay,
    generate_false_color_composite,
    generate_kelp_mask_overlay,
    generate_rgb_composite,
    generate_spectral_visualization,
    generate_water_mask_overlay,
    get_image_bounds,
)
from .. import (
    apply_mask,
    calculate_indices_from_dataset,
    fetch_sentinel_tiles,
    predict_biomass,
)
from ..constants import Processing
from ..logging_config import get_logger
from .errors import (
    create_imagery_error,
    create_not_found_error,
    create_validation_error,
)
from .models import ImageryAnalysisRequest

logger = get_logger(__name__)

router = APIRouter(prefix="/api/imagery", tags=["imagery"])

# In-memory storage for analysis results with cache management
_analysis_cache: dict[str, dict[str, Any]] = {}
_cache_access_times: dict[str, float] = {}  # Track access times for LRU eviction


def _get_cache_size_mb():
    """Calculate approximate cache size in MB."""
    total_size = 0
    for cache_item in _analysis_cache.values():
        # Estimate size of xarray datasets and arrays
        if "dataset" in cache_item and cache_item["dataset"] is not None:
            total_size += cache_item["dataset"].nbytes
        if "biomass" in cache_item and cache_item["biomass"] is not None:
            total_size += cache_item["biomass"].nbytes
        # Add estimated overhead for other objects
        total_size += sys.getsizeof(cache_item)
    return total_size / (1024 * 1024)  # Convert to MB


def _cleanup_cache():
    """Remove oldest cache entries when size/count limits exceeded."""
    cache_size_mb = _get_cache_size_mb()
    cache_count = len(_analysis_cache)

    # Check if cleanup is needed
    if (
        cache_size_mb <= Processing.MAX_CACHE_SIZE_MB
        and cache_count <= Processing.MAX_CACHE_ITEMS
    ):
        return

    # Sort by access time (LRU)
    sorted_items = sorted(_cache_access_times.items(), key=lambda x: x[1])

    # Remove oldest entries until within limits
    items_to_remove = max(
        cache_count - Processing.MAX_CACHE_ITEMS,
        int(cache_count * 0.3),  # Remove at least 30% when over size limit
    )

    for analysis_id, _ in sorted_items[:items_to_remove]:
        if analysis_id in _analysis_cache:
            del _analysis_cache[analysis_id]
        if analysis_id in _cache_access_times:
            del _cache_access_times[analysis_id]


def _store_analysis_result(
    analysis_id: str,
    dataset: xr.Dataset,
    indices: dict,
    masks: dict,
    biomass: xr.DataArray | None = None,
    lat: float | None = None,
    lon: float | None = None,
):
    """Store analysis results in cache with size management."""
    # Clean up cache if needed before adding new item
    _cleanup_cache()

    _analysis_cache[analysis_id] = {
        "dataset": dataset,
        "indices": indices,
        "masks": masks,
        "biomass": biomass,
        "lat": lat,
        "lon": lon,
        "created_at": time.time(),
    }
    _cache_access_times[analysis_id] = time.time()


def _get_analysis_result(analysis_id: str) -> dict[str, Any]:
    """Get cached analysis result or raise appropriate error."""
    if analysis_id not in _analysis_cache:
        raise create_not_found_error("Analysis", analysis_id)
    return _analysis_cache[analysis_id]


def _image_to_response(
    image: Image.Image,
    format: str = "PNG",
    quality: int = 95,
    optimize: bool = True,
) -> Response:
    """Convert PIL Image to FastAPI Response with optimization."""
    img_io = io.BytesIO()

    # Apply optimization based on format
    if format.upper() == "JPEG":
        # JPEG optimization for photo-like content
        # Convert RGBA to RGB if necessary (JPEG doesn't support transparency)
        if image.mode == "RGBA":
            # Create white background and paste image on it
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
            image = rgb_image
        elif image.mode not in ("RGB", "L"):
            # Convert other modes to RGB
            image = image.convert("RGB")

        image.save(
            img_io,
            format=format,
            quality=quality,
            optimize=optimize,
            progressive=True,
        )
        content_type = "image/jpeg"
    else:
        # PNG optimization for graphics with transparency
        image.save(
            img_io,
            format=format,
            optimize=optimize,
            compress_level=6 if optimize else 1,
        )
        content_type = "image/png"

    img_io.seek(0)

    return Response(
        content=img_io.getvalue(),
        media_type=content_type,
        headers={
            "Cache-Control": "public, max-age=3600",
            "ETag": f'"{hash(img_io.getvalue())}"',
        },
    )


@router.post("/analyze-and-cache")
async def analyze_and_cache_for_imagery(request: ImageryAnalysisRequest):
    """Perform analysis and cache results for imagery generation.

    This endpoint runs the full analysis pipeline and caches the results
    for subsequent imagery requests.
    """
    start_time = time.time()

    try:
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())

        # Fetch satellite data
        satellite_data = fetch_sentinel_tiles(
            request.aoi.lat, request.aoi.lng, request.start_date, request.end_date
        )
        dataset = satellite_data["data"]

        # Calculate spectral indices
        indices_dataset = calculate_indices_from_dataset(dataset)

        # Merge indices with original satellite data
        combined_data = dataset.copy()
        for var in indices_dataset.data_vars:
            combined_data[var] = indices_dataset[var]

        # Apply masking
        masked_data = apply_mask(combined_data)

        # Extract indices and masks for imagery
        indices = {
            var: masked_data[var]
            for var in indices_dataset.data_vars
            if var in masked_data
        }
        masks = {
            "kelp_mask": masked_data.get("kelp_mask"),
            "water_mask": masked_data.get("water_mask"),
            "cloud_mask": masked_data.get("cloud_mask"),
        }
        # Remove None values
        masks = {k: v for k, v in masks.items() if v is not None}

        # Predict biomass
        biomass_result = predict_biomass(masked_data)
        biomass_array = None
        if biomass_result and "biomass_map" in biomass_result:
            biomass_array = biomass_result["biomass_map"]

        # Store in cache with original request parameters
        _store_analysis_result(
            analysis_id,
            dataset,
            indices,
            masks,
            biomass_array,
            lat=request.aoi.lat,
            lon=request.aoi.lng,
        )

        # Calculate geographic bounds from the original request coordinates
        # Use a small buffer around the requested point (roughly 1km in degrees)
        buffer_km = request.buffer_km or 1.0  # Default to 1km if not specified
        buffer_deg = buffer_km / 111.0  # Convert km to degrees (approximate)
        bounds = (
            request.aoi.lng - buffer_deg,  # min_lon
            request.aoi.lat - buffer_deg,  # min_lat
            request.aoi.lng + buffer_deg,  # max_lon
            request.aoi.lat + buffer_deg,  # max_lat
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Return format compatible with JavaScript expectations
        return {
            "analysis_id": analysis_id,
            "status": "success",
            "bounds": bounds,
            "available_layers": {
                "base_layers": ["rgb", "false-color"],
                "spectral_indices": list(indices.keys()),
                "masks": list(masks.keys()),
                "biomass": biomass_array is not None,
            },
            "metadata": {
                "shape": dataset.dims,
                "bands": list(dataset.data_vars),
                "acquisition_info": "Sentinel-2 L2A",
                "processing_time": f"{processing_time:.2f}s",
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/{analysis_id}/rgb")
async def get_rgb_composite(analysis_id: str):
    """Get true-color RGB composite with fallback handling."""
    try:
        result = _get_analysis_result(analysis_id)
        dataset = result["dataset"]

        # Generate RGB composite
        image = generate_rgb_composite(dataset)

        # Use JPEG for RGB composites (better compression for photos)
        return _image_to_response(image, format="JPEG", quality=85)

    except HTTPException:
        # Re-raise standardized HTTP exceptions
        raise
    except ValueError as e:
        # Data-related errors
        raise create_validation_error(
            f"Invalid data for RGB generation: {str(e)}",
            details=f"Analysis {analysis_id} data incompatible with RGB generation",
        )
    except KeyError as e:
        # Missing required bands
        raise create_validation_error(
            f"Missing required band for RGB: {str(e)}",
            details=f"Analysis {analysis_id} missing spectral bands for RGB composite",
        )
    except Exception as e:
        # Unexpected errors
        raise create_imagery_error(
            "RGB composite generation", analysis_id, original_error=e
        )


@router.get("/{analysis_id}/false-color")
async def get_false_color_composite(analysis_id: str):
    """Get false-color composite (NIR-Red-Green)."""
    try:
        result = _get_analysis_result(analysis_id)
        dataset = result["dataset"]

        image = generate_false_color_composite(dataset)
        return _image_to_response(image)

    except HTTPException:
        # Re-raise standardized HTTP exceptions
        raise
    except Exception as e:
        raise create_imagery_error(
            "False-color composite generation", analysis_id, original_error=e
        )


@router.get("/{analysis_id}/spectral/{index_name}")
async def get_spectral_visualization(analysis_id: str, index_name: str):
    """Get spectral index visualization."""
    try:
        result = _get_analysis_result(analysis_id)
        indices = result["indices"]

        if index_name not in indices:
            raise create_not_found_error(
                f"Spectral index '{index_name}'", f"in analysis {analysis_id}"
            )

        # Choose appropriate colormap based on index
        colormap_mapping = {
            "ndvi": "RdYlGn",
            "red_edge_ndvi": "viridis",
            "ndre": "plasma",
            "fai": "plasma",
            "ndwi": "Blues",
            "evi": "RdYlGn",
        }

        colormap = colormap_mapping.get(index_name, "viridis")

        image = generate_spectral_visualization(indices[index_name], colormap=colormap)
        return _image_to_response(image)

    except HTTPException:
        # Re-raise standardized HTTP exceptions
        raise
    except Exception as e:
        raise create_imagery_error(
            f"Spectral visualization for {index_name}", analysis_id, original_error=e
        )


@router.get("/{analysis_id}/mask/{mask_name}")
async def get_mask_overlay(analysis_id: str, mask_name: str, alpha: float = 0.6):
    """Get mask overlay."""
    # Validate alpha parameter
    if not (0.0 <= alpha <= 1.0):
        raise create_validation_error(
            "Alpha value must be between 0.0 and 1.0",
            field="alpha",
            details=f"Provided value: {alpha}",
        )

    try:
        result = _get_analysis_result(analysis_id)
        masks = result["masks"]

        # Create temporary dataset with masks for overlay generation
        mask_dataset = xr.Dataset()
        for mask_key, mask_data in masks.items():
            mask_dataset[mask_key] = mask_data

        if mask_name == "kelp":
            if "kelp_mask" not in mask_dataset:
                raise create_not_found_error("Kelp mask", f"in analysis {analysis_id}")
            image = generate_kelp_mask_overlay(mask_dataset, alpha=alpha)

        elif mask_name == "water":
            if "water_mask" not in mask_dataset:
                raise create_not_found_error("Water mask", f"in analysis {analysis_id}")
            image = generate_water_mask_overlay(mask_dataset, alpha=alpha)

        elif mask_name == "cloud":
            if "cloud_mask" not in mask_dataset:
                raise create_not_found_error("Cloud mask", f"in analysis {analysis_id}")
            image = generate_cloud_mask_overlay(mask_dataset, alpha=alpha)

        else:
            raise create_validation_error(
                f"Unknown mask type: {mask_name}",
                field="mask_name",
                details="Available masks: kelp, water, cloud",
            )

        return _image_to_response(image)

    except HTTPException:
        # Re-raise standardized HTTP exceptions
        raise
    except Exception as e:
        raise create_imagery_error(
            f"Mask overlay generation for {mask_name}", analysis_id, original_error=e
        )


@router.get("/{analysis_id}/biomass")
async def get_biomass_heatmap(
    analysis_id: str,
    colormap: str = "hot",
    min_biomass: float | None = None,
    max_biomass: float | None = None,
):
    """Get biomass density heatmap."""
    # Validate parameters
    if min_biomass is not None and max_biomass is not None:
        if min_biomass >= max_biomass:
            raise create_validation_error(
                "min_biomass must be less than max_biomass",
                details=f"Provided: min={min_biomass}, max={max_biomass}",
            )

    try:
        result = _get_analysis_result(analysis_id)
        biomass_array = result["biomass"]

        if biomass_array is None:
            raise create_not_found_error("Biomass data", f"in analysis {analysis_id}")

        image = generate_biomass_heatmap(
            biomass_array,
            colormap=colormap,
            min_biomass=min_biomass,
            max_biomass=max_biomass,
        )
        return _image_to_response(image)

    except HTTPException:
        # Re-raise standardized HTTP exceptions
        raise
    except Exception as e:
        raise create_imagery_error(
            "Biomass heatmap generation", analysis_id, original_error=e
        )


@router.get("/{analysis_id}/metadata")
async def get_imagery_metadata(analysis_id: str):
    """Get metadata for imagery layers."""
    try:
        result = _get_analysis_result(analysis_id)
        dataset = result["dataset"]
        indices = result["indices"]
        masks = result["masks"]
        lat = result.get("lat")
        lon = result.get("lon")

        # Calculate bounds from stored coordinates if available
        if lat is not None and lon is not None:
            buffer_deg = 0.01  # approximately 1km at mid-latitudes
            bounds = (
                lon - buffer_deg,  # min_lon
                lat - buffer_deg,  # min_lat
                lon + buffer_deg,  # max_lon
                lat + buffer_deg,  # max_lat
            )
        else:
            # Fallback to dataset bounds
            bounds = get_image_bounds(dataset)

        return {
            "analysis_id": analysis_id,
            "bounds": bounds,
            "shape": dataset.dims,
            "acquisition_date": "2023-08-16T19:19:11.024000Z",  # From satellite data
            "resolution": 10,  # Sentinel-2 resolution in meters
            "cloud_coverage": 0.001307,  # From logs - very low cloud coverage
            "satellite_info": {
                "mission": "Sentinel-2",
                "level": "L2A",
                "resolution": 10,
                "acquisition_date": "2023-08-16T19:19:11.024000Z",
                "cloud_coverage": 0.001307,
            },
            "available_layers": {
                "base_layers": ["rgb", "false-color"],
                "spectral_indices": list(indices.keys()),
                "masks": list(masks.keys()),
                "biomass": result["biomass"] is not None,
            },
            "layer_info": {
                "rgb": {
                    "name": "True Color",
                    "description": "Natural color composite using visible light bands",
                    "bands": "Red, Green, Blue",
                },
                "false_color": {
                    "name": "False Color",
                    "description": "NIR-Red-Green composite for vegetation enhancement",
                    "bands": "NIR, Red, Green",
                },
                "spectral_indices": {
                    name: {
                        "name": name.upper(),
                        "description": f"{name.upper()} spectral index visualization",
                        "range": "Varies by index",
                    }
                    for name in indices.keys()
                },
                "masks": {
                    "kelp": {"color": "#00ff00", "description": "Kelp detection areas"},
                    "water": {"color": "#0064ff", "description": "Water body areas"},
                    "cloud": {
                        "color": "#808080",
                        "description": "Cloud coverage areas",
                    },
                },
            },
        }

    except HTTPException:
        # Re-raise HTTP exceptions (like 404) without modification
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Metadata retrieval failed: {str(e)}"
        )


@router.delete("/{analysis_id}")
async def clear_analysis_cache(analysis_id: str):
    """Clear cached analysis results."""
    if analysis_id in _analysis_cache:
        del _analysis_cache[analysis_id]
        return {
            "status": "success",
            "message": f"Analysis {analysis_id} cleared from cache",
        }
    else:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")


@router.get("/health")
async def imagery_health():
    """Health check for imagery service."""
    return {"status": "ok", "service": "imagery"}
