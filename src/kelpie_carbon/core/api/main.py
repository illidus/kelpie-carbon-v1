"""FastAPI application for Kelpie-Carbon v1."""

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_settings
from ..constants import KelpAnalysis, Network
from ..logging_config import get_logger, setup_logging
from .errors import (
    create_coordinate_error,
    create_date_range_error,
    create_processing_error,
    create_service_unavailable_error,
    handle_unexpected_error,
)
from .imagery import router as imagery_router
from .models import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatus,
    HealthResponse,
    ReadinessCheck,
    ReadinessResponse,
)

# Setup logging and configuration
setup_logging()
settings = get_settings()
logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    description=settings.description,
    version=settings.app_version,
    debug=settings.debug,
)


# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add comprehensive security headers to all responses."""
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    # Content Security Policy - allows necessary external resources for Leaflet
    csp_policy = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com; "
        "style-src 'self' 'unsafe-inline' https://unpkg.com; "
        "img-src 'self' data: https: blob:; "
        "connect-src 'self' https:; "
        "font-src 'self' data: https:; "
        "frame-ancestors 'none'"
    )
    response.headers["Content-Security-Policy"] = csp_policy

    # HSTS for HTTPS deployments
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = (
            f"max-age={Network.HSTS_MAX_AGE}; includeSubDomains"
        )

    return response


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.allow_origins,
    allow_credentials=settings.cors.allow_credentials,
    allow_methods=settings.cors.allow_methods,
    allow_headers=settings.cors.allow_headers,
)

# Include imagery router
app.include_router(imagery_router)

# Mount static files for web interface
static_path = Path(settings.paths.static_files)
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    logger.info(f"Mounted static files from {static_path}")
else:
    logger.warning(f"Static files directory not found: {static_path}")


# Model imports moved to top of file


@app.get("/")
def root():
    """Serve the web interface."""
    html_path = static_path / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {
        "message": f"{settings.app_name} API",
        "version": settings.app_version,
        "environment": settings.environment,
        "web_ui": "Not available",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.environment,
        timestamp=time.time(),
    )


@app.get("/ready", response_model=ReadinessResponse)
def readiness():
    """Readiness check endpoint."""
    # Check if critical components are available
    try:
        # Test configuration loading
        _ = get_settings()

        # Test static files
        static_available = static_path.exists()

        return ReadinessResponse(
            status="ready",
            checks=ReadinessCheck(
                config=True,
                static_files=static_available,
                database=None,
                external_services=None,
            ),
            timestamp=time.time(),
        )
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise create_service_unavailable_error(
            "Application readiness check", details=str(e)[:100]
        )


@app.post("/api/run", response_model=AnalysisResponse)
def run_analysis(request: AnalysisRequest):
    """Run kelp forest carbon analysis for given AOI and date range."""
    start_time = time.time()

    # Generate unique analysis ID
    analysis_id = str(uuid.uuid4())[:8]

    logger.info(
        f"Starting analysis {analysis_id} for location ({request.aoi.lat}, {request.aoi.lng})"
    )

    # Validate coordinates
    if not (-90 <= request.aoi.lat <= 90):
        raise create_coordinate_error(
            "Invalid latitude value", lat=request.aoi.lat, lng=request.aoi.lng
        )

    if not (-180 <= request.aoi.lng <= 180):
        raise create_coordinate_error(
            "Invalid longitude value", lat=request.aoi.lat, lng=request.aoi.lng
        )

    # Validate date range
    from datetime import datetime

    try:
        start_dt = datetime.fromisoformat(request.start_date)
        end_dt = datetime.fromisoformat(request.end_date)

        if end_dt <= start_dt:
            raise create_date_range_error(
                "End date must be after start date",
                start_date=request.start_date,
                end_date=request.end_date,
            )

        # Check if date range is reasonable (not more than 1 year)
        if (end_dt - start_dt).days > 365:
            raise create_date_range_error(
                "Date range cannot exceed 1 year",
                start_date=request.start_date,
                end_date=request.end_date,
            )

    except ValueError:
        raise create_date_range_error(
            "Invalid date format",
            start_date=request.start_date,
            end_date=request.end_date,
        )

    try:
        # Import pipeline modules
        from .. import (
            calculate_indices_from_dataset,
            fetch_sentinel_tiles,
            predict_biomass,
        )

        # Step 1: Fetch satellite data
        try:
            satellite_data = fetch_sentinel_tiles(
                lat=request.aoi.lat,
                lng=request.aoi.lng,
                start_date=request.start_date,
                end_date=request.end_date,
            )
        except Exception as e:
            raise create_processing_error(
                "Satellite data retrieval", e, analysis_id=analysis_id
            )

        # Step 2: Calculate spectral indices and merge with satellite data
        try:
            indices = calculate_indices_from_dataset(satellite_data["data"])

            # Merge indices with original satellite data
            combined_data = satellite_data["data"].copy()
            for var in indices.data_vars:
                combined_data[var] = indices[var]
        except Exception as e:
            raise create_processing_error(
                "Spectral index calculation", e, analysis_id=analysis_id
            )

        # Step 3: Apply advanced masking
        try:
            from .. import apply_mask, get_mask_statistics

            masked_data = apply_mask(combined_data)
            mask_stats = get_mask_statistics(masked_data)
        except Exception as e:
            raise create_processing_error("Data masking", e, analysis_id=analysis_id)

        # Step 4: Predict biomass using Random Forest model
        try:
            biomass_result = predict_biomass(masked_data)
        except Exception as e:
            raise create_processing_error(
                "Biomass prediction", e, analysis_id=analysis_id
            )

        # Extract biomass prediction
        biomass_kg_ha = biomass_result["biomass_kg_per_hectare"]
        confidence = biomass_result["prediction_confidence"]
        model_type = biomass_result["model_type"]

        biomass_str = f"{biomass_kg_ha:.1f} kg/ha (confidence: {confidence:.2f})"

        # Calculate carbon sequestration
        # Convert kg/ha to kg/m² and apply carbon content factor (35% typical for kelp)
        biomass_kg_m2 = (
            biomass_kg_ha / KelpAnalysis.HECTARE_TO_M2
        )  # Convert hectare to m²
        carbon_kg_m2 = (
            biomass_kg_m2 * KelpAnalysis.CARBON_CONTENT_FACTOR
        )  # Carbon content factor for kelp
        carbon_str = f"{carbon_kg_m2:.4f} kg C/m² ({carbon_kg_m2 * 10000:.1f} kg C/ha)"

        processing_time = f"{time.time() - start_time:.2f}s"

        logger.info(
            f"Analysis {analysis_id} completed successfully in {processing_time}"
        )

        # Calculate average spectral indices for summary
        spectral_summary = {}
        if "ndvi" in masked_data:
            spectral_summary["avg_ndvi"] = float(masked_data["ndvi"].mean().values)
        if "ndre" in masked_data:
            spectral_summary["avg_ndre"] = float(masked_data["ndre"].mean().values)
        if "fai" in masked_data:
            spectral_summary["avg_fai"] = float(masked_data["fai"].mean().values)
        if "red_edge_ndvi" in masked_data:
            spectral_summary["avg_red_edge_ndvi"] = float(
                masked_data["red_edge_ndvi"].mean().values
            )
        if "kelp_index" in masked_data:
            spectral_summary["avg_kelp_index"] = float(
                masked_data["kelp_index"].mean().values
            )

        from .models import MaskStatisticsModel, ModelInfoModel, SpectralIndicesModel

        return AnalysisResponse(
            analysis_id=analysis_id,
            status=AnalysisStatus.COMPLETED,
            processing_time=processing_time,
            aoi=request.aoi,
            date_range={"start": request.start_date, "end": request.end_date},
            biomass=biomass_str,
            carbon=carbon_str,
            model_info=ModelInfoModel(
                type=model_type,
                confidence=f"{confidence:.2f}",
                feature_count=str(biomass_result.get("feature_count", 0)),
                algorithm=biomass_result.get("algorithm", "RandomForest"),
                version=biomass_result.get("version", "1.0"),
            ),
            mask_statistics=(
                MaskStatisticsModel(
                    water_coverage=mask_stats.get("water_coverage", 0.0),
                    cloud_coverage=mask_stats.get("cloud_coverage", 0.0),
                    kelp_coverage=mask_stats.get("kelp_coverage", 0.0),
                    land_coverage=mask_stats.get("land_coverage", 0.0),
                    valid_pixels=int(mask_stats.get("valid_pixels", 0)),
                    total_pixels=int(mask_stats.get("total_pixels", 0)),
                )
                if mask_stats
                else MaskStatisticsModel(
                    water_coverage=0.0,
                    cloud_coverage=0.0,
                    kelp_coverage=0.0,
                    land_coverage=0.0,
                    valid_pixels=0,
                    total_pixels=0,
                )
            ),
            spectral_indices=(
                SpectralIndicesModel(**spectral_summary)
                if spectral_summary
                else SpectralIndicesModel(
                    avg_ndvi=0.0,
                    avg_ndre=0.0,
                    avg_fai=0.0,
                    avg_red_edge_ndvi=0.0,
                    avg_kelp_index=0.0,
                    std_ndvi=0.0,
                    std_ndre=0.0,
                    std_fai=0.0,
                )
            ),
            top_features=biomass_result.get("top_features", []),
            error_message=None,
        )

    except Exception as e:
        # Handle any unexpected errors that weren't caught by specific handlers
        if isinstance(e, HTTPException):
            # Re-raise our standardized errors
            raise e

        # Handle truly unexpected errors
        processing_time = f"{time.time() - start_time:.2f}s"
        logger.error(f"Analysis {analysis_id} failed after {processing_time}: {str(e)}")

        raise handle_unexpected_error("analysis processing", e, analysis_id=analysis_id)
