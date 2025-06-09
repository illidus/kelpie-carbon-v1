"""FastAPI application for Kelpie-Carbon v1."""
import time
import uuid
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ..config import get_settings
from ..constants import KelpAnalysis, Network
from ..logging_config import setup_logging, get_logger, log_api_request
from .imagery import router as imagery_router
from .models import (
    AnalysisRequest, 
    AnalysisResponse, 
    HealthResponse, 
    ReadinessResponse,
    ReadinessCheck,
    ErrorResponse
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
        response.headers["Strict-Transport-Security"] = f"max-age={Network.HSTS_MAX_AGE}; includeSubDomains"
    
    return response

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

# Include imagery router
app.include_router(imagery_router)

# Mount static files for web interface
static_path = Path(settings.static_files_path)
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
            ),
            timestamp=time.time(),
        )
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@app.post("/api/run", response_model=AnalysisResponse)
def run_analysis(request: AnalysisRequest):
    """Run kelp forest carbon analysis for given AOI and date range."""
    start_time = time.time()

    # Generate unique analysis ID
    analysis_id = str(uuid.uuid4())[:8]
    
    logger.info(f"Starting analysis {analysis_id} for location ({request.aoi.lat}, {request.aoi.lng})")

    try:
        # Import pipeline modules
        from ..core import fetch_sentinel_tiles, calculate_indices_from_dataset, predict_biomass

        # Step 1: Fetch satellite data
        satellite_data = fetch_sentinel_tiles(
            lat=request.aoi.lat,
            lng=request.aoi.lng,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        # Step 2: Calculate spectral indices and merge with satellite data
        indices = calculate_indices_from_dataset(satellite_data["data"])

        # Merge indices with original satellite data
        combined_data = satellite_data["data"].copy()
        for var in indices.data_vars:
            combined_data[var] = indices[var]

        # Step 3: Apply advanced masking
        from ..core import apply_mask, get_mask_statistics

        masked_data = apply_mask(combined_data)
        mask_stats = get_mask_statistics(masked_data)

        # Step 4: Predict biomass using Random Forest model
        biomass_result = predict_biomass(masked_data)

        # Extract biomass prediction
        biomass_kg_ha = biomass_result["biomass_kg_per_hectare"]
        confidence = biomass_result["prediction_confidence"]
        model_type = biomass_result["model_type"]

        biomass_str = f"{biomass_kg_ha:.1f} kg/ha (confidence: {confidence:.2f})"

        # Calculate carbon sequestration
        # Convert kg/ha to kg/m² and apply carbon content factor (35% typical for kelp)
        biomass_kg_m2 = biomass_kg_ha / KelpAnalysis.HECTARE_TO_M2  # Convert hectare to m²
        carbon_kg_m2 = biomass_kg_m2 * KelpAnalysis.CARBON_CONTENT_FACTOR  # Carbon content factor for kelp
        carbon_str = f"{carbon_kg_m2:.4f} kg C/m² ({carbon_kg_m2 * 10000:.1f} kg C/ha)"

        processing_time = f"{time.time() - start_time:.2f}s"
        
        logger.info(f"Analysis {analysis_id} completed successfully in {processing_time}")

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

        from .models import ModelInfoModel, MaskStatisticsModel, SpectralIndicesModel
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            processing_time=processing_time,
            aoi=request.aoi,
            date_range={"start": request.start_date, "end": request.end_date},
            biomass=biomass_str,
            carbon=carbon_str,
            model_info=ModelInfoModel(
                type=model_type,
                confidence=f"{confidence:.2f}",
                feature_count=str(biomass_result.get("feature_count", 0)),
            ),
            mask_statistics=MaskStatisticsModel(**mask_stats) if mask_stats else MaskStatisticsModel(),
            spectral_indices=SpectralIndicesModel(**spectral_summary) if spectral_summary else SpectralIndicesModel(),
            top_features=biomass_result.get("top_features", []),
        )

    except Exception as e:
        processing_time = f"{time.time() - start_time:.2f}s"
        logger.error(f"Analysis {analysis_id} failed after {processing_time}: {str(e)}")
        
        from .models import ModelInfoModel, MaskStatisticsModel, SpectralIndicesModel
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            status="error",
            processing_time=processing_time,
            aoi=request.aoi,
            date_range={"start": request.start_date, "end": request.end_date},
            biomass=f"Error: {str(e)[:100]}",
            carbon="Error in processing",
            model_info=ModelInfoModel(type="Error", confidence="0", feature_count="0"),
            mask_statistics=MaskStatisticsModel(),
            spectral_indices=SpectralIndicesModel(),
            top_features=[],
            error_message=str(e)[:200],
        )
