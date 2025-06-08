"""FastAPI application for Kelpie-Carbon v1."""
import time
import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Kelpie Carbon v1", description="Kelp Forest Carbon Assessment API")

# Mount static files for web interface
static_path = Path(__file__).parent.parent / "web" / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


class AnalysisRequest(BaseModel):
    """Request model for analysis endpoint."""

    aoi: Dict[str, float]  # {"lat": float, "lng": float}
    start_date: str
    end_date: str


class AnalysisResponse(BaseModel):
    """Response model for analysis endpoint."""

    analysis_id: str
    status: str
    processing_time: str
    aoi: Dict[str, float]
    date_range: Dict[str, str]
    biomass: str
    carbon: str


@app.get("/")
def root():
    """Serve the web interface."""
    html_path = static_path / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {"message": "Kelpie Carbon v1 API", "web_ui": "Not available"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/api/run", response_model=AnalysisResponse)
def run_analysis(request: AnalysisRequest):
    """Run kelp forest carbon analysis for given AOI and date range."""
    start_time = time.time()

    # Generate unique analysis ID
    analysis_id = str(uuid.uuid4())[:8]

    try:
        # Import pipeline modules
        from ..fetch import fetch_sentinel_tiles
        from ..indices import calculate_indices_from_dataset
        from ..model import predict_biomass

        # Step 1: Fetch satellite data
        satellite_data = fetch_sentinel_tiles(
            lat=request.aoi["lat"],
            lng=request.aoi["lng"],
            start_date=request.start_date,
            end_date=request.end_date,
        )

        # Step 2: Calculate spectral indices
        indices = calculate_indices_from_dataset(satellite_data["data"])

        # Step 3: Apply masking (cloud/water mask)
        # For now, use a simple mask based on cloud_mask
        cloud_mask = satellite_data["data"]["cloud_mask"]
        clean_data = indices.where(cloud_mask == 0)  # Keep non-cloudy pixels

        # Step 4: Predict biomass (using mock for now)
        try:
            biomass_result = predict_biomass(clean_data)
            biomass_str = f"{biomass_result:.2f} kg/m²"
        except NotImplementedError:
            biomass_str = "Model pending implementation"

        # Calculate carbon sequestration (rough estimate: 1 kg biomass = 0.4 kg carbon)
        if "pending" not in biomass_str.lower():
            try:
                biomass_val = float(biomass_str.split()[0])
                carbon_val = biomass_val * 0.4  # Carbon content factor
                carbon_str = f"{carbon_val:.2f} kg C/m²"
            except ValueError:
                carbon_str = "Calculation error"
        else:
            carbon_str = "Pending biomass calculation"

        processing_time = f"{time.time() - start_time:.2f}s"

        return AnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            processing_time=processing_time,
            aoi=request.aoi,
            date_range={"start": request.start_date, "end": request.end_date},
            biomass=biomass_str,
            carbon=carbon_str,
        )

    except Exception as e:
        processing_time = f"{time.time() - start_time:.2f}s"
        return AnalysisResponse(
            analysis_id=analysis_id,
            status="error",
            processing_time=processing_time,
            aoi=request.aoi,
            date_range={"start": request.start_date, "end": request.end_date},
            biomass=f"Error: {str(e)[:100]}",
            carbon="Error in processing",
        )
