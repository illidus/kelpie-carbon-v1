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

    # TODO: Integrate with actual pipeline (fetch -> mask -> indices -> model)
    # For now, return mock results

    processing_time = f"{time.time() - start_time:.2f}s"

    return AnalysisResponse(
        analysis_id=analysis_id,
        status="completed",
        processing_time=processing_time,
        aoi=request.aoi,
        date_range={"start": request.start_date, "end": request.end_date},
        biomass="Pending implementation",
        carbon="Pending implementation",
    )
