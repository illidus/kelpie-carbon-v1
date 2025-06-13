"""Pydantic models for API request/response validation."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.types import StrictFloat, StrictInt, StrictStr


class AnalysisStatus(str, Enum):
    """Status of an analysis request."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class CoordinateModel(BaseModel):
    """Geographic coordinate model with validation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    lat: StrictFloat = Field(
        ..., ge=-90.0, le=90.0, description="Latitude in decimal degrees"
    )
    lng: StrictFloat = Field(
        ..., ge=-180.0, le=180.0, description="Longitude in decimal degrees"
    )

    @field_validator("lat")
    @classmethod
    def validate_latitude(cls, v):
        """Validate latitude is within valid range."""
        if not -90.0 <= v <= 90.0:
            raise ValueError("Latitude must be between -90 and 90 degrees")
        return v

    @field_validator("lng")
    @classmethod
    def validate_longitude(cls, v):
        """Validate longitude is within valid range."""
        if not -180.0 <= v <= 180.0:
            raise ValueError("Longitude must be between -180 and 180 degrees")
        return v


class DateRangeModel(BaseModel):
    """Date range model with validation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    start_date: StrictStr = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Start date in YYYY-MM-DD format",
    )
    end_date: StrictStr = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="End date in YYYY-MM-DD format"
    )

    @field_validator("end_date")
    @classmethod
    def validate_date_order(cls, v, info):
        """Ensure end_date is after start_date."""
        if info.data and "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v


class AnalysisRequest(BaseModel):
    """Request model for analysis endpoint."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "aoi": {"lat": 36.8, "lng": -121.9},
                "start_date": "2023-08-01",
                "end_date": "2023-08-31",
            }
        },
    )

    aoi: CoordinateModel = Field(..., description="Area of interest coordinates")
    start_date: StrictStr = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Analysis start date in YYYY-MM-DD format",
    )
    end_date: StrictStr = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Analysis end date in YYYY-MM-DD format",
    )

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, v, info):
        """Validate date range is logical."""
        if info.data and "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v


class ModelInfoModel(BaseModel):
    """Model information for analysis results."""

    model_config = ConfigDict(str_strip_whitespace=True)

    type: StrictStr = Field(..., description="Model type used for analysis")
    confidence: StrictStr = Field(..., description="Model confidence score")
    feature_count: StrictStr = Field(..., description="Number of features used")
    algorithm: StrictStr | None = Field(None, description="Algorithm used")
    version: StrictStr | None = Field(None, description="Model version")


class MaskStatisticsModel(BaseModel):
    """Mask statistics for analysis results."""

    model_config = ConfigDict(str_strip_whitespace=True)

    water_coverage: StrictFloat | None = Field(
        None, ge=0.0, le=1.0, description="Water coverage ratio"
    )
    cloud_coverage: StrictFloat | None = Field(
        None, ge=0.0, le=1.0, description="Cloud coverage ratio"
    )
    kelp_coverage: StrictFloat | None = Field(
        None, ge=0.0, le=1.0, description="Kelp coverage ratio"
    )
    land_coverage: StrictFloat | None = Field(
        None, ge=0.0, le=1.0, description="Land coverage ratio"
    )
    valid_pixels: StrictInt | None = Field(
        None, ge=0, description="Number of valid pixels"
    )
    total_pixels: StrictInt | None = Field(
        None, ge=0, description="Total number of pixels"
    )


class SpectralIndicesModel(BaseModel):
    """Spectral indices for analysis results."""

    model_config = ConfigDict(str_strip_whitespace=True)

    avg_ndvi: StrictFloat | None = Field(None, description="Average NDVI value")
    avg_ndre: StrictFloat | None = Field(
        None, description="Average NDRE value (enhanced kelp detection)"
    )
    avg_fai: StrictFloat | None = Field(None, description="Average FAI value")
    avg_red_edge_ndvi: StrictFloat | None = Field(
        None, description="Average Red Edge NDVI value"
    )
    avg_kelp_index: StrictFloat | None = Field(
        None, description="Average Kelp Index value"
    )
    std_ndvi: StrictFloat | None = Field(None, description="Standard deviation of NDVI")
    std_ndre: StrictFloat | None = Field(None, description="Standard deviation of NDRE")
    std_fai: StrictFloat | None = Field(None, description="Standard deviation of FAI")


class AnalysisResponse(BaseModel):
    """Response model for analysis endpoint."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "analysis_id": "abc12345",
                "status": "completed",
                "processing_time": "15.2s",
                "aoi": {"lat": 36.8, "lng": -121.9},
                "date_range": {"start": "2023-08-01", "end": "2023-08-31"},
                "biomass": "250.5 kg/ha (confidence: 0.87)",
                "carbon": "0.0088 kg C/m² (88.2 kg C/ha)",
                "model_info": {
                    "type": "RandomForest",
                    "confidence": "0.87",
                    "feature_count": "15",
                },
            }
        },
    )

    analysis_id: StrictStr = Field(..., description="Unique analysis identifier")
    status: AnalysisStatus = Field(..., description="Analysis status")
    processing_time: StrictStr = Field(
        ..., description="Time taken to process analysis"
    )
    aoi: CoordinateModel = Field(..., description="Area of interest coordinates")
    date_range: dict[str, str] = Field(..., description="Date range for analysis")
    biomass: StrictStr = Field(..., description="Biomass estimation result")
    carbon: StrictStr = Field(..., description="Carbon sequestration result")
    model_info: ModelInfoModel = Field(
        default_factory=lambda: ModelInfoModel(
            type="", confidence="", feature_count="", algorithm="", version=""
        )
    )
    mask_statistics: MaskStatisticsModel = Field(
        default_factory=lambda: MaskStatisticsModel(
            water_coverage=0.0,
            cloud_coverage=0.0,
            kelp_coverage=0.0,
            land_coverage=0.0,
            valid_pixels=0,
            total_pixels=0,
        )
    )
    spectral_indices: SpectralIndicesModel = Field(
        default_factory=lambda: SpectralIndicesModel(
            avg_ndvi=0.0,
            avg_ndre=0.0,
            avg_fai=0.0,
            avg_red_edge_ndvi=0.0,
            avg_kelp_index=0.0,
            std_ndvi=0.0,
            std_ndre=0.0,
            std_fai=0.0,
        )
    )
    top_features: list[str] = Field(
        default_factory=list, description="Top contributing features"
    )
    error_message: StrictStr | None = Field(
        None, description="Error message if status is error"
    )


class HealthResponse(BaseModel):
    """Response model for health endpoint."""

    model_config = ConfigDict(str_strip_whitespace=True)

    status: StrictStr = Field(..., description="Health status")
    version: StrictStr = Field(..., description="Application version")
    environment: StrictStr = Field(..., description="Environment name")
    timestamp: StrictFloat = Field(..., description="Response timestamp")


class ReadinessCheck(BaseModel):
    """Individual readiness check result."""

    model_config = ConfigDict(str_strip_whitespace=True)

    config: bool = Field(..., description="Configuration loading status")
    static_files: bool = Field(..., description="Static files availability")
    database: bool | None = Field(
        None, description="Database connectivity (if applicable)"
    )
    external_services: bool | None = Field(None, description="External services status")


class ReadinessResponse(BaseModel):
    """Response model for readiness endpoint."""

    model_config = ConfigDict(str_strip_whitespace=True)

    status: StrictStr = Field(..., description="Readiness status")
    checks: ReadinessCheck = Field(..., description="Individual check results")
    timestamp: StrictFloat = Field(..., description="Response timestamp")


class ImageryAnalysisRequest(BaseModel):
    """Request model for imagery analysis endpoint."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "aoi": {"lat": 36.8, "lng": -121.9},
                "start_date": "2023-08-01",
                "end_date": "2023-08-31",
                "buffer_km": 1.0,
                "max_cloud_cover": 0.2,
            }
        },
    )

    aoi: CoordinateModel = Field(..., description="Area of interest coordinates")
    start_date: StrictStr = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Start date in YYYY-MM-DD format",
    )
    end_date: StrictStr = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="End date in YYYY-MM-DD format"
    )
    buffer_km: StrictFloat | None = Field(
        1.0, ge=0.1, le=10.0, description="Buffer around point in kilometers"
    )
    max_cloud_cover: StrictFloat | None = Field(
        0.3, ge=0.0, le=1.0, description="Maximum cloud cover threshold"
    )


class LayerInfo(BaseModel):
    """Information about an available layer."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: StrictStr = Field(..., description="Layer name")
    type: StrictStr = Field(..., description="Layer type (rgb, spectral, mask, etc.)")
    description: StrictStr = Field(..., description="Layer description")
    available: bool = Field(..., description="Whether layer is available")
    url_template: StrictStr | None = Field(
        None, description="URL template for accessing layer"
    )


class ImageryAnalysisResponse(BaseModel):
    """Response model for imagery analysis endpoint."""

    model_config = ConfigDict(str_strip_whitespace=True)

    analysis_id: StrictStr = Field(..., description="Unique analysis identifier")
    status: AnalysisStatus = Field(..., description="Analysis status")
    processing_time: StrictStr = Field(..., description="Processing time")
    bounds: list[StrictFloat] = Field(
        ..., description="Geographic bounds [minX, minY, maxX, maxY]"
    )
    available_layers: list[str] = Field(
        ..., description="List of available layer names"
    )
    layer_info: dict[str, LayerInfo] = Field(
        ..., description="Detailed layer information"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""

    model_config = ConfigDict(str_strip_whitespace=True)

    error: StrictStr = Field(..., description="Error type")
    message: StrictStr = Field(..., description="Error message")
    details: dict[str, Any] | None = Field(None, description="Additional error details")
    timestamp: StrictFloat = Field(..., description="Error timestamp")
    request_id: StrictStr | None = Field(
        None, description="Request identifier for tracking"
    )


# Response model unions for OpenAPI documentation
AnalysisResponseUnion = AnalysisResponse | ErrorResponse
ImageryAnalysisResponseUnion = ImageryAnalysisResponse | ErrorResponse
