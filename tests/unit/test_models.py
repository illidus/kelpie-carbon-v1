"""Tests for API models and validation."""

import pytest
from pydantic import ValidationError

from kelpie_carbon_v1.api.models import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatus,
    CoordinateModel,
    ErrorResponse,
    HealthResponse,
    MaskStatisticsModel,
    ModelInfoModel,
    ReadinessCheck,
    ReadinessResponse,
    SpectralIndicesModel,
)


@pytest.mark.unit
@pytest.mark.api
class TestCoordinateModel:
    """Test coordinate validation."""

    def test_valid_coordinates(self):
        """Test valid coordinate creation."""
        coord = CoordinateModel(lat=36.8, lng=-121.9)
        assert coord.lat == 36.8
        assert coord.lng == -121.9

    def test_latitude_validation(self):
        """Test latitude boundary validation."""
        # Valid boundaries
        CoordinateModel(lat=-90.0, lng=0.0)
        CoordinateModel(lat=90.0, lng=0.0)

        # Invalid latitudes
        with pytest.raises(ValidationError):
            CoordinateModel(lat=-91.0, lng=0.0)

        with pytest.raises(ValidationError):
            CoordinateModel(lat=91.0, lng=0.0)

    def test_longitude_validation(self):
        """Test longitude boundary validation."""
        # Valid boundaries
        CoordinateModel(lat=0.0, lng=-180.0)
        CoordinateModel(lat=0.0, lng=180.0)

        # Invalid longitudes
        with pytest.raises(ValidationError):
            CoordinateModel(lat=0.0, lng=-181.0)

        with pytest.raises(ValidationError):
            CoordinateModel(lat=0.0, lng=181.0)

    def test_string_coordinates(self):
        """Test that string coordinates are rejected."""
        with pytest.raises(ValidationError):
            CoordinateModel(lat="36.8", lng=-121.9)


@pytest.mark.unit
@pytest.mark.api
class TestAnalysisRequest:
    """Test analysis request validation."""

    def test_valid_request(self, sample_coordinates):
        """Test valid analysis request creation."""
        request = AnalysisRequest(
            aoi=CoordinateModel(
                lat=sample_coordinates["lat"], lng=sample_coordinates["lng"]
            ),
            start_date=sample_coordinates["start_date"],
            end_date=sample_coordinates["end_date"],
        )

        assert request.aoi.lat == sample_coordinates["lat"]
        assert request.start_date == sample_coordinates["start_date"]
        assert request.end_date == sample_coordinates["end_date"]

    def test_date_format_validation(self):
        """Test date format validation."""
        coord = CoordinateModel(lat=36.8, lng=-121.9)

        # Valid dates
        AnalysisRequest(aoi=coord, start_date="2023-08-01", end_date="2023-08-31")

        # Invalid date formats
        with pytest.raises(ValidationError):
            AnalysisRequest(aoi=coord, start_date="2023/08/01", end_date="2023-08-31")

        with pytest.raises(ValidationError):
            AnalysisRequest(aoi=coord, start_date="08-01-2023", end_date="2023-08-31")

        with pytest.raises(ValidationError):
            AnalysisRequest(aoi=coord, start_date="2023-8-1", end_date="2023-08-31")

    def test_date_order_validation(self):
        """Test that end_date must be after start_date."""
        coord = CoordinateModel(lat=36.8, lng=-121.9)

        # Valid order
        AnalysisRequest(aoi=coord, start_date="2023-08-01", end_date="2023-08-31")

        # Invalid order
        with pytest.raises(ValidationError):
            AnalysisRequest(aoi=coord, start_date="2023-08-31", end_date="2023-08-01")

        # Same date
        with pytest.raises(ValidationError):
            AnalysisRequest(aoi=coord, start_date="2023-08-01", end_date="2023-08-01")


@pytest.mark.unit
@pytest.mark.api
class TestAnalysisResponse:
    """Test analysis response model."""

    def test_valid_response_creation(self):
        """Test creating a valid analysis response."""
        coord = CoordinateModel(lat=36.8, lng=-121.9)
        model_info = ModelInfoModel(
            type="RandomForest", confidence="0.87", feature_count="15"
        )

        response = AnalysisResponse(
            analysis_id="test123",
            status=AnalysisStatus.COMPLETED,
            processing_time="15.2s",
            aoi=coord,
            date_range={"start": "2023-08-01", "end": "2023-08-31"},
            biomass="250.5 kg/ha",
            carbon="0.0088 kg C/m²",
            model_info=model_info,
        )

        assert response.analysis_id == "test123"
        assert response.status == AnalysisStatus.COMPLETED
        assert response.model_info.type == "RandomForest"

    def test_status_enum(self):
        """Test analysis status enum values."""
        assert AnalysisStatus.PENDING == "pending"
        assert AnalysisStatus.PROCESSING == "processing"
        assert AnalysisStatus.COMPLETED == "completed"
        assert AnalysisStatus.ERROR == "error"
        assert AnalysisStatus.CANCELLED == "cancelled"


@pytest.mark.unit
@pytest.mark.api
class TestHealthResponse:
    """Test health response model."""

    def test_health_response_creation(self):
        """Test creating a health response."""
        response = HealthResponse(
            status="ok",
            version="0.1.0",
            environment="development",
            timestamp=1234567890.0,
        )

        assert response.status == "ok"
        assert response.version == "0.1.0"
        assert response.environment == "development"
        assert response.timestamp == 1234567890.0


@pytest.mark.unit
@pytest.mark.api
class TestReadinessResponse:
    """Test readiness response model."""

    def test_readiness_response_creation(self):
        """Test creating a readiness response."""
        checks = ReadinessCheck(config=True, static_files=True)
        response = ReadinessResponse(
            status="ready", checks=checks, timestamp=1234567890.0
        )

        assert response.status == "ready"
        assert response.checks.config is True
        assert response.checks.static_files is True


@pytest.mark.unit
@pytest.mark.api
class TestMaskStatisticsModel:
    """Test mask statistics model."""

    def test_coverage_validation(self):
        """Test coverage ratio validation."""
        # Valid coverage values
        MaskStatisticsModel(water_coverage=0.5, cloud_coverage=0.2)
        MaskStatisticsModel(water_coverage=0.0, cloud_coverage=1.0)

        # Invalid coverage values
        with pytest.raises(ValidationError):
            MaskStatisticsModel(water_coverage=1.5)

        with pytest.raises(ValidationError):
            MaskStatisticsModel(cloud_coverage=-0.1)

    def test_pixel_count_validation(self):
        """Test pixel count validation."""
        # Valid pixel counts
        MaskStatisticsModel(valid_pixels=1000, total_pixels=2000)
        MaskStatisticsModel(valid_pixels=0, total_pixels=0)

        # Invalid pixel counts
        with pytest.raises(ValidationError):
            MaskStatisticsModel(valid_pixels=-1)

        with pytest.raises(ValidationError):
            MaskStatisticsModel(total_pixels=-1)


@pytest.mark.unit
@pytest.mark.api
class TestErrorResponse:
    """Test error response model."""

    def test_error_response_creation(self):
        """Test creating an error response."""
        response = ErrorResponse(
            error="ValidationError",
            message="Invalid coordinates provided",
            timestamp=1234567890.0,
            details={"field": "lat", "value": "invalid"},
        )

        assert response.error == "ValidationError"
        assert response.message == "Invalid coordinates provided"
        assert response.details["field"] == "lat"


@pytest.mark.unit
@pytest.mark.api
class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_analysis_request_json(self):
        """Test analysis request JSON serialization."""
        request_data = {
            "aoi": {"lat": 36.8, "lng": -121.9},
            "start_date": "2023-08-01",
            "end_date": "2023-08-31",
        }

        request = AnalysisRequest(**request_data)
        json_data = request.model_dump()

        assert json_data["aoi"]["lat"] == 36.8
        assert json_data["start_date"] == "2023-08-01"

    def test_analysis_response_json(self):
        """Test analysis response JSON serialization."""
        coord = CoordinateModel(lat=36.8, lng=-121.9)
        model_info = ModelInfoModel(
            type="RandomForest", confidence="0.87", feature_count="15"
        )

        response = AnalysisResponse(
            analysis_id="test123",
            status=AnalysisStatus.COMPLETED,
            processing_time="15.2s",
            aoi=coord,
            date_range={"start": "2023-08-01", "end": "2023-08-31"},
            biomass="250.5 kg/ha",
            carbon="0.0088 kg C/m²",
            model_info=model_info,
        )

        json_data = response.model_dump()
        assert json_data["status"] == "completed"
        assert json_data["model_info"]["type"] == "RandomForest"
