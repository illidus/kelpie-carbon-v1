"""Tests for API module."""
from fastapi.testclient import TestClient

from kelpie_carbon_v1.api.main import app


def test_health_endpoint():
    """Test health endpoint with sync client."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint():
    """Test that root endpoint returns web interface or API info."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200


def test_run_analysis_endpoint():
    """Test the /api/run endpoint with valid request."""
    test_request = {
        "aoi": {"lat": 49.2827, "lng": -123.1207},
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
    }

    client = TestClient(app)
    response = client.post("/api/run", json=test_request)

    assert response.status_code == 200
    result = response.json()

    # Check response structure
    assert "analysis_id" in result
    assert result["status"] == "completed"
    assert "processing_time" in result
    assert result["aoi"] == test_request["aoi"]
    assert result["date_range"]["start"] == test_request["start_date"]
    assert result["date_range"]["end"] == test_request["end_date"]


def test_run_analysis_endpoint_invalid_request():
    """Test the /api/run endpoint with invalid request."""
    invalid_request = {
        "aoi": {"lat": "invalid"},  # Should be float
        "start_date": "2023-01-01"
        # Missing end_date
    }

    client = TestClient(app)
    response = client.post("/api/run", json=invalid_request)

    assert response.status_code == 422  # Validation error
