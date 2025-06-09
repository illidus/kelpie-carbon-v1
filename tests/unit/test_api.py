"""Tests for API module."""
import pytest


@pytest.mark.api
@pytest.mark.unit
def test_health_endpoint(test_client):
    """Test health endpoint with sync client."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "environment" in data
    assert "timestamp" in data


@pytest.mark.api
@pytest.mark.unit
def test_readiness_endpoint(test_client):
    """Test readiness endpoint."""
    response = test_client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "checks" in data
    assert "timestamp" in data


@pytest.mark.api
@pytest.mark.unit
def test_root_endpoint(test_client):
    """Test that root endpoint returns web interface or API info."""
    response = test_client.get("/")
    assert response.status_code == 200


@pytest.mark.api
@pytest.mark.integration
@pytest.mark.slow
def test_run_analysis_endpoint(test_client, sample_coordinates):
    """Test the /api/run endpoint with valid request."""
    test_request = {
        "aoi": {"lat": sample_coordinates["lat"], "lng": sample_coordinates["lng"]},
        "start_date": sample_coordinates["start_date"],
        "end_date": sample_coordinates["end_date"],
    }

    response = test_client.post("/api/run", json=test_request)

    assert response.status_code == 200
    result = response.json()

    # Check response structure
    assert "analysis_id" in result
    assert result["status"] in [
        "completed",
        "error",
    ]  # Allow for both success and error
    assert "processing_time" in result
    assert result["aoi"] == test_request["aoi"]
    assert result["date_range"]["start"] == test_request["start_date"]
    assert result["date_range"]["end"] == test_request["end_date"]


@pytest.mark.api
@pytest.mark.unit
def test_run_analysis_endpoint_invalid_request(test_client):
    """Test the /api/run endpoint with invalid request."""
    invalid_request = {
        "aoi": {"lat": "invalid"},  # Should be float
        "start_date": "2023-01-01"
        # Missing end_date
    }

    response = test_client.post("/api/run", json=invalid_request)
    assert response.status_code == 422  # Validation error


@pytest.mark.api
@pytest.mark.unit
def test_run_analysis_endpoint_invalid_coordinates(test_client, invalid_coordinates):
    """Test the /api/run endpoint with invalid coordinates."""
    for invalid_coord in invalid_coordinates:
        if isinstance(invalid_coord["lat"], (int, float)) and isinstance(
            invalid_coord["lng"], (int, float)
        ):
            test_request = {
                "aoi": invalid_coord,
                "start_date": "2023-01-01",
                "end_date": "2023-01-31",
            }

            response = test_client.post("/api/run", json=test_request)
            # Should either return validation error or error status in response
            assert response.status_code in [200, 422]

            if response.status_code == 200:
                result = response.json()
                assert result["status"] == "error"
