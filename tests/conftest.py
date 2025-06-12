import asyncio

import numpy as np
import pytest
import xarray as xr

# ===== SESSION-SCOPED FIXTURES FOR PERFORMANCE =====


@pytest.fixture(scope="session")
def sample_historical_dataset():
    """Minimal historical dataset for testing - loaded once per session."""
    # Create minimal test data instead of loading full dataset
    data = {
        "kelp_biomass": (
            ["time", "lat", "lon"],
            np.random.rand(12, 50, 50),
        ),  # 50x50 instead of 10000x10000
        "temperature": (["time", "lat", "lon"], 20 + 5 * np.random.rand(12, 50, 50)),
        "salinity": (["time", "lat", "lon"], 30 + 5 * np.random.rand(12, 50, 50)),
    }
    coords = {
        "time": np.arange(12),
        "lat": np.linspace(48.0, 49.0, 50),
        "lon": np.linspace(-124.0, -123.0, 50),
    }
    return xr.Dataset(data, coords=coords)


@pytest.fixture(scope="session")
def sample_sentinel_array():
    """Minimal Sentinel array data - loaded once per session."""
    return np.random.rand(50, 50, 4)  # 50x50 instead of full resolution


@pytest.fixture(scope="session")
def mock_fastapi_client():
    """Session-scoped FastAPI test client to avoid repeated startup."""
    from fastapi.testclient import TestClient

    from src.kelpie_carbon.api.main import app

    # Configure for test mode
    app.state.testing = True
    return TestClient(app)


@pytest.fixture(scope="session")
def optimized_cache():
    """Pre-warmed cache for session."""
    cache = {}
    # Pre-populate with common test data
    cache["test_coordinates"] = (48.5, -123.5)
    cache["test_date_range"] = ("2023-01-01", "2023-12-31")
    return cache


# ===== MOCK FIXTURES TO ELIMINATE SLOW OPERATIONS =====


@pytest.fixture
def mock_sleep(monkeypatch):
    """Replace all sleep operations with immediate returns."""
    import time

    def instant_sleep(duration):
        pass

    async def instant_async_sleep(duration):
        pass

    monkeypatch.setattr(time, "sleep", instant_sleep)
    monkeypatch.setattr(asyncio, "sleep", instant_async_sleep)


@pytest.fixture
def mock_network_requests(monkeypatch):
    """Mock all network requests for speed."""
    import requests

    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self.json_data = json_data
            self.status_code = status_code
            self.content = b"mock content"

        def json(self):
            return self.json_data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError()

    def mock_get(*args, **kwargs):
        return MockResponse({"status": "success", "data": []})

    def mock_post(*args, **kwargs):
        return MockResponse({"status": "created", "data": {"id": 1}})

    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests, "post", mock_post)


@pytest.fixture
def mock_satellite_data():
    """Mock satellite data fetching to avoid real API calls."""
    return {
        "bands": np.random.rand(50, 50, 4),
        "metadata": {
            "date": "2023-06-01",
            "cloud_coverage": 0.1,
            "coordinates": (48.5, -123.5),
        },
    }


# ===== PERFORMANCE OPTIMIZATION FIXTURES =====


@pytest.fixture(autouse=True)
def performance_mode():
    """Auto-applied fixture to enable performance optimizations."""
    import os

    os.environ["TESTING_MODE"] = "performance"
    yield
    if "TESTING_MODE" in os.environ:
        del os.environ["TESTING_MODE"]
