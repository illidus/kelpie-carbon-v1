import asyncio
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
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
def sentinel_tile():
    """Session-scoped Sentinel-2 tile fixture with minimal data for performance."""
    # Create minimal but realistic Sentinel-2 data structure
    data = {
        "red": (["y", "x"], np.random.rand(50, 50) * 0.3),  # Typical reflectance values
        "green": (["y", "x"], np.random.rand(50, 50) * 0.3),
        "blue": (["y", "x"], np.random.rand(50, 50) * 0.3),
        "nir": (["y", "x"], np.random.rand(50, 50) * 0.4),
        "red_edge": (["y", "x"], np.random.rand(50, 50) * 0.35),
        "swir1": (["y", "x"], np.random.rand(50, 50) * 0.25),
    }
    coords = {
        "y": np.linspace(5400000, 5450000, 50),  # UTM coordinates
        "x": np.linspace(300000, 350000, 50),
    }

    dataset = xr.Dataset(data, coords=coords)
    dataset.attrs.update(
        {
            "cloud_cover": 5.0,
            "acquisition_date": "2023-08-15",
            "source": "Sentinel-2",
            "crs": "EPSG:32610",
        }
    )
    return dataset


@pytest.fixture(scope="session")
def rf_model():
    """Session-scoped Random Forest model fixture - trained once per session."""
    from sklearn.ensemble import RandomForestRegressor

    # Create minimal training data
    n_samples = 100  # Reduced from typical 1000+ samples
    n_features = 20  # Reduced from typical 50+ features

    x = np.random.rand(n_samples, n_features)
    # Create realistic biomass targets (0-5000 kg/ha)
    y = np.random.exponential(scale=800, size=n_samples)
    y = np.clip(y, 0, 5000)

    # Train minimal model
    model = RandomForestRegressor(
        n_estimators=10,  # Reduced from typical 100+
        max_depth=5,  # Reduced depth for speed
        random_state=42,
        n_jobs=1,  # Single thread for consistency
    )
    model.fit(x, y)

    return {
        "model": model,
        "feature_names": [f"feature_{i}" for i in range(n_features)],
        "training_score": model.score(x, y),
        "n_samples": n_samples,
        "n_features": n_features,
    }


@pytest.fixture(scope="session")
def minimal_training_data():
    """Session-scoped minimal training dataset for model tests."""
    # Create 10 samples instead of typical 100+
    training_samples = []

    for _i in range(10):
        # Create minimal xarray dataset
        data = {
            "red": (["y", "x"], np.random.rand(20, 20) * 0.3),
            "nir": (["y", "x"], np.random.rand(20, 20) * 0.4),
            "kelp_mask": (
                ["y", "x"],
                np.random.choice([0, 1], size=(20, 20), p=[0.7, 0.3]),
            ),
        }
        coords = {"y": np.arange(20), "x": np.arange(20)}
        dataset = xr.Dataset(data, coords=coords)

        # Random biomass target
        biomass = np.random.exponential(scale=500)
        training_samples.append((dataset, biomass))

    return training_samples


@pytest.fixture(scope="session")
def mock_fastapi_client():
    """Session-scoped FastAPI test client to avoid repeated startup."""
    from fastapi.testclient import TestClient

    from src.kelpie_carbon.core.api.main import app

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
    cache["monterey_bay"] = (36.8, -121.9)
    cache["vancouver_island"] = (49.2827, -123.1207)
    return cache


@pytest.fixture(scope="session")
def temp_data_dir():
    """Session-scoped temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create some test files
        (temp_path / "test_data.json").write_text('{"test": "data"}')
        (temp_path / "validation_results.json").write_text('{"accuracy": 0.95}')

        yield temp_path


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
    import httpx
    import requests

    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self.json_data = json_data
            self.status_code = status_code
            self.content = b"mock content"
            self.text = "mock text"

        def json(self):
            return self.json_data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError()

    class MockAsyncResponse:
        def __init__(self, json_data, status_code=200):
            self.json_data = json_data
            self.status_code = status_code
            self.content = b"mock content"
            self.text = "mock text"

        def json(self):
            return self.json_data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("Mock error", request=None, response=self)

    def mock_get(*args, **kwargs):
        return MockResponse({"status": "success", "data": []})

    def mock_post(*args, **kwargs):
        return MockResponse({"status": "created", "data": {"id": 1}})

    async def mock_async_get(*args, **kwargs):
        return MockAsyncResponse({"status": "success", "data": []})

    async def mock_async_post(*args, **kwargs):
        return MockAsyncResponse({"status": "created", "data": {"id": 1}})

    # Mock requests library
    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests, "post", mock_post)

    # Mock httpx library
    monkeypatch.setattr(httpx, "get", mock_async_get)
    monkeypatch.setattr(httpx, "post", mock_async_post)


@pytest.fixture
def mock_satellite_data(sentinel_tile):
    """Mock satellite data fetching to avoid real API calls."""
    return {
        "data": sentinel_tile,
        "bands": sentinel_tile.to_array().values,
        "metadata": {
            "date": "2023-06-01",
            "cloud_coverage": 0.1,
            "coordinates": (48.5, -123.5),
            "source": "mock_sentinel",
        },
    }


@pytest.fixture
def mock_heavy_computation(monkeypatch):
    """Mock computationally expensive operations."""

    def fast_model_training(*args, **kwargs):
        """Mock model training that returns immediately."""
        return {
            "model": "mock_trained_model",
            "metrics": {"accuracy": 0.95, "r2": 0.88},
            "training_time": 0.001,  # Instant
        }

    def fast_feature_extraction(dataset, *args, **kwargs):
        """Mock feature extraction with minimal computation."""
        # Return minimal feature set based on dataset size
        n_pixels = dataset.sizes.get("x", 10) * dataset.sizes.get("y", 10)
        features = pd.DataFrame(
            {f"feature_{i}": np.random.rand(1) for i in range(min(10, n_pixels // 10))}
        )
        return features

    # Apply mocks to common heavy operations
    try:
        from kelpie_carbon.core.model import KelpBiomassModel

        monkeypatch.setattr(KelpBiomassModel, "train", fast_model_training)
        monkeypatch.setattr(
            KelpBiomassModel, "extract_features", fast_feature_extraction
        )
    except ImportError:
        pass  # Module might not exist in all test contexts


# ===== PERFORMANCE OPTIMIZATION FIXTURES =====


@pytest.fixture(autouse=True)
def performance_mode():
    """Auto-applied fixture to enable performance optimizations."""
    import os

    os.environ["TESTING_MODE"] = "performance"
    os.environ["MINIMAL_DATA_SIZE"] = "true"
    os.environ["DISABLE_SLOW_OPERATIONS"] = "true"
    yield
    # Cleanup
    for key in ["TESTING_MODE", "MINIMAL_DATA_SIZE", "DISABLE_SLOW_OPERATIONS"]:
        if key in os.environ:
            del os.environ[key]


@pytest.fixture
def slice_data_to_minimal():
    """Utility fixture to slice large datasets to minimal size for testing."""

    def slicer(dataset, max_size=50):
        """Slice xarray dataset to maximum size for performance."""
        if isinstance(dataset, xr.Dataset):
            slices = {}
            for dim in dataset.dims:
                current_size = dataset.sizes[dim]
                if current_size > max_size:
                    slices[dim] = slice(0, max_size)
            if slices:
                return dataset.isel(slices)
        return dataset

    return slicer
