"""Pytest configuration and shared fixtures for Kelpie Carbon v1 tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator

from fastapi.testclient import TestClient
from kelpie_carbon_v1.api.main import app
from kelpie_carbon_v1.config import get_settings


@pytest.fixture(scope="session")
def test_client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture(scope="session")
def test_settings():
    """Get test configuration settings."""
    return get_settings()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_coordinates():
    """Sample coordinates for testing (Monterey Bay, CA)."""
    return {
        "lat": 36.8,
        "lng": -121.9,
        "start_date": "2023-08-01",
        "end_date": "2023-08-31",
    }


@pytest.fixture
def invalid_coordinates():
    """Invalid coordinates for testing error handling."""
    return [
        {"lat": 91.0, "lng": 0.0},  # Invalid latitude
        {"lat": 0.0, "lng": 181.0},  # Invalid longitude
        {"lat": "invalid", "lng": 0.0},  # Non-numeric latitude
    ]


# Test markers for categorizing tests
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (slower, with external dependencies)",
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests (slowest, full system)"
    )
    config.addinivalue_line("markers", "api: marks tests as API tests")
    config.addinivalue_line("markers", "core: marks tests as core functionality tests")
    config.addinivalue_line(
        "markers", "imagery: marks tests as imagery processing tests"
    )
    config.addinivalue_line("markers", "cli: marks tests as CLI tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
