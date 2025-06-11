"""Tests for satellite imagery API endpoints."""

import io
from unittest.mock import Mock, patch

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient
from PIL import Image

from kelpie_carbon_v1.api.imagery import _analysis_cache
from kelpie_carbon_v1.api.main import app


class TestImageryAPI:
    """Test imagery API endpoints."""

    def setup_method(self):
        """Set up test client and mock data."""
        self.client = TestClient(app)

        # Clear cache before each test
        _analysis_cache.clear()

    def test_imagery_health_check(self):
        """Test imagery service health check."""
        response = self.client.get("/api/imagery/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "ok"
        assert data["service"] == "imagery"

    def test_get_nonexistent_analysis(self):
        """Test accessing non-existent analysis."""
        response = self.client.get("/api/imagery/nonexistent/rgb")

        assert response.status_code == 404
        error_data = response.json()
        assert "detail" in error_data
        assert "error" in error_data["detail"]
        assert "not found" in error_data["detail"]["error"]["message"].lower()
