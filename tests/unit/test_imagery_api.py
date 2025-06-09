"""Tests for satellite imagery API endpoints."""
import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from PIL import Image
import io

from kelpie_carbon_v1.api.main import app
from kelpie_carbon_v1.api.imagery import _analysis_cache


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
        
        assert data['status'] == 'ok'
        assert data['service'] == 'imagery'
        assert 'cached_analyses' in data
        assert 'supported_formats' in data
        assert 'available_colormaps' in data

    def test_get_nonexistent_analysis(self):
        """Test accessing non-existent analysis."""
        response = self.client.get("/api/imagery/nonexistent/rgb")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"] 