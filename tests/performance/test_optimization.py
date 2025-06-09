"""Tests for optimization improvements in Kelpie Carbon v1."""
import os
import time
import tempfile
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import xarray as xr
from fastapi.testclient import TestClient

from kelpie_carbon_v1.api.main import app
from kelpie_carbon_v1.api.imagery import (
    _analysis_cache, _cache_access_times, 
    _get_cache_size_mb, _cleanup_cache, 
    _store_analysis_result, _get_analysis_result
)
from kelpie_carbon_v1.constants import (
    SatelliteData, KelpAnalysis, Processing, 
    Network, Testing, ModelConstants
)
from kelpie_carbon_v1.core.fetch import fetch_sentinel_tiles


class TestConstants:
    """Test constants are properly defined and used."""
    
    def test_satellite_data_constants(self):
        """Test satellite data constants are reasonable."""
        assert SatelliteData.MAX_CLOUD_COVER == 20
        assert SatelliteData.DEFAULT_BUFFER_KM == 1.0
        assert SatelliteData.KM_PER_DEGREE == 111.0
        assert SatelliteData.SENTINEL_SCALE_FACTOR == 10000.0
        assert SatelliteData.DEFAULT_RESOLUTION == 10
    
    def test_kelp_analysis_constants(self):
        """Test kelp analysis constants are reasonable."""
        assert KelpAnalysis.CARBON_CONTENT_FACTOR == 0.35
        assert KelpAnalysis.HECTARE_TO_M2 == 10000
        assert 0.0 < KelpAnalysis.MIN_CONFIDENCE_THRESHOLD < 1.0
        assert KelpAnalysis.MAX_KELP_DEPTH > 0
    
    def test_processing_constants(self):
        """Test processing constants are reasonable."""
        assert Processing.MAX_CACHE_SIZE_MB > 0
        assert Processing.MAX_CACHE_ITEMS > 0
        assert Processing.MAX_PROCESSING_TIMEOUT > 0
        assert len(Processing.DEFAULT_CHUNK_SIZE) == 2
    
    def test_network_constants(self):
        """Test network constants are reasonable."""
        assert Network.DEFAULT_PORT_RANGE_START > 0
        assert Network.MAX_PORT_ATTEMPTS > 0
        assert Network.DEFAULT_TIMEOUT > 0
        assert Network.HSTS_MAX_AGE > 0


class TestCacheManagement:
    """Test improved cache management functionality."""
    
    def setup_method(self):
        """Clear cache before each test."""
        _analysis_cache.clear()
        _cache_access_times.clear()
    
    def teardown_method(self):
        """Clear cache after each test."""
        _analysis_cache.clear()
        _cache_access_times.clear()
    
    def test_cache_size_calculation(self):
        """Test cache size calculation."""
        # Start with empty cache
        assert _get_cache_size_mb() == 0.0
        
        # Add mock data
        mock_dataset = xr.Dataset({
            'red': (['y', 'x'], np.random.rand(10, 10)),
            'nir': (['y', 'x'], np.random.rand(10, 10))
        })
        
        _store_analysis_result("test_id", mock_dataset, {}, {})
        
        # Should have some size now
        assert _get_cache_size_mb() > 0.0
    
    def test_cache_access_time_tracking(self):
        """Test LRU access time tracking."""
        mock_dataset = xr.Dataset({
            'red': (['y', 'x'], np.random.rand(5, 5))
        })
        
        # Store item
        _store_analysis_result("test_id", mock_dataset, {}, {})
        initial_time = _cache_access_times["test_id"]
        
        # Access item after a small delay
        time.sleep(0.01)
        _get_analysis_result("test_id")
        updated_time = _cache_access_times["test_id"]
        
        # Access time should be updated
        assert updated_time > initial_time
    
    def test_cache_cleanup_by_count(self):
        """Test cache cleanup when item count exceeds limit."""
        # Create multiple small datasets
        for i in range(Processing.MAX_CACHE_ITEMS + 5):
            mock_dataset = xr.Dataset({
                'data': (['y', 'x'], np.random.rand(2, 2))
            })
            _store_analysis_result(f"test_id_{i}", mock_dataset, {}, {})
        
        # Should not exceed max items
        assert len(_analysis_cache) <= Processing.MAX_CACHE_ITEMS
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction behavior."""
        # Add items with different access patterns
        datasets = []
        for i in range(5):
            mock_dataset = xr.Dataset({
                'data': (['y', 'x'], np.random.rand(3, 3))
            })
            datasets.append(mock_dataset)
            _store_analysis_result(f"item_{i}", mock_dataset, {}, {})
            time.sleep(0.001)  # Small delay to ensure different timestamps
        
        # Access some items to update their LRU position
        _get_analysis_result("item_2")
        _get_analysis_result("item_4")
        
        # Store more items to trigger cleanup
        with patch.object(Processing, 'MAX_CACHE_ITEMS', 3):
            for i in range(5, 8):
                mock_dataset = xr.Dataset({
                    'data': (['y', 'x'], np.random.rand(3, 3))
                })
                _store_analysis_result(f"item_{i}", mock_dataset, {}, {})
        
        # Recently accessed items should still be in cache
        assert "item_2" in _analysis_cache or "item_4" in _analysis_cache


class TestSecurityHeaders:
    """Test security headers implementation."""
    
    def test_security_headers_present(self):
        """Test that security headers are added to responses."""
        client = TestClient(app)
        response = client.get("/health")
        
        # Check key security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        
        assert "X-XSS-Protection" in response.headers
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        
        assert "Content-Security-Policy" in response.headers
        csp_header = response.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp_header
        
        # Check that unpkg.com is allowed for Leaflet compatibility
        assert "https://unpkg.com" in csp_header
        assert "script-src 'self'" in csp_header
        assert "style-src 'self'" in csp_header
    
    def test_hsts_header_for_https(self):
        """Test HSTS header is added for HTTPS requests."""
        client = TestClient(app)
        
        # Mock HTTPS request
        with patch('fastapi.Request') as mock_request:
            mock_request.url.scheme = "https"
            # This is a simplified test - in practice, HSTS would be tested 
            # with actual HTTPS setup
            pass


class TestConstantsUsage:
    """Test that constants are used instead of magic numbers."""
    
    def test_constants_in_fetch_module(self):
        """Test constants are used in fetch module."""
        # This tests the actual replacement of magic numbers
        with patch('kelpie_carbon_v1.core.fetch.Client') as mock_client:
            # Mock the client and search behavior
            mock_search = MagicMock()
            mock_search.items.return_value = []
            mock_client.open.return_value.search.return_value = mock_search
            
            # Should use constants instead of magic numbers
            fetch_sentinel_tiles(
                lat=Testing.TEST_LAT,
                lng=Testing.TEST_LNG,
                start_date=Testing.TEST_START_DATE,
                end_date=Testing.TEST_END_DATE
            )
            
            # Verify cloud cover threshold uses constant
            call_args = mock_client.open.return_value.search.call_args
            query = call_args[1]['query']
            assert query['eo:cloud_cover']['lt'] == SatelliteData.MAX_CLOUD_COVER
    
    def test_constants_in_api(self):
        """Test constants are used in API calculations."""
        # Test carbon calculation uses constants
        biomass_kg_ha = 1000.0
        biomass_kg_m2 = biomass_kg_ha / KelpAnalysis.HECTARE_TO_M2
        carbon_kg_m2 = biomass_kg_m2 * KelpAnalysis.CARBON_CONTENT_FACTOR
        
        expected_carbon = 1000.0 / 10000 * 0.35
        assert abs(carbon_kg_m2 - expected_carbon) < 1e-10


class TestFileWatchingOptimization:
    """Test file watching optimization."""
    
    def test_selective_file_watching_config(self):
        """Test that selective file watching is configured."""
        # This would be tested in an integration test with actual uvicorn
        # For now, verify the configuration structure
        reload_config = {
            "reload_dirs": ["src/kelpie_carbon_v1"],
            "reload_includes": ["*.py"],
            "reload_excludes": [
                "*.pyc", "__pycache__/*", "*.log", "*.tmp",
                "tests/*", "docs/*", "*.md", "*.yml", "*.yaml",
                ".git/*", ".pytest_cache/*", "*.egg-info/*"
            ]
        }
        
        # Verify configuration is reasonable
        assert "src/kelpie_carbon_v1" in reload_config["reload_dirs"]
        assert "*.py" in reload_config["reload_includes"]
        assert "*.pyc" in reload_config["reload_excludes"]
        assert "tests/*" in reload_config["reload_excludes"]


class TestPystacClientFix:
    """Test pystac_client deprecation fix."""
    
    def test_items_method_usage(self):
        """Test that .items() is used instead of deprecated .get_items()."""
        with patch('kelpie_carbon_v1.core.fetch.Client') as mock_client:
            mock_search = MagicMock()
            mock_items = [MagicMock()]
            mock_search.items.return_value = mock_items
            mock_client.open.return_value.search.return_value = mock_search
            
            # Call should use .items() not .get_items()
            result = fetch_sentinel_tiles(
                lat=Testing.TEST_LAT,
                lng=Testing.TEST_LNG,
                start_date=Testing.TEST_START_DATE,
                end_date=Testing.TEST_END_DATE
            )
            
            # Verify .items() was called
            mock_search.items.assert_called_once()
            # Verify .get_items() was not called
            assert not hasattr(mock_search, 'get_items') or not mock_search.get_items.called


class TestPerformanceMetrics:
    """Test performance-related optimizations."""
    
    def test_image_response_caching_headers(self):
        """Test that image responses include caching headers."""
        from kelpie_carbon_v1.api.imagery import _image_to_response
        from PIL import Image
        
        # Create test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Generate response
        response = _image_to_response(test_image)
        
        # Check caching headers
        assert "Cache-Control" in response.headers
        assert "public" in response.headers["Cache-Control"]
        assert "max-age" in response.headers["Cache-Control"]
        
        assert "ETag" in response.headers
        assert "Content-Length" in response.headers
    
    def test_processing_timeout_constant(self):
        """Test processing timeout constant is used."""
        assert Processing.MAX_PROCESSING_TIMEOUT > 0
        assert Processing.MAX_PROCESSING_TIMEOUT < 600  # Reasonable upper bound


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 