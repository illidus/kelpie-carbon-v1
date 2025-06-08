"""Tests for API module."""
import pytest
from httpx import AsyncClient

from kelpie_carbon_v1.api.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test that /health returns 200 and correct JSON."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
