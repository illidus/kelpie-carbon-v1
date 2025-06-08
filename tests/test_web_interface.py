"""Tests for web interface functionality."""
import pytest
from fastapi.testclient import TestClient

from kelpie_carbon_v1.api.main import app


def test_root_serves_html():
    """Test that root endpoint serves HTML content."""
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_static_css_accessible():
    """Test that CSS file is accessible via static mount."""
    client = TestClient(app)
    response = client.get("/static/style.css")

    assert response.status_code == 200
    assert "text/css" in response.headers.get("content-type", "")

    # Check that CSS contains expected styles
    css_content = response.text
    assert "body" in css_content
    assert "map" in css_content
    assert "#app" in css_content


def test_static_js_accessible():
    """Test that JavaScript file is accessible via static mount."""
    client = TestClient(app)
    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "application/javascript" in response.headers.get(
        "content-type", ""
    ) or "text/javascript" in response.headers.get("content-type", "")

    # Check that JS contains expected functionality
    js_content = response.text
    assert "map" in js_content.lower()
    assert "leaflet" in js_content.lower() or "L.map" in js_content
    assert "fetch" in js_content  # For API calls


def test_html_references_correct_static_paths():
    """Test that HTML file references static files with correct paths."""
    client = TestClient(app)
    response = client.get("/")

    html_content = response.text

    # Check that HTML references static files correctly
    assert "/static/style.css" in html_content
    assert "/static/app.js" in html_content

    # Check for Leaflet CDN references
    assert "leaflet@1.9.4" in html_content


def test_html_contains_map_elements():
    """Test that HTML contains required elements for map functionality."""
    client = TestClient(app)
    response = client.get("/")

    html_content = response.text

    # Check for required HTML elements
    assert 'id="map"' in html_content
    assert 'id="run-analysis"' in html_content
    assert 'id="start-date"' in html_content
    assert 'id="end-date"' in html_content
    assert 'id="status"' in html_content
    assert 'id="results"' in html_content


def test_html_has_proper_structure():
    """Test that HTML has proper structure for the application."""
    client = TestClient(app)
    response = client.get("/")

    html_content = response.text

    # Check basic HTML structure
    assert "<!DOCTYPE html>" in html_content
    assert '<html lang="en">' in html_content
    assert "<head>" in html_content
    assert "<body>" in html_content
    assert "<title>Kelpie Carbon v1" in html_content

    # Check for key sections
    assert "Kelpie Carbon v1" in html_content
    assert "Kelp Forest Carbon Sequestration Assessment" in html_content
    assert "Click on the map to select Area of Interest" in html_content


def test_nonexistent_static_file_returns_404():
    """Test that non-existent static files return 404."""
    client = TestClient(app)
    response = client.get("/static/nonexistent.css")

    assert response.status_code == 404


@pytest.mark.parametrize(
    "path,expected_type",
    [
        ("/static/style.css", "text/css"),
        ("/static/app.js", "javascript"),
    ],
)
def test_static_file_content_types(path, expected_type):
    """Test that static files return correct content types."""
    client = TestClient(app)
    response = client.get(path)

    if response.status_code == 200:  # Only test if file exists
        content_type = response.headers.get("content-type", "")
        assert expected_type in content_type


def test_javascript_contains_required_functions():
    """Test that JavaScript contains required functions for map and API interaction."""
    client = TestClient(app)
    response = client.get("/static/app.js")

    if response.status_code == 200:
        js_content = response.text

        # Check for essential JavaScript functionality
        assert (
            "L.map" in js_content or "map =" in js_content
        )  # Leaflet map initialization
        assert "fetch(" in js_content or "XMLHttpRequest" in js_content  # API calls
        assert "click" in js_content  # Event handling
        assert "/api/run" in js_content  # API endpoint reference


def test_css_contains_required_styles():
    """Test that CSS contains required styles for proper layout."""
    client = TestClient(app)
    response = client.get("/static/style.css")

    if response.status_code == 200:
        css_content = response.text

        # Check for essential CSS selectors
        assert "#map" in css_content  # Map container styling
        assert "#run-analysis" in css_content  # Button styling
        assert ".controls" in css_content  # Control panel styling
        assert "height:" in css_content  # Map must have height defined
