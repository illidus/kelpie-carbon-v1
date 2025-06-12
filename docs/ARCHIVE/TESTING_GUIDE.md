# 🧪 Kelpie Carbon v1: Testing Guide

## **Overview**

The Kelpie Carbon v1 project includes comprehensive testing across all phases of development. This guide covers test strategies, implementation patterns, and best practices for maintaining code quality and reliability.

## **Testing Philosophy**

### **Test Pyramid Structure**
```
    🔺 E2E Tests (Few)
   🔸🔸 Integration Tests (Some)
  🔹🔹🔹 Unit Tests (Many)
```

- **Unit Tests (70%)**: Fast, isolated component testing
- **Integration Tests (20%)**: Component interaction testing
- **End-to-End Tests (10%)**: Full workflow validation

### **Testing Principles**
- **Fast Feedback**: Tests should run quickly during development
- **Reliable**: Tests should be deterministic and stable
- **Maintainable**: Tests should be easy to understand and modify
- **Comprehensive**: High coverage of critical business logic

## **Test Structure**

### **Test Organization**
```
tests/
├── unit/                           # Unit tests by module
│   ├── test_api.py                # API endpoint tests
│   ├── test_fetch.py              # Satellite data fetching
│   ├── test_model.py              # ML model tests
│   ├── test_imagery.py            # Image processing tests
│   ├── test_indices.py            # Spectral index calculations
│   └── test_mask.py               # Masking operations
├── integration/                    # Integration tests
│   ├── test_imagery_api.py        # API + image processing
│   ├── test_satellite_integration.py  # Real satellite data
│   └── test_web_interface.py      # Frontend integration
├── performance/                    # Performance tests
│   ├── test_phase5_performance.py # Phase 5 performance features
│   └── test_load_testing.py       # Load and stress tests
├── fixtures/                       # Test data and mocks
│   ├── sample_satellite_data.json
│   ├── mock_responses.py
│   └── test_images/
└── conftest.py                     # Pytest configuration
```

## **Running Tests**

### **Basic Test Execution**
```bash
# Run all tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run specific test file
poetry run pytest tests/test_api.py

# Run specific test function
poetry run pytest tests/test_api.py::test_run_analysis_endpoint

# Run tests matching pattern
poetry run pytest -k "test_satellite"
```

### **Coverage Reporting**
```bash
# Run with coverage
poetry run pytest --cov=src

# Generate HTML coverage report
poetry run pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### **Test Categories**
```bash
# Unit tests only
poetry run pytest tests/unit/

# Integration tests only
poetry run pytest tests/integration/

# Performance tests only
poetry run pytest tests/performance/

# Slow tests (marked with @pytest.mark.slow)
poetry run pytest -m slow

# Fast tests only
poetry run pytest -m "not slow"
```

## **Test Implementation Patterns**

### **Unit Test Example**
```python
# tests/unit/test_indices.py
import pytest
import numpy as np
from src.kelpie_carbon_v1.imagery.indices import calculate_ndvi, calculate_fai

class TestSpectralIndices:
    """Unit tests for spectral index calculations."""

    def test_calculate_ndvi_basic(self):
        """Test NDVI calculation with simple values."""
        # Arrange
        red = np.array([[0.1, 0.2], [0.3, 0.4]])
        nir = np.array([[0.5, 0.6], [0.7, 0.8]])

        # Act
        ndvi = calculate_ndvi(red, nir)

        # Assert
        expected = (nir - red) / (nir + red)
        np.testing.assert_array_almost_equal(ndvi, expected)

    def test_calculate_ndvi_edge_cases(self):
        """Test NDVI calculation with edge cases."""
        # Test with zeros (should handle division by zero)
        red = np.array([[0.0, 0.1]])
        nir = np.array([[0.0, 0.1]])

        ndvi = calculate_ndvi(red, nir)

        # First pixel should be NaN (0/0), second should be 0
        assert np.isnan(ndvi[0, 0])
        assert ndvi[0, 1] == 0.0

    @pytest.mark.parametrize("red,nir,expected", [
        (0.1, 0.5, 0.6667),
        (0.2, 0.8, 0.6),
        (0.5, 0.5, 0.0),
    ])
    def test_ndvi_parametrized(self, red, nir, expected):
        """Test NDVI with parametrized inputs."""
        result = calculate_ndvi(np.array([[red]]), np.array([[nir]]))
        np.testing.assert_almost_equal(result[0, 0], expected, decimal=4)
```

### **Integration Test Example**
```python
# tests/integration/test_imagery_api.py
import pytest
from fastapi.testclient import TestClient
from src.kelpie_carbon_v1.api.main import app

class TestImageryAPI:
    """Integration tests for imagery API endpoints."""

    def setup_method(self):
        self.client = TestClient(app)
        self.test_analysis_id = "test-analysis-123"

    def test_imagery_workflow_complete(self):
        """Test complete imagery generation workflow."""
        # Step 1: Generate imagery
        response = self.client.post("/api/imagery/analyze-and-cache", json={
            "lat": 34.4140,
            "lon": -119.8489,
            "start_date": "2023-06-01",
            "end_date": "2023-08-31"
        })

        assert response.status_code == 200
        result = response.json()
        analysis_id = result["analysis_id"]

        # Step 2: Get metadata
        metadata_response = self.client.get(f"/api/imagery/{analysis_id}/metadata")
        assert metadata_response.status_code == 200
        metadata = metadata_response.json()
        assert "available_layers" in metadata

        # Step 3: Get RGB image
        rgb_response = self.client.get(f"/api/imagery/{analysis_id}/rgb")
        assert rgb_response.status_code == 200
        assert rgb_response.headers["content-type"] == "image/jpeg"

        # Step 4: Get spectral index
        fai_response = self.client.get(f"/api/imagery/{analysis_id}/spectral/fai")
        assert fai_response.status_code == 200
        assert fai_response.headers["content-type"] == "image/png"
```

### **Performance Test Example**
```python
# tests/performance/test_load_testing.py
import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

class TestLoadPerformance:
    """Performance and load testing."""

    @pytest.mark.performance
    def test_api_response_time(self):
        """Test API response times under normal load."""
        client = TestClient(app)

        start_time = time.time()
        response = client.get("/")
        duration = time.time() - start_time

        assert response.status_code == 200
        assert duration < 1.0  # Should respond within 1 second

    @pytest.mark.performance
    @pytest.mark.slow
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        client = TestClient(app)

        def make_request():
            response = client.get("/")
            return response.status_code == 200

        # Test 10 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in futures]

        # All requests should succeed
        assert all(results)
```

## **Test Data and Fixtures**

### **Pytest Fixtures**
```python
# conftest.py
import pytest
import numpy as np
from unittest.mock import Mock

@pytest.fixture
def sample_satellite_data():
    """Provide sample satellite data for testing."""
    return {
        'red': np.random.rand(100, 100) * 0.3,
        'nir': np.random.rand(100, 100) * 0.8,
        'swir1': np.random.rand(100, 100) * 0.2,
        'red_edge': np.random.rand(100, 100) * 0.5
    }

@pytest.fixture
def mock_planetary_computer():
    """Mock Microsoft Planetary Computer API responses."""
    mock = Mock()
    mock.search.return_value.items.return_value = [
        Mock(id="test-scene", properties={
            "datetime": "2023-08-16T19:19:11.024000Z",
            "eo:cloud_cover": 0.001307
        })
    ]
    return mock

@pytest.fixture
def test_coordinates():
    """Provide standard test coordinates."""
    return {"lat": 34.4140, "lng": -119.8489}
```

### **Mock Data Management**
```python
# tests/fixtures/mock_responses.py
class MockSatelliteResponse:
    """Mock satellite data responses for testing."""

    @staticmethod
    def successful_analysis():
        return {
            "analysis_id": "test-123",
            "status": "completed",
            "biomass": "150.0 tons/hectare",
            "carbon": "75.0 tons C/hectare"
        }

    @staticmethod
    def successful_imagery():
        return {
            "analysis_id": "imagery-456",
            "available_layers": {
                "base_layers": ["rgb"],
                "spectral_indices": ["ndvi", "fai"],
                "masks": ["kelp", "water"]
            }
        }
```

## **Testing Strategies by Component**

### **API Testing**
```python
class TestAPIEndpoints:
    """Comprehensive API endpoint testing."""

    def test_input_validation(self):
        """Test input validation and error handling."""
        client = TestClient(app)

        # Test invalid coordinates
        response = client.post("/api/run", json={
            "aoi": {"lat": 91.0, "lng": -119.8489},  # Invalid latitude
            "start_date": "2023-06-01",
            "end_date": "2023-08-31"
        })
        assert response.status_code == 422

        # Test invalid date range
        response = client.post("/api/run", json={
            "aoi": {"lat": 34.4140, "lng": -119.8489},
            "start_date": "2023-08-31",
            "end_date": "2023-06-01"  # End before start
        })
        assert response.status_code == 422

    def test_error_handling(self):
        """Test error handling for various failure scenarios."""
        # Test with non-existent analysis ID
        client = TestClient(app)
        response = client.get("/api/imagery/non-existent-id/rgb")
        assert response.status_code == 404
```

### **Image Processing Testing**
```python
class TestImageProcessing:
    """Test image generation and processing functions."""

    def test_rgb_composite_generation(self, sample_satellite_data):
        """Test RGB composite image generation."""
        from src.kelpie_carbon_v1.core.generators import generate_rgb_composite

        # Create mock dataset
        dataset = Mock()
        dataset.red.values = sample_satellite_data['red']
        dataset.green.values = sample_satellite_data['red']  # Use red for green
        dataset.blue.values = sample_satellite_data['red']  # Use red for blue

        # Generate RGB composite
        image = generate_rgb_composite(dataset)

        # Verify image properties
        assert image.mode in ['RGB', 'RGBA']
        assert image.size[0] > 0 and image.size[1] > 0

    def test_mask_generation(self, sample_satellite_data):
        """Test mask overlay generation."""
        from src.kelpie_carbon_v1.core.overlays import generate_kelp_mask

        # Generate kelp mask
        mask = generate_kelp_mask(sample_satellite_data)

        # Verify mask properties
        assert mask.dtype == np.bool_
        assert mask.shape == sample_satellite_data['red'].shape
```

### **Machine Learning Testing**
```python
class TestMLModels:
    """Test machine learning components."""

    def test_feature_extraction(self, sample_satellite_data):
        """Test feature extraction for ML models."""
        from src.kelpie_carbon_v1.model import extract_features

        features = extract_features(sample_satellite_data)

        # Verify feature structure
        assert isinstance(features, dict)
        assert 'red_mean' in features
        assert 'ndvi_mean' in features
        assert all(isinstance(v, (int, float, list)) for v in features.values())

    @pytest.mark.slow
    def test_model_prediction(self, sample_satellite_data):
        """Test model prediction pipeline."""
        from src.kelpie_carbon_v1.model import predict_kelp_biomass

        # Mock the model to avoid loading actual model files
        with patch('src.kelpie_carbon_v1.model.load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.8])  # High kelp probability
            mock_load.return_value = mock_model

            result = predict_kelp_biomass(sample_satellite_data)

            assert 'biomass_estimate' in result
            assert isinstance(result['biomass_estimate'], (int, float))
```

## **Frontend Testing**

### **JavaScript Testing Strategy**
```javascript
// tests/frontend/test_layer_management.js
describe('Layer Management', () => {
    let layerManager;

    beforeEach(() => {
        // Setup mock Leaflet map
        const mockMap = {
            addLayer: jest.fn(),
            removeLayer: jest.fn(),
            fitBounds: jest.fn()
        };

        layerManager = new SatelliteLayerManager(mockMap);
    });

    test('should add RGB layer successfully', () => {
        const analysisId = 'test-123';
        const layer = layerManager.addRGBLayer(analysisId);

        expect(layer).toBeDefined();
        expect(layer._url).toContain(analysisId);
        expect(layer._url).toContain('/rgb');
    });

    test('should handle layer loading errors', async () => {
        // Mock failed network request
        global.fetch = jest.fn().mockRejectedValue(new Error('Network error'));

        const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

        await layerManager.loadAllLayers('invalid-id');

        expect(consoleSpy).toHaveBeenCalledWith(
            expect.stringContaining('Failed to load')
        );

        consoleSpy.mockRestore();
    });
});
```

### **Browser Testing with Selenium**
```python
# tests/browser/test_user_workflows.py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait

class TestUserWorkflows:
    """Browser-based end-to-end testing."""

    def setup_method(self):
        self.driver = webdriver.Chrome()
        self.wait = WebDriverWait(self.driver, 10)

    def teardown_method(self):
        self.driver.quit()

    def test_complete_analysis_workflow(self):
        """Test complete user workflow from start to finish."""
        # Navigate to application
        self.driver.get("http://localhost:8000")

        # Click on map to select AOI
        map_element = self.driver.find_element(By.ID, "map")
        map_element.click()

        # Set date range
        start_date = self.driver.find_element(By.ID, "start-date")
        start_date.send_keys("2023-06-01")

        end_date = self.driver.find_element(By.ID, "end-date")
        end_date.send_keys("2023-08-31")

        # Run analysis
        run_button = self.driver.find_element(By.ID, "run-analysis")
        run_button.click()

        # Wait for results
        results = self.wait.until(
            lambda d: d.find_element(By.ID, "results-section")
        )

        assert results.is_displayed()
```

## **Test Configuration**

### **Pytest Configuration**
```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --strict-markers
    --strict-config
    --verbose
    --tb=short
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    performance: marks tests as performance tests
    browser: marks tests requiring browser automation
    unit: marks tests as unit tests
filterwarnings =
    ignore::UserWarning
    ignore::DeprecationWarning
```

### **Coverage Configuration**
```ini
# .coveragerc
[run]
source = src/
omit =
    */tests/*
    */test_*
    */__pycache__/*
    */venv/*
    */migrations/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
```

## **Continuous Integration**

### **GitHub Actions Workflow**
```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install Poetry
      uses: snok/install-poetry@v1

    - name: Install dependencies
      run: poetry install

    - name: Run unit tests
      run: poetry run pytest tests/unit/ --cov=src --cov-report=xml

    - name: Run integration tests
      run: poetry run pytest tests/integration/

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## **Performance Testing**

### **Load Testing with Locust**
```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between

class KelpieUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """Setup for each simulated user."""
        self.analysis_id = None

    @task(3)
    def view_homepage(self):
        """Simulate viewing the homepage."""
        self.client.get("/")

    @task(1)
    def run_analysis(self):
        """Simulate running an analysis."""
        response = self.client.post("/api/run", json={
            "aoi": {"lat": 34.4140, "lng": -119.8489},
            "start_date": "2023-06-01",
            "end_date": "2023-08-31"
        })

        if response.status_code == 200:
            result = response.json()
            self.analysis_id = result.get("analysis_id")

    @task(2)
    def view_imagery(self):
        """Simulate viewing imagery layers."""
        if self.analysis_id:
            self.client.get(f"/api/imagery/{self.analysis_id}/rgb")
            self.client.get(f"/api/imagery/{self.analysis_id}/spectral/fai")
```

### **Memory and Performance Profiling**
```python
# tests/performance/test_memory_usage.py
import psutil
import pytest
from memory_profiler import profile

class TestMemoryUsage:
    """Test memory usage and performance characteristics."""

    @profile
    def test_image_generation_memory(self):
        """Profile memory usage during image generation."""
        from src.kelpie_carbon_v1.core.generators import generate_rgb_composite

        # Monitor memory before
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Generate large image
        # ... test implementation

        # Monitor memory after
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Assert reasonable memory usage (less than 500MB increase)
        assert memory_increase < 500
```

## **Test Data Management**

### **Test Database Setup**
```python
# tests/fixtures/database.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="session")
def test_db():
    """Create test database for integration tests."""
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(bind=engine)

    # Create tables
    Base.metadata.create_all(bind=engine)

    yield TestingSessionLocal

    # Cleanup
    Base.metadata.drop_all(bind=engine)
```

### **Mock External Services**
```python
# tests/fixtures/mock_services.py
import responses
import json

@responses.activate
def test_with_mocked_planetary_computer():
    """Test with mocked external API calls."""
    # Mock Planetary Computer API
    responses.add(
        responses.GET,
        "https://planetarycomputer.microsoft.com/api/stac/v1/search",
        json={"features": [{"id": "test-scene"}]},
        status=200
    )

    # Run test that uses Planetary Computer
    # ... test implementation
```

## **Best Practices**

### **Test Writing Guidelines**
1. **Arrange-Act-Assert**: Structure tests clearly
2. **One Assertion Per Test**: Focus on single behavior
3. **Descriptive Names**: Test names should explain what they test
4. **Independent Tests**: Tests should not depend on each other
5. **Fast Execution**: Unit tests should run in milliseconds

### **Test Maintenance**
1. **Regular Review**: Update tests when requirements change
2. **Remove Obsolete Tests**: Delete tests for removed features
3. **Refactor Test Code**: Apply same quality standards as production code
4. **Monitor Coverage**: Aim for 80%+ coverage on critical paths

### **Testing Anti-Patterns to Avoid**
- **Testing Implementation Details**: Test behavior, not internals
- **Excessive Mocking**: Don't mock everything, test real interactions
- **Flaky Tests**: Ensure tests are deterministic and reliable
- **Slow Test Suites**: Keep feedback loops fast
- **Unclear Test Failures**: Make assertion messages descriptive

---

This testing guide provides a comprehensive framework for maintaining high code quality and reliability in the Kelpie Carbon v1 project. Regular testing ensures the application remains robust and maintainable as it evolves.
