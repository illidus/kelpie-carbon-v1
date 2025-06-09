# 🧪 Kelpie Carbon v1 Test Suite

This directory contains a comprehensive test suite for the Kelpie Carbon v1 application, organized by test type for better maintainability and execution control.

## 📁 Test Structure

```
tests/
├── conftest.py                 # Shared pytest configuration and fixtures
├── __init__.py                 # Test package initialization
│
├── unit/                       # Unit tests (fast, isolated)
│   ├── __init__.py
│   ├── test_api.py            # API endpoint unit tests
│   ├── test_cli.py            # CLI interface tests
│   ├── test_fetch.py          # Data fetching logic tests
│   ├── test_imagery_api.py    # Imagery API tests
│   ├── test_imagery.py        # Image processing tests
│   ├── test_indices.py        # Spectral index calculation tests
│   ├── test_mask.py           # Masking operations tests
│   ├── test_model.py          # ML model tests
│   ├── test_models.py         # Data model validation tests
│   ├── test_simple_config.py  # Configuration tests
│   ├── test_validation.py     # Validation framework tests
│   └── test_web_interface.py  # Web interface tests
│
├── integration/                # Integration tests (with external deps)
│   ├── __init__.py
│   ├── test_integration.py                    # General integration tests
│   ├── test_real_satellite_data.py          # Real satellite data tests
│   ├── test_real_satellite_integration.py   # Satellite API integration
│   └── test_satellite_imagery_integration.py # Imagery pipeline integration
│
├── e2e/                       # End-to-end tests (complete workflows)
│   ├── __init__.py
│   └── test_integration_comprehensive.py    # Complete user workflows
│
└── performance/               # Performance tests (optimization & metrics)
    ├── __init__.py
    ├── test_optimization.py          # Optimization tests
    └── test_phase5_performance.py    # Performance metrics and monitoring
```

---

## 🚀 Running Tests

### Prerequisites
```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Test Categories

#### **Unit Tests** (Fast, Isolated)
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific unit test file
pytest tests/unit/test_api.py -v

# Run unit tests with coverage
pytest tests/unit/ --cov=src --cov-report=html
```

#### **Integration Tests** (External Dependencies)
```bash
# Run all integration tests
pytest tests/integration/ -v

# Run satellite data integration tests
pytest tests/integration/test_real_satellite_integration.py -v

# Skip integration tests if no internet
pytest tests/unit/ tests/e2e/ tests/performance/ -v
```

#### **End-to-End Tests** (Complete Workflows)
```bash
# Run end-to-end tests
pytest tests/e2e/ -v

# Run comprehensive workflow test
pytest tests/e2e/test_integration_comprehensive.py -v
```

#### **Performance Tests** (Optimization & Metrics)
```bash
# Run performance tests
pytest tests/performance/ -v

# Run optimization tests
pytest tests/performance/test_optimization.py -v
```

### All Tests
```bash
# Run complete test suite
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html --cov-report=term
```

---

## 🏷️ Test Markers

The test suite uses pytest markers for categorization:

```bash
# Run by marker
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m e2e           # End-to-end tests only
pytest -m slow          # Slow-running tests
pytest -m api           # API-related tests
pytest -m imagery       # Image processing tests
```

### Available Markers
- `unit` - Fast, isolated unit tests
- `integration` - Tests with external dependencies
- `e2e` - End-to-end workflow tests  
- `api` - API endpoint tests
- `core` - Core functionality tests
- `imagery` - Image processing tests
- `cli` - Command-line interface tests
- `slow` - Long-running tests

---

## 🛠️ Test Configuration

### Pytest Configuration (`pytest.ini`)
Key settings for the test suite:
- Test discovery patterns
- Marker definitions
- Coverage configuration
- Output formatting

### Shared Fixtures (`conftest.py`)
Common test fixtures available across all tests:
- `test_client` - FastAPI test client
- `test_settings` - Test configuration
- `temp_dir` - Temporary directory for test files
- `sample_coordinates` - Test coordinates (Monterey Bay)
- `invalid_coordinates` - Invalid test data

---

## 📊 Test Types Explained

### **Unit Tests** (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Characteristics**: Fast execution, no external dependencies
- **Mocking**: Heavy use of mocks for external services
- **Examples**: API endpoint logic, data models, calculations

### **Integration Tests** (`tests/integration/`)
- **Purpose**: Test component interactions and external services
- **Characteristics**: Slower execution, real external calls
- **Dependencies**: Internet connection, satellite data APIs
- **Examples**: Real satellite data fetching, API integrations

### **End-to-End Tests** (`tests/e2e/`)
- **Purpose**: Test complete user workflows
- **Characteristics**: Full system testing, user perspective
- **Scope**: From API request to final response
- **Examples**: Complete analysis workflow, web interface

### **Performance Tests** (`tests/performance/`)
- **Purpose**: Verify system performance and optimization
- **Characteristics**: Resource monitoring, timing analysis
- **Metrics**: Response times, memory usage, caching
- **Examples**: Image optimization, caching effectiveness

---

## 🎯 Best Practices

### Writing Tests
1. **Follow naming conventions**: `test_function_name` or `test_behavior`
2. **Use descriptive test names**: Clearly describe what's being tested
3. **Keep tests isolated**: Each test should be independent
4. **Use appropriate fixtures**: Leverage shared fixtures from `conftest.py`
5. **Mock external dependencies**: Use mocks for external services in unit tests

### Test Organization
1. **Group related tests**: Use test classes to group related functionality
2. **Use proper markers**: Tag tests with appropriate markers
3. **Write test documentation**: Include docstrings for complex tests
4. **Keep tests focused**: One test should verify one behavior

### Running Tests in CI/CD
```bash
# Fast feedback loop (unit tests only)
pytest tests/unit/ --maxfail=5

# Full test suite with coverage
pytest --cov=src --cov-report=xml --maxfail=10

# Performance regression testing
pytest tests/performance/ --benchmark-json=results.json
```

---

## 🔧 Troubleshooting Tests

### Common Issues

#### **Satellite Data Tests Failing**
```bash
# Check internet connection
ping planetarycomputer.microsoft.com

# Run without integration tests
pytest tests/unit/ tests/e2e/ tests/performance/
```

#### **Test Configuration Issues**
```bash
# Verify pytest installation
pytest --version

# Check test discovery
pytest --collect-only
```

#### **Performance Test Failures**
```bash
# Run with verbose output for debugging
pytest tests/performance/ -v -s

# Check system resources during tests
pytest tests/performance/ --capture=no
```

---

## 📈 Test Coverage

Target coverage goals:
- **Unit Tests**: 90%+ coverage for core modules
- **Integration Tests**: Cover all external API interactions
- **E2E Tests**: Cover all major user workflows
- **Performance Tests**: Cover optimization features

Generate coverage reports:
```bash
# HTML report (open htmlcov/index.html)
pytest --cov=src --cov-report=html

# Terminal report
pytest --cov=src --cov-report=term-missing
```

---

## 🔗 Related Documentation

- **[Testing Guide](../docs/TESTING_GUIDE.md)** - Comprehensive testing documentation
- **[Development Guide](../docs/DEVELOPMENT_GUIDE.md)** - Development practices
- **[API Reference](../docs/API_REFERENCE.md)** - API documentation for test development
- **[Contributing](../CONTRIBUTING.md)** - Contribution guidelines including test requirements 