# Development Guide

This guide covers the development practices, architecture, and workflows for the Kelpie Carbon v1 project.

## 🏗️ **Architecture Overview**

### **Configuration Management**

The application uses a hierarchical configuration system:

```
config/
├── base.yml          # Common settings for all environments
├── development.yml   # Development-specific overrides
└── production.yml    # Production-specific overrides
```

**Usage:**
```python
from kelpie_carbon_v1.config import get_settings

settings = get_settings()  # Loads based on KELPIE_ENV
print(settings.server.port)
```

**Environment Variables:**
- `KELPIE_ENV`: Set environment (development/production)
- Configuration supports `${VAR_NAME}` syntax for environment variable expansion

### **Logging System**

Centralized logging with environment-specific configuration:

```python
from kelpie_carbon_v1.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Application started")
```

**Features:**
- Colored console output for development
- JSON structured logging for production
- Automatic log rotation
- Performance and API request logging utilities

### **Module Organization**

```
src/kelpie_carbon_v1/
├── core/              # Core business logic
│   ├── fetch.py       # Satellite data fetching
│   ├── model.py       # ML models and prediction
│   ├── mask.py        # Data masking and filtering
│   └── indices.py     # Spectral index calculations
├── api/               # FastAPI web API
├── imagery/           # Image processing utilities
├── web/               # Frontend static files
├── config.py          # Configuration management
├── logging_config.py  # Logging setup
└── cli.py             # Command-line interface
```

## 🛠️ **Development Workflow**

### **Setting Up Development Environment**

1. **Clone and Install:**
   ```bash
   git clone <repository>
   cd kelpie-carbon-v1
   poetry install
   poetry shell
   ```

2. **Set Development Environment:**
   ```bash
   export KELPIE_ENV=development
   ```

3. **Start Development Server:**
   ```bash
   poetry run kelpie-carbon-v1 serve --reload
   ```

### **Code Quality Tools**

The project uses pre-commit hooks for code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

**Tools included:**
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting with docstring checks
- **mypy**: Type checking
- **bandit**: Security scanning

### **Testing Strategy**

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific test categories
poetry run pytest tests/test_api.py
poetry run pytest tests/test_core/

# Using CLI
poetry run kelpie-carbon-v1 test --verbose
```

**Test Organization:**
- `tests/test_api.py`: API endpoint tests
- `tests/test_core/`: Core module tests
- `tests/test_integration.py`: End-to-end tests
- `tests/test_real_satellite_*.py`: Real data integration tests

## 📝 **Development Guidelines**

### **Adding New Features**

1. **Configuration First:**
   - Add new settings to `config/base.yml`
   - Update `config.py` with new configuration classes
   - Environment-specific overrides in `development.yml`/`production.yml`

2. **Core Logic:**
   - Add business logic to appropriate `core/` module
   - Update `core/__init__.py` to export new functions
   - Follow existing patterns for error handling and logging

3. **API Endpoints:**
   - Add new endpoints to `api/` modules
   - Use Pydantic models for request/response validation
   - Include proper error handling and logging

4. **Testing:**
   - Write unit tests for core logic
   - Add integration tests for API endpoints
   - Include real data tests where applicable

### **Code Style Guidelines**

**Imports:**
```python
# Standard library
import os
import time
from pathlib import Path

# Third-party
import numpy as np
from fastapi import FastAPI

# Local imports
from .config import get_settings
from .core import fetch_sentinel_tiles
```

**Logging:**
```python
from kelpie_carbon_v1.logging_config import get_logger

logger = get_logger(__name__)

def my_function():
    logger.info("Starting operation")
    try:
        # ... operation
        logger.info("Operation completed successfully")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise
```

**Configuration Usage:**
```python
from kelpie_carbon_v1.config import get_settings

def my_function():
    settings = get_settings()
    timeout = settings.satellite.timeout
    # ... use configuration
```

**Error Handling:**
```python
# Use specific exceptions
from fastapi import HTTPException

def api_endpoint():
    try:
        result = some_operation()
        return result
    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### **Performance Considerations**

1. **Caching:**
   - Use configuration-based cache settings
   - Implement proper cache invalidation
   - Monitor cache hit rates

2. **Logging:**
   - Use appropriate log levels
   - Avoid logging in tight loops
   - Use structured logging for production

3. **Resource Management:**
   - Close file handles and network connections
   - Use context managers where appropriate
   - Monitor memory usage in long-running operations

## 🚀 **Deployment**

### **Environment Configuration**

**Development:**
```bash
export KELPIE_ENV=development
poetry run kelpie-carbon-v1 serve
```

**Production:**
```bash
export KELPIE_ENV=production
export DATABASE_URL=postgresql://...
export REDIS_URL=redis://...
export SENTRY_DSN=https://...

poetry run kelpie-carbon-v1 serve --workers 4
```

### **Docker Deployment**

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev

COPY . .
ENV KELPIE_ENV=production

CMD ["poetry", "run", "kelpie-carbon-v1", "serve"]
```

### **Health Checks**

The application provides health check endpoints:

- `/health`: Basic health check
- `/ready`: Readiness check with dependency validation
- `/metrics`: Performance metrics (if enabled)

## 🔧 **Troubleshooting**

### **Common Issues**

1. **Configuration Not Loading:**
   ```bash
   # Check environment
   echo $KELPIE_ENV

   # Verify config files exist
   ls config/

   # Test configuration loading
   poetry run kelpie-carbon-v1 config
   ```

2. **Import Errors:**
   ```bash
   # Ensure you're in the poetry shell
   poetry shell

   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

3. **Port Already in Use:**
   ```bash
   # Use different port
   poetry run kelpie-carbon-v1 serve --port 8001

   # Or kill existing process
   lsof -ti:8000 | xargs kill -9
   ```

### **Debug Mode**

Enable debug logging:
```bash
export KELPIE_ENV=development
# Edit config/development.yml to set logging.level: DEBUG
poetry run kelpie-carbon-v1 serve
```

### **Performance Monitoring**

Access performance dashboard:
- Web UI: Press `Ctrl+Shift+P`
- CLI: `poetry run kelpie-carbon-v1 config` to see performance settings

## 📚 **Additional Resources**

- [API Reference](API_REFERENCE.md)
- [Testing Guide](TESTING_GUIDE.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Architecture Documentation](ARCHITECTURE.md)
