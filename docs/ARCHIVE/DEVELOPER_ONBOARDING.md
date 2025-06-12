# 🚀 Developer Onboarding Guide

Welcome to the Kelpie Carbon v1 project! This guide will help you get up and running quickly.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.12+** - [Download here](https://www.python.org/downloads/)
- **Poetry** - [Installation guide](https://python-poetry.org/docs/#installation)
- **Git** - [Download here](https://git-scm.com/downloads)
- **Modern web browser** (Chrome, Firefox, Safari, Edge)

## 🏁 Quick Start (5 minutes)

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd kelpie-carbon-v1

# Install dependencies and setup development environment
make setup
```

### 2. Start Development Server
```bash
# Start server with automatic port detection
make serve-auto

# Or use the CLI directly
poetry run kelpie-carbon-v1 serve --auto-port --reload
```

### 3. Verify Installation
- Open http://localhost:8000 in your browser
- You should see the Kelpie Carbon v1 web interface
- API documentation is available at http://localhost:8000/docs

### 4. Run Tests
```bash
# Run quick unit tests
make test-unit

# Run all tests
make test
```

## 📁 Project Structure

```
kelpie-carbon-v1/
├── 📁 src/kelpie_carbon_v1/     # Main application code
│   ├── 📁 api/                  # FastAPI backend
│   │   ├── main.py             # API entry point
│   │   └── imagery.py          # Imagery endpoints
│   ├── 📁 core/                 # Core processing modules
│   │   ├── fetch.py            # Satellite data fetching
│   │   ├── model.py            # ML models
│   │   ├── mask.py             # Masking operations
│   │   └── indices.py          # Spectral calculations
│   ├── 📁 imagery/              # Image processing
│   │   ├── generators.py       # Image generation
│   │   ├── overlays.py         # Analysis overlays
│   │   └── utils.py            # Utilities
│   ├── 📁 web/static/           # Frontend files
│   │   ├── index.html          # Main interface
│   │   ├── app.js              # Application logic
│   │   └── *.js, *.css         # Other frontend assets
│   ├── config.py               # Configuration management
│   ├── logging_config.py       # Logging setup
│   └── cli.py                  # Command line interface
├── 📁 tests/                    # Test suite
├── 📁 docs/                     # Documentation
├── 📁 config/                   # Configuration files
├── pyproject.toml              # Project configuration
├── Makefile                    # Development commands
└── README.md                   # Project overview
```

## 🛠️ Development Workflow

### Daily Development
```bash
# Start your day
make serve-auto          # Start development server

# Make changes to code...

# Check your work
make check              # Format, lint, and test
```

### Testing
```bash
make test-unit          # Fast unit tests (< 30 seconds)
make test-integration   # Integration tests (1-2 minutes)
make test-slow          # Full test suite (2-5 minutes)
make test-cov           # Tests with coverage report
```

### Code Quality
```bash
make format             # Format code with black & isort
make lint               # Check code with flake8 & mypy
make check              # Run all quality checks
```

### Common Commands
```bash
# CLI commands
poetry run kelpie-carbon-v1 --help
poetry run kelpie-carbon-v1 serve --help
poetry run kelpie-carbon-v1 analyze --help

# Direct API testing
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

## 🧪 Testing Strategy

### Test Categories
- **Unit Tests** (`@pytest.mark.unit`) - Fast, isolated tests
- **Integration Tests** (`@pytest.mark.integration`) - Component interaction tests
- **E2E Tests** (`@pytest.mark.e2e`) - Full system tests
- **Slow Tests** (`@pytest.mark.slow`) - Long-running tests

### Running Specific Tests
```bash
# By category
pytest -m "unit"
pytest -m "api"
pytest -m "core"

# By file
pytest tests/test_api.py
pytest tests/test_cli.py

# Specific test
pytest tests/test_api.py::test_health_endpoint
```

## 🏗️ Architecture Overview

### Backend (Python/FastAPI)
- **FastAPI** - Modern, fast web framework
- **Pydantic** - Data validation and serialization
- **NumPy/SciPy** - Scientific computing
- **Scikit-learn** - Machine learning
- **Rasterio/Xarray** - Geospatial data processing

### Frontend (Vanilla JavaScript)
- **Leaflet.js** - Interactive mapping
- **Vanilla JS** - No framework dependencies
- **Progressive loading** - Optimized user experience

### Data Sources
- **Microsoft Planetary Computer** - Sentinel-2 satellite data
- **ESA Copernicus** - Earth observation data

## 🔧 Configuration

### Environment Variables
```bash
# Set environment (development is default)
export KELPIE_ENV=development

# Custom configuration
export KELPIE_LOG_LEVEL=DEBUG
export KELPIE_PORT=8001
```

### Configuration Files
- `config/base.yml` - Common settings
- `config/development.yml` - Development overrides
- `config/production.yml` - Production settings

## 🐛 Debugging

### Logging
```python
from kelpie_carbon_v1.logging_config import get_logger

logger = get_logger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### Common Issues

#### Port Already in Use
```bash
# Use auto port detection
poetry run kelpie-carbon-v1 serve --auto-port

# Or specify different port
poetry run kelpie-carbon-v1 serve --port 8001
```

#### Import Errors
```bash
# Reinstall dependencies
poetry install

# Check Python path
poetry run python -c "import sys; print(sys.path)"
```

#### Test Failures
```bash
# Run with verbose output
pytest -v

# Run specific failing test
pytest tests/test_api.py::test_health_endpoint -v
```

## 📚 Learning Resources

### Project Documentation
- [README.md](../README.md) - Project overview
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [API_REFERENCE.md](API_REFERENCE.md) - API documentation
- [USER_GUIDE.md](USER_GUIDE.md) - End-user guide

### External Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Leaflet Documentation](https://leafletjs.com/reference.html)

## 🤝 Contributing

### Before Making Changes
1. Create a new branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run quality checks: `make check`
4. Write/update tests
5. Update documentation if needed

### Code Style
- Follow PEP 8 (enforced by black and flake8)
- Use type hints for all functions
- Write comprehensive docstrings
- Add tests for new functionality

### Commit Messages
```
feat: add new satellite data source
fix: resolve port binding issue
docs: update API documentation
test: add unit tests for core module
refactor: simplify configuration system
```

## 🆘 Getting Help

### Internal Resources
1. Check existing documentation in `docs/`
2. Look at test files for usage examples
3. Review code comments and docstrings

### External Help
1. Check GitHub issues
2. Review FastAPI/Poetry documentation
3. Ask team members

### Reporting Issues
When reporting bugs, include:
- Python version: `python --version`
- Poetry version: `poetry --version`
- Operating system
- Error messages and stack traces
- Steps to reproduce

## 🎯 Next Steps

After completing this onboarding:

1. **Explore the codebase** - Read through the main modules
2. **Run the full test suite** - `make test`
3. **Try the CLI commands** - `poetry run kelpie-carbon-v1 --help`
4. **Make a small change** - Fix a typo or add a comment
5. **Read the architecture docs** - Understand the system design
