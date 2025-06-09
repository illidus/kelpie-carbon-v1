# Kelpie Carbon v1 Development Makefile

.PHONY: help install test test-unit test-integration test-slow lint format clean serve docs

# Default target
help:
	@echo "Kelpie Carbon v1 Development Commands"
	@echo "====================================="
	@echo ""
	@echo "Setup:"
	@echo "  install     Install dependencies with Poetry"
	@echo "  install-dev Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test        Run all tests"
	@echo "  test-unit   Run only unit tests (fast)"
	@echo "  test-integration  Run integration tests"
	@echo "  test-slow   Run slow tests"
	@echo "  test-models Run model validation tests"
	@echo "  test-config Run configuration tests"
	@echo "  test-cov    Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint        Run linting (flake8, mypy)"
	@echo "  format      Format code (black, isort)"
	@echo "  check       Run all quality checks"
	@echo ""
	@echo "Development:"
	@echo "  serve       Start development server"
	@echo "  serve-auto  Start server with auto port detection"
	@echo "  clean       Clean temporary files"
	@echo ""
	@echo "Documentation:"
	@echo "  docs        Generate documentation"

# Installation
install:
	poetry install

install-dev:
	poetry install --with dev
	poetry run pre-commit install

# Testing
test:
	poetry run pytest

test-unit:
	poetry run pytest -m "unit"

test-integration:
	poetry run pytest -m "integration"

test-slow:
	poetry run pytest -m "slow"

test-models:
	poetry run pytest tests/test_models.py tests/test_model.py -v

test-config:
	poetry run pytest tests/test_simple_config.py -v

test-cov:
	poetry run pytest --cov=src/kelpie_carbon_v1 --cov-report=html --cov-report=term

# Code Quality
lint:
	poetry run flake8 src tests
	poetry run mypy src

format:
	poetry run black src tests
	poetry run isort src tests

check: format lint test-unit
	@echo "All quality checks passed!"

# Development
serve:
	poetry run kelpie-carbon-v1 serve --reload

serve-auto:
	poetry run kelpie-carbon-v1 serve --reload --auto-port

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml

# Documentation
docs:
	@echo "Documentation available in docs/ directory"
	@echo "API docs available at http://localhost:8000/docs when server is running"

# Quick development setup
setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make serve' to start the development server" 