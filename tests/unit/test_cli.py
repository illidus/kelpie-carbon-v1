"""Tests for CLI module."""

import pytest
from typer.testing import CliRunner

from kelpie_carbon_v1.cli import app


@pytest.mark.cli
@pytest.mark.unit
def test_cli_help():
    """Test CLI help command."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Kelpie Carbon v1" in result.stdout


@pytest.mark.cli
@pytest.mark.unit
def test_version_command():
    """Test version command."""
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Kelpie Carbon v1" in result.stdout


@pytest.mark.cli
@pytest.mark.unit
def test_config_command():
    """Test config command."""
    runner = CliRunner()
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "Configuration" in result.stdout


@pytest.mark.cli
@pytest.mark.integration
@pytest.mark.slow
def test_analyze_command_invalid_coordinates():
    """Test analyze command with invalid coordinates."""
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "91.0", "0.0", "2023-01-01", "2023-01-31"])
    assert result.exit_code == 1  # Should fail with invalid latitude


@pytest.mark.cli
@pytest.mark.unit
def test_serve_command_help():
    """Test serve command help."""
    runner = CliRunner()
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "Start the Kelpie Carbon v1 web server" in result.stdout
    assert "--auto-port" in result.stdout
