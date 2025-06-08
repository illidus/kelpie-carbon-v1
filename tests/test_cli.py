"""Tests for CLI module."""
from typer.testing import CliRunner

from kelpie_carbon_v1.cli import app


def test_hello_command():
    """Test the hello command functionality."""
    runner = CliRunner()
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "lives" in result.stdout
