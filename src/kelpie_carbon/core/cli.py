"""Command Line Interface for Kelpie Carbon v1.

This module provides comprehensive CLI functionality for kelp carbon monitoring
including data processing, analysis, and validation workflows.
"""

from __future__ import annotations

import json
import socket

import typer
import uvicorn

from .config import get_settings
from .constants import KelpAnalysis, Network
from .logging_config import get_logger, setup_logging

app = typer.Typer(
    name="kelpie-carbon-v1",
    help="Kelpie Carbon v1: Kelp Forest Carbon Sequestration Assessment CLI",
)

logger = get_logger(__name__)


def _find_available_port(
    start_port: int, max_attempts: int = Network.MAX_PORT_ATTEMPTS
) -> int:
    """Find an available port starting from start_port.

    Args:
        start_port: Port to start searching from
        max_attempts: Maximum number of ports to try

    Returns:
        First available port number

    Raises:
        RuntimeError: If no available port is found in the range

    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue

    raise RuntimeError(
        f"Could not find available port in range {start_port}-{start_port + max_attempts}"
    )


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    workers: int | None = None,
):
    """Start the Kelpie Carbon v1 web server."""
    # Setup logging first
    setup_logging()

    # Load configuration
    settings = get_settings()

    # Determine port to use
    target_port = port or settings.server.port

    if workers is None:
        workers = settings.server.workers

    logger.info(f"Starting Kelpie Carbon v1 server on {host}:{target_port}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

    try:
        uvicorn.run(
            "kelpie_carbon.core.api.main:app",
            host=host,
            port=target_port,
            reload=reload,
            log_level=settings.server.log_level.lower(),
            access_log=settings.server.access_log,
            workers=workers,
        )
    except OSError as e:
        if "Address already in use" in str(e) or "10048" in str(e):
            logger.error(
                f"Port {target_port} is already in use. Try using --auto-port flag or specify a different port with --port"
            )
            logger.info("Example: poetry run kelpie-carbon-v1 serve --auto-port")
            logger.info("Or: poetry run kelpie-carbon-v1 serve --port 8001")
        else:
            logger.error(f"Failed to start server: {e}")
        raise typer.Exit(1) from e


@app.command()
def analyze(
    lat: float = typer.Argument(..., help="Latitude of the area of interest"),
    lng: float = typer.Argument(..., help="Longitude of the area of interest"),
    start_date: str = typer.Argument(..., help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Argument(..., help="End date (YYYY-MM-DD)"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Run kelp forest carbon analysis from the command line."""
    setup_logging()
    logger.info(
        f"Starting analysis for location ({lat}, {lng}) from {start_date} to {end_date}"
    )

    try:
        from . import (
            apply_mask,
            calculate_indices_from_dataset,
            fetch_sentinel_tiles,
            predict_biomass,
        )

        # Fetch satellite data
        logger.info("Fetching satellite data...")
        satellite_data = fetch_sentinel_tiles(
            lat=lat, lng=lng, start_date=start_date, end_date=end_date
        )

        # Calculate indices
        logger.info("Calculating spectral indices...")
        indices = calculate_indices_from_dataset(satellite_data["data"])

        # Merge data
        combined_data = satellite_data["data"].copy()
        for var in indices.data_vars:
            combined_data[var] = indices[var]

        # Apply masking
        logger.info("Applying masks...")
        masked_data = apply_mask(combined_data)

        # Predict biomass
        logger.info("Predicting biomass...")
        biomass_result = predict_biomass(masked_data)

        # Display results
        biomass = biomass_result["biomass_kg_per_hectare"]
        confidence = biomass_result["prediction_confidence"]
        carbon = (
            biomass * KelpAnalysis.CARBON_CONTENT_FACTOR / KelpAnalysis.HECTARE_TO_M2
        )  # Convert to kg C/m²

        print("\n🌊 Kelp Forest Carbon Analysis Results")
        print(f"{'=' * 50}")
        print(f"Location: {lat:.4f}, {lng:.4f}")
        print(f"Date Range: {start_date} to {end_date}")
        print(f"Biomass: {biomass:.1f} kg/ha")
        print(f"Carbon: {carbon:.4f} kg C/m² ({carbon * 10000:.1f} kg C/ha)")
        print(f"Confidence: {confidence:.2f}")
        print(f"Model: {biomass_result['model_type']}")

        if output:
            import json

            output_data = {
                "location": {"lat": lat, "lng": lng},
                "date_range": {"start": start_date, "end": end_date},
                "biomass_kg_per_hectare": biomass,
                "carbon_kg_per_m2": carbon,
                "confidence": confidence,
                "model_type": biomass_result["model_type"],
                "top_features": biomass_result.get("top_features", []),
            }

            with open(output, "w") as f:
                json.dump(output_data, f, indent=2)

            logger.info(f"Results saved to {output}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def config(
    env: str | None = typer.Option(
        "development", "--env", "-e", help="Environment to show config for"
    ),
):
    """Show current configuration."""
    from dataclasses import asdict

    settings = get_settings()
    config_dict = asdict(settings)

    print(f"🔧 Kelpie Carbon v1 Configuration ({env})")
    print(f"{'=' * 50}")
    print(json.dumps(config_dict, indent=2, default=str))


@app.command()
def test(
    pattern: str | None = typer.Option(
        "test_*.py", "--pattern", "-p", help="Test file pattern"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run the test suite."""
    import subprocess

    setup_logging()
    logger.info("Running test suite...")

    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if pattern and pattern != "test_*.py":
        cmd.extend(["-k", pattern])

    cmd.append("tests/")

    try:
        subprocess.run(cmd, check=True)
        logger.info("Tests completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed with exit code {e.returncode}")
        raise typer.Exit(e.returncode) from e


# Import validation CLI for proxy command
try:
    from kelpie_carbon.validation.cli import app as _val_app

    validation_available = True
except ImportError:
    logger.warning("Validation CLI not available")
    validation_available = False


# Define command-line arguments for validation as module-level variables
VALIDATION_ARGS = typer.Argument(..., help="Arguments for validation CLI")


@app.command()
def validation(
    args: list[str] = VALIDATION_ARGS,
):
    """Run validation commands to test system accuracy against ground truth.

    This command forwards arguments to the validation CLI.
    """
    if not validation_available:
        logger.error("Validation CLI is not available")
        raise typer.Exit(1)

    _val_app(standalone_mode=False, args=args)


@app.command()
def version():
    """Show version information."""
    from . import __version__

    print(f"Kelpie Carbon v1 version {__version__}")
    print("Kelp Forest Carbon Sequestration Assessment")


if __name__ == "__main__":
    app()
