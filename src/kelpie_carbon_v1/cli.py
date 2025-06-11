"""CLI module for Kelpie-Carbon v1."""

import socket
from typing import Optional

import typer
import uvicorn

from .config import get_settings
from .constants import Network, KelpAnalysis
from .logging_config import setup_logging, get_logger

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
    host: Optional[str] = typer.Option(None, "--host", "-h", help="Host to bind to"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Port to bind to"),
    reload: Optional[bool] = typer.Option(None, "--reload", help="Enable auto-reload"),
    env: Optional[str] = typer.Option(
        None, "--env", "-e", help="Environment (development/production)"
    ),
    auto_port: bool = typer.Option(
        False,
        "--auto-port",
        help="Automatically find available port if specified port is busy",
    ),
):
    """Start the Kelpie Carbon v1 web server."""
    # Setup logging first
    setup_logging()

    # Load configuration
    settings = get_settings()

    # Determine port to use
    target_port = port or settings.port

    if auto_port:
        try:
            target_port = _find_available_port(target_port)
            if target_port != (port or settings.port):
                logger.info(
                    f"Port {port or settings.port} was busy, using port {target_port} instead"
                )
        except RuntimeError as e:
            logger.error(f"Failed to find available port: {e}")
            raise typer.Exit(1)

    # Override with CLI arguments if provided
    server_config = {
        "app": "kelpie_carbon_v1.api.main:app",
        "host": host or settings.host,
        "port": target_port,
        "reload": reload if reload is not None else settings.reload,
        "log_level": settings.log_level.lower(),
        "access_log": True,
    }

    # Configure selective file watching for reload
    if server_config["reload"]:
        server_config.update(
            {
                "reload_dirs": ["src/kelpie_carbon_v1"],
                "reload_includes": ["*.py"],
                "reload_excludes": [
                    "*.pyc",
                    "__pycache__/*",
                    "*.log",
                    "*.tmp",
                    "tests/*",
                    "docs/*",
                    "*.md",
                    "*.yml",
                    "*.yaml",
                    ".git/*",
                    ".pytest_cache/*",
                    "*.egg-info/*",
                ],
                "reload_delay": 2.0,  # Delay between file checks (seconds)
            }
        )

    if not settings.reload:
        server_config["workers"] = settings.workers

    logger.info(
        f"Starting Kelpie Carbon v1 server on {server_config['host']}:{server_config['port']}"
    )
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

    try:
        # Extract values and pass them directly to avoid dict typing issues
        from typing import Any, Dict, cast
        
        host_str = str(server_config["host"])
        port_int = int(server_config["port"]) if isinstance(server_config["port"], (int, str)) else 8000
        reload_bool = bool(server_config["reload"])
        log_level_str = str(server_config["log_level"])
        access_log_bool = bool(server_config["access_log"])
        
        # Prepare conditional arguments
        additional_args: Dict[str, Any] = {}
        
        # Add reload-specific args if reload is enabled  
        if reload_bool and "reload_dirs" in server_config:
            additional_args.update({
                "reload_dirs": server_config["reload_dirs"],
                "reload_includes": server_config["reload_includes"],
                "reload_excludes": server_config["reload_excludes"],
                "reload_delay": float(cast(float, server_config["reload_delay"])) if "reload_delay" in server_config and server_config["reload_delay"] is not None else 2.0,
            })
        
        # Add workers if not in reload mode
        if "workers" in server_config and not reload_bool and server_config["workers"] is not None:
            additional_args["workers"] = int(cast(int, server_config["workers"]))

        uvicorn.run(
            "kelpie_carbon_v1.api.main:app",
            host=host_str,
            port=port_int,
            reload=reload_bool,
            log_level=log_level_str,
            access_log=access_log_bool,
            **additional_args
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
        raise typer.Exit(1)


@app.command()
def analyze(
    lat: float = typer.Argument(..., help="Latitude of the area of interest"),
    lng: float = typer.Argument(..., help="Longitude of the area of interest"),
    start_date: str = typer.Argument(..., help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Argument(..., help="End date (YYYY-MM-DD)"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
):
    """Run kelp forest carbon analysis from the command line."""
    setup_logging()
    logger.info(
        f"Starting analysis for location ({lat}, {lng}) from {start_date} to {end_date}"
    )

    try:
        from .core import (
            fetch_sentinel_tiles,
            calculate_indices_from_dataset,
            apply_mask,
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

        print(f"\n🌊 Kelp Forest Carbon Analysis Results")
        print(f"{'='*50}")
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
        raise typer.Exit(1)


@app.command()
def config(
    env: Optional[str] = typer.Option(
        "development", "--env", "-e", help="Environment to show config for"
    ),
):
    """Show current configuration."""
    import json
    from dataclasses import asdict

    settings = get_settings()
    config_dict = asdict(settings)

    print(f"🔧 Kelpie Carbon v1 Configuration ({env})")
    print(f"{'='*50}")
    print(json.dumps(config_dict, indent=2, default=str))


@app.command()
def test(
    pattern: Optional[str] = typer.Option(
        "test_*.py", "--pattern", "-p", help="Test file pattern"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run the test suite."""
    import subprocess
    import sys

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
        raise typer.Exit(e.returncode)


@app.command()
def version():
    """Show version information."""
    from . import __version__

    print(f"Kelpie Carbon v1 version {__version__}")
    print("Kelp Forest Carbon Sequestration Assessment")


if __name__ == "__main__":
    app()
