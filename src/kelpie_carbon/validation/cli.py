"""
Validation CLI for Kelpie-Carbon.

Provides command-line interface for validation tasks including:
- Dataset validation with MAE, RMSE, R² metrics
- Report generation in JSON and Markdown formats
- Integration with unified configuration system

Usage:
    kelpie validate --dataset <path> --out validation/results
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..core.config import load
from ..core.logging_config import get_logger
from .core.metrics import MetricHelpers, ValidationMetrics, ValidationResult

# Initialize CLI app and utilities
app = typer.Typer(
    name="validation",
    help="Validation framework for Kelpie-Carbon",
    add_completion=False,
)
console = Console()
logger = get_logger(__name__)

# Define module-level Typer Option variables
DATASET_OPTION = typer.Option(
    ...,
    "--dataset",
    "-d",
    help="Path to validation dataset (JSON format)",
    exists=True,
    file_okay=True,
    dir_okay=False,
)

OUTPUT_OPTION = typer.Option(
    "validation/results",
    "--out",
    "-o",
    help="Output directory for validation results",
)


def load_validation_config() -> dict:
    """
    Load validation configuration from unified config.

    Returns:
        Validation configuration dictionary
    """
    try:
        config = load()
        return config.get("validation", {})
    except Exception as e:
        logger.error(f"Failed to load validation config: {e}")
        return {}


def load_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load validation dataset from file.

    Supports JSON format with 'y_true' and 'y_pred' arrays.

    Args:
        dataset_path: Path to dataset file

    Returns:
        Tuple of (y_true, y_pred) numpy arrays

    Raises:
        typer.Exit: If dataset cannot be loaded
    """
    try:
        if not dataset_path.exists():
            console.print(f"[red]Error:[/red] Dataset file not found: {dataset_path}")
            raise typer.Exit(1)

        if dataset_path.suffix.lower() == ".json":
            with open(dataset_path) as f:
                data = json.load(f)

            if "y_true" not in data or "y_pred" not in data:
                console.print(
                    "[red]Error:[/red] Dataset must contain 'y_true' and 'y_pred' arrays"
                )
                raise typer.Exit(1)

            y_true = np.array(data["y_true"])
            y_pred = np.array(data["y_pred"])

            logger.info(f"Loaded dataset: {len(y_true)} samples")
            return y_true, y_pred

        else:
            console.print(
                f"[red]Error:[/red] Unsupported dataset format: {dataset_path.suffix}"
            )
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error loading dataset:[/red] {e}")
        raise typer.Exit(1) from e


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute validation metrics using MetricHelpers.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary of computed metrics
    """
    helpers = MetricHelpers()

    metrics = {
        "mae": helpers.calculate_mae(y_true, y_pred),
        "rmse": helpers.calculate_rmse(y_true, y_pred),
        "r2": helpers.calculate_r2(y_true, y_pred),
    }

    # Add segmentation metrics if data appears to be binary
    if np.all(np.isin(y_true, [0, 1])) and np.all((y_pred >= 0) & (y_pred <= 1)):
        metrics["iou"] = helpers.calculate_iou(y_true, y_pred)
        metrics["dice_coefficient"] = helpers.calculate_dice_coefficient(y_true, y_pred)

    return metrics


def generate_json_report(
    validation_result: ValidationResult, output_path: Path
) -> None:
    """
    Generate JSON validation report.

    Args:
        validation_result: ValidationResult object
        output_path: Output directory path
    """
    json_path = output_path / "validation_report.json"

    try:
        with open(json_path, "w") as f:
            json.dump(validation_result.model_dump(), f, indent=2, default=str)

        console.print(f"[green]✓[/green] JSON report saved: {json_path}")
        logger.info(f"JSON report generated: {json_path}")

    except Exception as e:
        console.print(f"[red]Error generating JSON report:[/red] {e}")
        logger.error(f"Failed to generate JSON report: {e}")


def generate_markdown_report(
    validation_result: ValidationResult, metrics: dict, output_path: Path
) -> None:
    """
    Generate Markdown validation report.

    Args:
        validation_result: ValidationResult object
        metrics: Computed metrics dictionary
        output_path: Output directory path
    """
    md_path = output_path / "validation_report.md"

    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Validation Report\n\n")
            f.write(f"**Campaign ID:** {validation_result.campaign_id}\n")
            f.write(f"**Timestamp:** {validation_result.timestamp}\n")
            f.write(f"**Test Site:** {validation_result.test_site}\n")
            f.write(f"**Model:** {validation_result.model_name}\n\n")

            f.write("## Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")

            for metric_name, value in metrics.items():
                if value is not None:
                    f.write(f"| {metric_name.upper()} | {value:.4f} |\n")

            f.write("\n## Validation Status\n\n")
            status = "✅ PASSED" if validation_result.passed_validation else "❌ FAILED"
            f.write(f"**Status:** {status}\n\n")

            if validation_result.validation_errors:
                f.write("### Errors\n\n")
                for error in validation_result.validation_errors:
                    f.write(f"- {error}\n")
                f.write("\n")

            f.write("## Dataset Information\n\n")
            for key, value in validation_result.dataset_info.items():
                f.write(f"- **{key}:** {value}\n")

        console.print(f"[green]✓[/green] Markdown report saved: {md_path}")
        logger.info(f"Markdown report generated: {md_path}")

    except Exception as e:
        console.print(f"[red]Error generating Markdown report:[/red] {e}")
        logger.error(f"Failed to generate Markdown report: {e}")


@app.command()
def validate(
    dataset: Path = DATASET_OPTION,
    output: Path = OUTPUT_OPTION,
    campaign_id: str | None = typer.Option(
        None,
        "--campaign-id",
        "-c",
        help="Validation campaign identifier (auto-generated if not provided)",
    ),
    test_site: str = typer.Option(
        "unknown",
        "--test-site",
        "-s",
        help="Test site name/identifier",
    ),
    model_name: str = typer.Option(
        "kelpie-carbon",
        "--model",
        "-m",
        help="Model name being validated",
    ),
) -> None:
    """
    Validate satellite-based kelp detection against ground truth.

    Processes validation dataset and generates comprehensive metrics report.
    """
    console.print("[bold blue]🌊 Kelpie-Carbon Validation[/bold blue]")

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    # Generate campaign ID if not provided
    if not campaign_id:
        campaign_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Load validation configuration
    config = load_validation_config()
    console.print(f"[dim]Loaded validation config with {len(config)} sections[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load dataset
        task1 = progress.add_task("Loading dataset...", total=None)
        y_true, y_pred = load_dataset(dataset)
        progress.update(task1, description=f"✓ Loaded {len(y_true)} samples")

        # Compute metrics
        task2 = progress.add_task("Computing metrics...", total=None)
        metrics = compute_metrics(y_true, y_pred)
        progress.update(task2, description="✓ Computed validation metrics")

        # Create validation result
        task3 = progress.add_task("Creating validation result...", total=None)
        validation_metrics = ValidationMetrics()

        validation_result = validation_metrics.create_validation_result(
            campaign_id=campaign_id,
            test_site=test_site,
            model_name=model_name,
            mae=metrics.get("mae"),
            rmse=metrics.get("rmse"),
            r2=metrics.get("r2"),
            iou=metrics.get("iou"),
            dice_coefficient=metrics.get("dice_coefficient"),
            dataset_info={
                "dataset_path": str(dataset),
                "sample_count": len(y_true),
                "data_type": "regression" if "iou" not in metrics else "segmentation",
            },
        )

        # Apply validation thresholds from config
        thresholds = config.get("thresholds", {})
        min_accuracy = thresholds.get("min_accuracy", 0.75)

        # Simple validation check (can be enhanced)
        if validation_result.r2 and validation_result.r2 >= min_accuracy:
            validation_result.passed_validation = True
        else:
            validation_result.validation_errors.append(
                f"R² score {validation_result.r2:.4f} below minimum threshold {min_accuracy}"
            )

        progress.update(task3, description="✓ Created validation result")

        # Generate reports
        task4 = progress.add_task("Generating reports...", total=None)
        generate_json_report(validation_result, output)
        generate_markdown_report(validation_result, metrics, output)
        progress.update(task4, description="✓ Generated reports")

    # Display results table
    table = Table(title="Validation Results")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for metric_name, value in metrics.items():
        if value is not None:
            table.add_row(metric_name.upper(), f"{value:.4f}")

    console.print(table)

    # Final status
    if validation_result.passed_validation:
        console.print(
            f"[green]✅ Validation PASSED[/green] - Results saved to {output}"
        )
    else:
        console.print("[red]❌ Validation FAILED[/red] - Check reports for details")
        for error in validation_result.validation_errors:
            console.print(f"[red]  • {error}[/red]")


@app.command()
def config() -> None:
    """Show current validation configuration."""
    console.print("[bold blue]🔧 Validation Configuration[/bold blue]")

    config = load_validation_config()

    if not config:
        console.print("[red]No validation configuration found[/red]")
        return

    # Display test sites
    if "test_sites" in config:
        table = Table(title="Test Sites")
        table.add_column("Name", style="cyan")
        table.add_column("Species", style="green")
        table.add_column("Coordinates", style="yellow")

        for site in config["test_sites"]:
            coords = site.get("coordinates", {})
            coord_str = f"{coords.get('lat', 'N/A')}, {coords.get('lon', 'N/A')}"
            table.add_row(
                site.get("name", "Unknown"), site.get("species", "Unknown"), coord_str
            )

        console.print(table)

    # Display thresholds
    if "thresholds" in config:
        console.print("\n[bold]Validation Thresholds:[/bold]")
        for key, value in config["thresholds"].items():
            console.print(f"  • {key}: {value}")


if __name__ == "__main__":
    app()
