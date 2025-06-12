"""
Tests for validation CLI functionality.

Tests cover:
- Dataset loading and validation
- Metric computation using MetricHelpers
- JSON and Markdown report generation
- Configuration loading
- CLI command execution
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from kelpie_carbon.validation.cli import (
    app,
    compute_metrics,
    generate_json_report,
    generate_markdown_report,
    load_dataset,
    load_validation_config,
)
from kelpie_carbon.validation.core.metrics import ValidationResult


class TestValidationCLI:
    """Test suite for validation CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        
        # Sample data for testing
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        
        # Binary data for segmentation testing
        self.y_true_binary = np.array([0, 1, 1, 0, 1])
        self.y_pred_binary = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

    def test_load_validation_config_success(self):
        """Test successful loading of validation configuration."""
        mock_config = {
            "validation": {
                "thresholds": {"min_accuracy": 0.75},
                "test_sites": [{"name": "Test Site"}]
            }
        }
        
        with patch("kelpie_carbon.validation.cli.load", return_value=mock_config):
            config = load_validation_config()
            
        assert config == mock_config["validation"]
        assert config["thresholds"]["min_accuracy"] == 0.75

    def test_load_validation_config_failure(self):
        """Test handling of configuration loading failure."""
        with patch("kelpie_carbon.validation.cli.load", side_effect=Exception("Config error")):
            config = load_validation_config()
            
        assert config == {}

    def test_load_dataset_json_success(self):
        """Test successful loading of JSON dataset."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'y_true': self.y_true.tolist(),
                'y_pred': self.y_pred.tolist(),
            }, f)
            temp_path = Path(f.name)
        
        try:
            y_true, y_pred = load_dataset(temp_path)
            
            np.testing.assert_array_almost_equal(y_true, self.y_true)
            np.testing.assert_array_almost_equal(y_pred, self.y_pred)
        finally:
            temp_path.unlink()

    def test_load_dataset_missing_file(self):
        """Test handling of missing dataset file."""
        from click.exceptions import Exit
        nonexistent_path = Path("nonexistent_file.json")
        
        with pytest.raises(Exit):
            load_dataset(nonexistent_path)

    def test_load_dataset_invalid_format(self):
        """Test handling of invalid dataset format."""
        from click.exceptions import Exit
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid content")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(Exit):
                load_dataset(temp_path)
        finally:
            temp_path.unlink()

    def test_load_dataset_missing_keys(self):
        """Test handling of dataset with missing required keys."""
        from click.exceptions import Exit
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'y_true': self.y_true.tolist()}, f)  # Missing y_pred
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(Exit):
                load_dataset(temp_path)
        finally:
            temp_path.unlink()

    def test_compute_metrics_regression(self):
        """Test metric computation for regression data."""
        metrics = compute_metrics(self.y_true, self.y_pred)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0
        assert 0 <= metrics['r2'] <= 1

    def test_compute_metrics_segmentation(self):
        """Test metric computation for segmentation data."""
        metrics = compute_metrics(self.y_true_binary, self.y_pred_binary)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'iou' in metrics
        assert 'dice_coefficient' in metrics
        
        assert 0 <= metrics['iou'] <= 1
        assert 0 <= metrics['dice_coefficient'] <= 1

    def test_generate_json_report(self):
        """Test JSON report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Create mock validation result
            validation_result = ValidationResult(
                campaign_id="test_campaign",
                test_site="test_site", 
                model_name="test_model",
                mae=0.1,
                rmse=0.2,
                r2=0.9,
            )
            
            generate_json_report(validation_result, output_path)
            
            json_file = output_path / "validation_report.json"
            assert json_file.exists()
            
            with open(json_file) as f:
                data = json.load(f)
            
            assert data['campaign_id'] == "test_campaign"
            assert data['mae'] == 0.1
            assert data['rmse'] == 0.2
            assert data['r2'] == 0.9

    def test_generate_markdown_report(self):
        """Test Markdown report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Create mock validation result
            validation_result = ValidationResult(
                campaign_id="test_campaign",
                test_site="test_site",
                model_name="test_model", 
                mae=0.1,
                rmse=0.2,
                r2=0.9,
                passed_validation=True,
            )
            
            metrics = {'mae': 0.1, 'rmse': 0.2, 'r2': 0.9}
            
            generate_markdown_report(validation_result, metrics, output_path)
            
            md_file = output_path / "validation_report.md"
            assert md_file.exists()
            
            content = md_file.read_text(encoding='utf-8')
            assert "# Validation Report" in content
            assert "test_campaign" in content
            assert "MAE | 0.1000" in content
            assert "PASSED" in content  # Check for text without emoji

    def test_cli_validate_command_missing_dataset(self):
        """Test validate command with missing dataset file."""
        result = self.runner.invoke(app, [
            "validate", 
            "--dataset", "nonexistent.json",
            "--out", "test_output"
        ])
        
        assert result.exit_code == 2  # Typer exits with 2 for invalid argument

    def test_cli_config_command(self):
        """Test config command."""
        mock_config = {
            "test_sites": [{"name": "Test Site", "species": "Test Species"}],
            "thresholds": {"min_accuracy": 0.75}
        }
        
        with patch("kelpie_carbon.validation.cli.load_validation_config", return_value=mock_config):
            result = self.runner.invoke(app, ["config"])
            
        assert result.exit_code == 0
        assert "Test Sites" in result.stdout
        assert "Test Site" in result.stdout

    def test_cli_config_command_no_config(self):
        """Test config command with no configuration."""
        with patch("kelpie_carbon.validation.cli.load_validation_config", return_value={}):
            result = self.runner.invoke(app, ["config"])
            
        assert result.exit_code == 0
        assert "No validation configuration found" in result.stdout

    def test_integration_validate_command(self):
        """Integration test for complete validate command workflow."""
        # Create temporary dataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'y_true': self.y_true.tolist(),
                'y_pred': self.y_pred.tolist(),
            }, f)
            dataset_path = f.name
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                # Mock the validation config
                mock_config = {"thresholds": {"min_accuracy": 0.75}}
                
                with patch("kelpie_carbon.validation.cli.load_validation_config", return_value=mock_config):
                    result = self.runner.invoke(app, [
                        "validate",
                        "--dataset", dataset_path,
                        "--out", output_dir,
                        "--test-site", "integration_test",
                        "--model", "test_model"
                    ])
                
                # Check command succeeded
                assert result.exit_code == 0
                assert "Validation Results" in result.stdout
                
                # Check files were created
                output_path = Path(output_dir)
                assert (output_path / "validation_report.json").exists()
                assert (output_path / "validation_report.md").exists()
                
                # Verify JSON report content
                with open(output_path / "validation_report.json") as f:
                    report_data = json.load(f)
                
                assert report_data['test_site'] == "integration_test"
                assert report_data['model_name'] == "test_model"
                assert 'mae' in report_data
                assert 'rmse' in report_data
                assert 'r2' in report_data
                
            finally:
                Path(dataset_path).unlink()

    def test_metric_helpers_integration(self):
        """Test integration with MetricHelpers classes."""
        from kelpie_carbon.validation.core.metrics import MetricHelpers
        
        helpers = MetricHelpers()
        
        mae = helpers.calculate_mae(self.y_true, self.y_pred)
        rmse = helpers.calculate_rmse(self.y_true, self.y_pred)
        r2 = helpers.calculate_r2(self.y_true, self.y_pred)
        
        assert mae > 0
        assert rmse > 0
        assert 0 <= r2 <= 1
        
        # Test with binary data
        iou = helpers.calculate_iou(self.y_true_binary, self.y_pred_binary)
        dice = helpers.calculate_dice_coefficient(self.y_true_binary, self.y_pred_binary)
        
        assert 0 <= iou <= 1
        assert 0 <= dice <= 1


class TestValidationCLIEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        empty_true = np.array([])
        empty_pred = np.array([])
        
        metrics = compute_metrics(empty_true, empty_pred)
        
        # Should handle empty arrays gracefully
        assert metrics['mae'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['r2'] == 0.0

    def test_report_generation_error_handling(self):
        """Test error handling in report generation."""
        # Create invalid output path (read-only directory)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "readonly"
            output_path.mkdir()
            output_path.chmod(0o444)  # Read-only
            
            validation_result = ValidationResult(
                campaign_id="test",
                test_site="test",
                model_name="test"
            )
            
            try:
                # Should not raise exception, but should log error
                generate_json_report(validation_result, output_path)
                generate_markdown_report(validation_result, {}, output_path)
            except Exception:
                pytest.fail("Report generation should handle errors gracefully")
            finally:
                output_path.chmod(0o755)  # Restore permissions for cleanup


if __name__ == "__main__":
    pytest.main([__file__]) 