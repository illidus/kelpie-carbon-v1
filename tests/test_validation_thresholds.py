"""
Test config-driven validation thresholds (T2-002).
"""

import numpy as np

from kelpie_carbon.validation.cli import validate_against_thresholds


class TestValidationThresholds:
    """Test config-driven validation thresholds."""

    def test_validate_against_thresholds_pass(self):
        """Test validation passes when metrics meet thresholds."""
        metrics = {
            "accuracy": 0.85,
            "mae": 0.08,
            "rmse": 0.12,
            "r2": 0.90,
            "iou": 0.80,
            "dice_coefficient": 0.85,
        }

        config = {
            "thresholds": {
                "accuracy": {"min": 0.75},
                "mae": {"max": 0.15},
                "rmse": {"max": 0.20},
                "r2": {"min": 0.70},
                "iou": {"min": 0.60},
                "dice_coefficient": {"min": 0.65},
            }
        }

        passed, errors = validate_against_thresholds(metrics, config)

        assert passed is True
        assert len(errors) == 0

    def test_validate_against_thresholds_fail(self):
        """Test validation fails when metrics don't meet thresholds."""
        metrics = {
            "accuracy": 0.65,  # Below min 0.75
            "mae": 0.25,  # Above max 0.15
            "r2": 0.60,  # Below min 0.70
        }

        config = {
            "thresholds": {
                "accuracy": {"min": 0.75},
                "mae": {"max": 0.15},
                "r2": {"min": 0.70},
            }
        }

        passed, errors = validate_against_thresholds(metrics, config)

        assert passed is False
        assert len(errors) == 3
        assert "ACCURACY 0.6500 below minimum threshold 0.75" in errors
        assert "MAE 0.2500 exceeds maximum threshold 0.15" in errors
        assert "R2 0.6000 below minimum threshold 0.7" in errors

    def test_validate_against_thresholds_empty_config(self):
        """Test validation passes when no thresholds are configured."""
        metrics = {"accuracy": 0.50, "mae": 0.30}
        config = {}

        passed, errors = validate_against_thresholds(metrics, config)

        assert passed is True
        assert len(errors) == 0

    def test_validate_against_thresholds_none_values(self):
        """Test validation handles None metric values gracefully."""
        metrics = {
            "accuracy": None,
            "mae": 0.10,
            "r2": None,
        }

        config = {
            "thresholds": {
                "accuracy": {"min": 0.75},
                "mae": {"max": 0.15},
                "r2": {"min": 0.70},
            }
        }

        passed, errors = validate_against_thresholds(metrics, config)

        # Should pass because None values are ignored, and MAE meets threshold
        assert passed is True
        assert len(errors) == 0

    def test_validate_performance_metrics(self):
        """Test validation of performance metrics."""
        metrics = {
            "inference_time_ms": 6000,  # Above max 5000
            "memory_usage_mb": 3000,  # Above max 2048
        }

        config = {
            "thresholds": {
                "inference_time_ms": {"max": 5000},
                "memory_usage_mb": {"max": 2048},
            }
        }

        passed, errors = validate_against_thresholds(metrics, config)

        assert passed is False
        assert len(errors) == 2
        assert "inference_time_ms 6000 exceeds maximum threshold 5000" in errors
        assert "memory_usage_mb 3000 exceeds maximum threshold 2048" in errors

    def test_validate_against_thresholds_with_test_data(self):
        """Test validation with test data."""
        # Create test data for demonstration
        n_samples = 100
        # Note: x and y_pred are created for demonstration but not used in this test
        # In a real scenario, these would be used for model training/evaluation

        metrics = {
            "accuracy": 0.85,
            "mae": 0.08,
            "rmse": 0.12,
            "r2": 0.90,
            "iou": 0.80,
            "dice_coefficient": 0.85,
        }

        config = {
            "thresholds": {
                "accuracy": {"min": 0.75},
                "mae": {"max": 0.15},
                "rmse": {"max": 0.20},
                "r2": {"min": 0.70},
                "iou": {"min": 0.60},
                "dice_coefficient": {"min": 0.65},
            }
        }

        passed, errors = validate_against_thresholds(metrics, config)

        assert passed is True
        assert len(errors) == 0
