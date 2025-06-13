"""Comprehensive metrics framework for validation.

This module provides standardized metrics for evaluating model performance
across different validation scenarios and data types.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


class ValidationResult(BaseModel):
    """Standardized validation result container for all validation metrics.
    T2-001: ValidationResult implementation with comprehensive metric tracking.
    """

    # Metadata
    campaign_id: str = Field(
        ..., description="Unique identifier for validation campaign"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Validation timestamp"
    )
    test_site: str = Field(..., description="Test site name/identifier")
    model_name: str = Field(..., description="Model being validated")
    dataset_info: dict[str, Any] = Field(
        default_factory=dict, description="Dataset metadata"
    )

    # Core metrics
    accuracy: float | None = Field(
        None, ge=0.0, le=1.0, description="Classification accuracy"
    )
    precision: float | None = Field(None, ge=0.0, le=1.0, description="Precision score")
    recall: float | None = Field(None, ge=0.0, le=1.0, description="Recall score")
    f1_score: float | None = Field(None, ge=0.0, le=1.0, description="F1 score")

    # Regression metrics (T2-001 requirement)
    mae: float | None = Field(None, ge=0.0, description="Mean Absolute Error")
    rmse: float | None = Field(None, ge=0.0, description="Root Mean Square Error")
    r2: float | None = Field(None, description="R-squared coefficient")

    # Segmentation metrics (T2-001 requirement)
    iou: float | None = Field(
        None, ge=0.0, le=1.0, description="Intersection over Union"
    )
    dice_coefficient: float | None = Field(
        None, ge=0.0, le=1.0, description="Dice coefficient"
    )

    # Additional metrics
    auc_pr: float | None = Field(
        None, ge=0.0, le=1.0, description="Area Under Precision-Recall Curve"
    )
    auc_roc: float | None = Field(
        None, ge=0.0, le=1.0, description="Area Under ROC Curve"
    )

    # Performance metrics
    inference_time_ms: float | None = Field(
        None, ge=0.0, description="Inference time in milliseconds"
    )
    memory_usage_mb: float | None = Field(
        None, ge=0.0, description="Memory usage in MB"
    )

    # Validation status
    passed_validation: bool = Field(
        False, description="Whether validation passed all thresholds"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="List of validation errors"
    )

    # Raw results for detailed analysis
    raw_metrics: dict[str, Any] = Field(
        default_factory=dict, description="Raw metric calculations"
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda dt: dt.isoformat()}


class MetricHelpers:
    """T2-001: Standardized metric helper functions for MAE, RMSE, R², IoU, Dice.
    Provides consistent implementations across the validation framework.
    """

    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error (MAE).

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            MAE value

        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error (RMSE).

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            RMSE value

        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared coefficient.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            R² value

        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def calculate_iou(
        y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
    ) -> float:
        """Calculate Intersection over Union (IoU) for binary masks.

        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted probabilities or binary mask
            threshold: Threshold for converting probabilities to binary

        Returns:
            IoU value

        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0

        # Convert to binary if needed
        if y_pred.dtype != bool and np.max(y_pred) <= 1.0:
            y_pred_binary = y_pred >= threshold
        else:
            y_pred_binary = y_pred.astype(bool)

        y_true_binary = y_true.astype(bool)

        # Calculate intersection and union
        intersection = np.logical_and(y_true_binary, y_pred_binary).sum()
        union = np.logical_or(y_true_binary, y_pred_binary).sum()

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return float(intersection / union)

    @staticmethod
    def calculate_dice_coefficient(
        y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
    ) -> float:
        """Calculate Dice coefficient for binary masks.

        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted probabilities or binary mask
            threshold: Threshold for converting probabilities to binary

        Returns:
            Dice coefficient value

        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0

        # Convert to binary if needed
        if y_pred.dtype != bool and np.max(y_pred) <= 1.0:
            y_pred_binary = y_pred >= threshold
        else:
            y_pred_binary = y_pred.astype(bool)

        y_true_binary = y_true.astype(bool)

        # Calculate Dice coefficient
        intersection = np.logical_and(y_true_binary, y_pred_binary).sum()
        total = y_true_binary.sum() + y_pred_binary.sum()

        if total == 0:
            return 1.0 if intersection == 0 else 0.0

        return float(2.0 * intersection / total)


class ValidationMetrics:
    """Enhanced validation metrics for kelp detection accuracy with standardized helpers."""

    def __init__(self):
        """Initialize validation metrics with metric helpers."""
        self.metric_helpers = MetricHelpers()

    def create_validation_result(
        self, campaign_id: str, test_site: str, model_name: str, **kwargs: Any
    ) -> ValidationResult:
        """Create ValidationResult instance with standardized metrics.
        Enhanced for T2-001: Core framework for validation result creation.

        Args:
            campaign_id: Unique identifier for validation campaign
            test_site: Test site name/identifier
            model_name: Model being validated
            **kwargs: Additional validation metrics

        Returns:
            ValidationResult instance

        """
        return ValidationResult(
            campaign_id=campaign_id,
            test_site=test_site,
            model_name=model_name,
            **kwargs,
        )

    def calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
        is_segmentation: bool = False,
    ) -> dict[str, float]:
        """Calculate comprehensive metrics using standardized helpers.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            y_prob: Predicted probabilities (optional)
            is_segmentation: Whether this is a segmentation task

        Returns:
            Dictionary of calculated metrics

        """
        metrics = {}

        # Classification metrics
        if y_true.dtype == bool or np.all(np.isin(y_true, [0, 1])):
            y_true_binary = y_true.astype(bool)
            y_pred_binary = y_pred.astype(bool)

            metrics.update(
                {
                    "accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
                    "precision": float(
                        precision_score(y_true_binary, y_pred_binary, zero_division=0)
                    ),
                    "recall": float(
                        recall_score(y_true_binary, y_pred_binary, zero_division=0)
                    ),
                    "f1_score": float(
                        f1_score(y_true_binary, y_pred_binary, zero_division=0)
                    ),
                }
            )

            # Segmentation metrics
            if is_segmentation:
                metrics.update(
                    {
                        "iou": self.metric_helpers.calculate_iou(y_true, y_pred),
                        "dice_coefficient": self.metric_helpers.calculate_dice_coefficient(
                            y_true, y_pred
                        ),
                    }
                )

        # Regression metrics (for continuous values)
        else:
            metrics.update(
                {
                    "mae": self.metric_helpers.calculate_mae(y_true, y_pred),
                    "rmse": self.metric_helpers.calculate_rmse(y_true, y_pred),
                    "r2": self.metric_helpers.calculate_r2(y_true, y_pred),
                }
            )

        return metrics

    def calculate_detection_metrics(
        self,
        ground_truth: list[bool],
        ndre_predictions: list[bool],
        ndvi_predictions: list[bool],
    ) -> dict[str, Any]:
        """Calculate comprehensive detection accuracy metrics."""
        gt = np.array(ground_truth)
        ndre_pred = np.array(ndre_predictions)
        ndvi_pred = np.array(ndvi_predictions)

        # NDRE metrics using standardized helpers
        ndre_metrics = self.calculate_comprehensive_metrics(
            gt, ndre_pred, is_segmentation=True
        )

        # NDVI metrics using standardized helpers
        ndvi_metrics = self.calculate_comprehensive_metrics(
            gt, ndvi_pred, is_segmentation=True
        )

        # Improvements
        improvements = {
            "accuracy_improvement": ndre_metrics["accuracy"] - ndvi_metrics["accuracy"],
            "precision_improvement": ndre_metrics["precision"]
            - ndvi_metrics["precision"],
            "recall_improvement": ndre_metrics["recall"] - ndvi_metrics["recall"],
            "f1_improvement": ndre_metrics["f1_score"] - ndvi_metrics["f1_score"],
        }

        # Area detection
        ndre_area: int = int(np.sum(ndre_pred))
        ndvi_area: int = int(np.sum(ndvi_pred))
        actual_area: int = int(np.sum(gt))

        area_metrics = {
            "true_kelp_area": actual_area,
            "ndre_detected_area": ndre_area,
            "ndvi_detected_area": ndvi_area,
            "area_improvement_pct": ((ndre_area - ndvi_area) / max(ndvi_area, 1)) * 100,
        }

        return {
            "ndre_metrics": ndre_metrics,
            "ndvi_metrics": ndvi_metrics,
            "improvements": improvements,
            "area_metrics": area_metrics,
            "sample_size": len(ground_truth),
        }

    def calculate_skema_score(self, validation_results: dict[str, Any]) -> float:
        """Calculate SKEMA validation score based on research targets."""
        ndre_metrics = validation_results["ndre_metrics"]
        improvements = validation_results["improvements"]
        area_metrics = validation_results["area_metrics"]

        # Target scoring (0-1 scale)
        accuracy_score = min(ndre_metrics["accuracy"] / 0.80, 1.0)  # Target: 80%
        area_score = min(
            abs(area_metrics["area_improvement_pct"]) / 18.0, 1.0
        )  # Target: +18%
        recall_score = min(
            improvements["recall_improvement"] / 0.15, 1.0
        )  # Target: +15%

        # Combined score
        skema_score = 0.4 * accuracy_score + 0.3 * area_score + 0.3 * recall_score
        return max(0.0, min(1.0, skema_score))

    def generate_report(
        self, campaign_id: str, validation_results: dict[str, Any]
    ) -> ValidationResult:
        """Generate standardized validation report using ValidationResult.
        Enhanced for T2-001 compliance.
        """
        skema_score = self.calculate_skema_score(validation_results)
        ndre_metrics = validation_results["ndre_metrics"]

        # Create comprehensive ValidationResult
        validation_result = ValidationResult(
            campaign_id=campaign_id,
            test_site="SKEMA_NDRE_vs_NDVI",
            model_name="SKEMA_NDRE_Enhanced",
            # Core metrics from NDRE results
            accuracy=ndre_metrics.get("accuracy"),
            precision=ndre_metrics.get("precision"),
            recall=ndre_metrics.get("recall"),
            f1_score=ndre_metrics.get("f1_score"),
            iou=ndre_metrics.get("iou"),
            dice_coefficient=ndre_metrics.get("dice_coefficient"),
            # Validation assessment
            passed_validation=skema_score >= 0.6,
            raw_metrics={
                "skema_score": skema_score,
                "validation_results": validation_results,
                "assessment": (
                    "EXCELLENT"
                    if skema_score >= 0.8
                    else (
                        "GOOD"
                        if skema_score >= 0.6
                        else "MODERATE"
                        if skema_score >= 0.4
                        else "POOR"
                    )
                ),
                "meets_targets": {
                    "accuracy_80pct": ndre_metrics["accuracy"] >= 0.80,
                    "area_improvement_18pct": validation_results["area_metrics"][
                        "area_improvement_pct"
                    ]
                    >= 18.0,
                    "submerged_detection": validation_results["improvements"][
                        "recall_improvement"
                    ]
                    > 0.10,
                },
            },
        )

        return validation_result
