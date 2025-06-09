"""
ValidationMetrics - Task 2.3
Calculates validation metrics for SKEMA NDRE vs NDVI comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ValidationMetrics:
    """Validation metrics for kelp detection accuracy."""
    
    def calculate_detection_metrics(self, 
                                  ground_truth: List[bool],
                                  ndre_predictions: List[bool],
                                  ndvi_predictions: List[bool]) -> Dict[str, Any]:
        """Calculate comprehensive detection accuracy metrics."""
        
        gt = np.array(ground_truth)
        ndre_pred = np.array(ndre_predictions)
        ndvi_pred = np.array(ndvi_predictions)
        
        # NDRE metrics
        ndre_metrics = {
            "accuracy": accuracy_score(gt, ndre_pred),
            "precision": precision_score(gt, ndre_pred, zero_division=0),
            "recall": recall_score(gt, ndre_pred, zero_division=0),
            "f1_score": f1_score(gt, ndre_pred, zero_division=0)
        }
        
        # NDVI metrics
        ndvi_metrics = {
            "accuracy": accuracy_score(gt, ndvi_pred),
            "precision": precision_score(gt, ndvi_pred, zero_division=0),
            "recall": recall_score(gt, ndvi_pred, zero_division=0),
            "f1_score": f1_score(gt, ndvi_pred, zero_division=0)
        }
        
        # Improvements
        improvements = {
            "accuracy_improvement": ndre_metrics["accuracy"] - ndvi_metrics["accuracy"],
            "precision_improvement": ndre_metrics["precision"] - ndvi_metrics["precision"],
            "recall_improvement": ndre_metrics["recall"] - ndvi_metrics["recall"],
            "f1_improvement": ndre_metrics["f1_score"] - ndvi_metrics["f1_score"]
        }
        
        # Area detection
        ndre_area = np.sum(ndre_pred)
        ndvi_area = np.sum(ndvi_pred)
        actual_area = np.sum(gt)
        
        area_metrics = {
            "true_kelp_area": actual_area,
            "ndre_detected_area": ndre_area,
            "ndvi_detected_area": ndvi_area,
            "area_improvement_pct": ((ndre_area - ndvi_area) / max(ndvi_area, 1)) * 100
        }
        
        return {
            "ndre_metrics": ndre_metrics,
            "ndvi_metrics": ndvi_metrics,
            "improvements": improvements,
            "area_metrics": area_metrics,
            "sample_size": len(ground_truth)
        }
        
    def calculate_skema_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate SKEMA validation score based on research targets."""
        
        ndre_metrics = validation_results["ndre_metrics"]
        improvements = validation_results["improvements"]
        area_metrics = validation_results["area_metrics"]
        
        # Target scoring (0-1 scale)
        accuracy_score = min(ndre_metrics["accuracy"] / 0.80, 1.0)  # Target: 80%
        area_score = min(abs(area_metrics["area_improvement_pct"]) / 18.0, 1.0)  # Target: +18%
        recall_score = min(improvements["recall_improvement"] / 0.15, 1.0)  # Target: +15%
        
        # Combined score
        skema_score = (0.4 * accuracy_score + 0.3 * area_score + 0.3 * recall_score)
        return max(0.0, min(1.0, skema_score))
        
    def generate_report(self, campaign_id: str, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation report."""
        
        skema_score = self.calculate_skema_score(validation_results)
        
        return {
            "campaign_id": campaign_id,
            "skema_score": skema_score,
            "validation_results": validation_results,
            "assessment": "EXCELLENT" if skema_score >= 0.8 else 
                         "GOOD" if skema_score >= 0.6 else
                         "MODERATE" if skema_score >= 0.4 else "POOR",
            "meets_targets": {
                "accuracy_80pct": validation_results["ndre_metrics"]["accuracy"] >= 0.80,
                "area_improvement_18pct": validation_results["area_metrics"]["area_improvement_pct"] >= 18.0,
                "submerged_detection": validation_results["improvements"]["recall_improvement"] > 0.10
            }
        } 