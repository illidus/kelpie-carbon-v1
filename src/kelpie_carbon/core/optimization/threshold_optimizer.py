"""Threshold Optimizer for SKEMA Detection Pipeline.

This module implements adaptive threshold tuning based on real-world validation
results to optimize detection performance for different environmental conditions.

Task A2.7: Optimize detection pipeline - Performance Optimization
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..logging_config import get_logger

logger = get_logger(__name__)


class ThresholdOptimizer:
    """Optimize detection thresholds based on validation results."""

    def __init__(self):
        """Initialize threshold optimizer."""
        self.validation_history: list[dict] = []
        self.optimization_results: dict[str, Any] = {}

    def load_validation_results(self, results_path: str) -> dict[str, Any]:
        """Load validation results from JSON file.

        Args:
            results_path: Path to validation results JSON file

        Returns:
            Validation results dictionary

        """
        try:
            with open(results_path) as f:
                results = json.load(f)

            logger.info(f"Loaded validation results from {results_path}")
            logger.info(
                f"Results contain {results['total_sites']} sites with "
                f"{results['successful_validations']} successful validations"
            )

            self.validation_history.append(results)
            return results

        except Exception as e:
            logger.error(f"Failed to load validation results: {e}")
            raise

    def analyze_detection_rates(self, results: dict[str, Any]) -> dict[str, float]:
        """Analyze detection rates vs expectations.

        Args:
            results: Validation results dictionary

        Returns:
            Analysis metrics

        """
        analysis = {
            "mean_detection_rate": 0.0,
            "mean_expected_rate": 0.0,
            "over_detection_ratio": 0.0,
            "accuracy_score": 0.0,
            "sites_analyzed": len(results["results"]),
        }

        if not results["results"]:
            return analysis

        detection_rates = []
        expected_rates = []

        for site_result in results["results"]:
            detection_rates.append(site_result["actual_detection_rate"])
            expected_rates.append(site_result["expected_detection_rate"])

        analysis["mean_detection_rate"] = np.mean(detection_rates)
        analysis["mean_expected_rate"] = np.mean(expected_rates)
        analysis["over_detection_ratio"] = (
            analysis["mean_detection_rate"] / analysis["mean_expected_rate"]
        )

        # Calculate accuracy score (1.0 = perfect, 0.0 = worst)
        # Penalize both over-detection and under-detection
        detection_errors = [
            abs(actual - expected)
            for actual, expected in zip(detection_rates, expected_rates, strict=False)
        ]
        analysis["accuracy_score"] = max(0.0, 1.0 - np.mean(detection_errors))

        logger.info("Detection Analysis:")
        logger.info(f"  Mean detection rate: {analysis['mean_detection_rate']:.1%}")
        logger.info(f"  Mean expected rate: {analysis['mean_expected_rate']:.1%}")
        logger.info(f"  Over-detection ratio: {analysis['over_detection_ratio']:.1f}x")
        logger.info(f"  Accuracy score: {analysis['accuracy_score']:.3f}")

        return analysis

    def calculate_optimal_thresholds(self, results: dict[str, Any]) -> dict[str, float]:
        """Calculate optimal detection thresholds based on validation results.

        Args:
            results: Validation results dictionary

        Returns:
            Optimized threshold configuration

        """
        analysis = self.analyze_detection_rates(results)
        current_config = results["skema_configuration"]

        # Start with current thresholds
        optimal_thresholds = {
            "ndre_threshold": current_config.get("ndre_threshold", 0.0),
            "min_detection_threshold": results["configuration"].get(
                "min_detection_threshold", 0.01
            ),
            "kelp_fai_threshold": 0.01,  # Default FAI threshold
        }

        # If over-detecting significantly, increase thresholds
        if analysis["over_detection_ratio"] > 2.0:
            # Aggressive threshold increase for severe over-detection
            threshold_multiplier = min(analysis["over_detection_ratio"] / 2.0, 10.0)

            optimal_thresholds["ndre_threshold"] = max(
                0.05, current_config.get("ndre_threshold", 0.0) + 0.1
            )
            optimal_thresholds["min_detection_threshold"] = min(
                0.2,
                results["configuration"].get("min_detection_threshold", 0.01)
                * threshold_multiplier,
            )
            optimal_thresholds["kelp_fai_threshold"] = min(
                0.1, 0.01 * threshold_multiplier
            )

        elif analysis["over_detection_ratio"] > 1.5:
            # Moderate threshold increase
            optimal_thresholds["ndre_threshold"] = max(
                0.02, current_config.get("ndre_threshold", 0.0) + 0.05
            )
            optimal_thresholds["min_detection_threshold"] = min(
                0.1, results["configuration"].get("min_detection_threshold", 0.01) * 3.0
            )
            optimal_thresholds["kelp_fai_threshold"] = min(0.05, 0.01 * 2.0)

        elif analysis["over_detection_ratio"] < 0.5:
            # Under-detecting, decrease thresholds
            optimal_thresholds["ndre_threshold"] = max(
                0.0, current_config.get("ndre_threshold", 0.0) - 0.02
            )
            optimal_thresholds["min_detection_threshold"] = max(
                0.005,
                results["configuration"].get("min_detection_threshold", 0.01) * 0.5,
            )
            optimal_thresholds["kelp_fai_threshold"] = max(0.005, 0.01 * 0.5)

        logger.info("Calculated optimal thresholds:")
        for key, value in optimal_thresholds.items():
            logger.info(f"  {key}: {value:.3f}")

        return optimal_thresholds

    def create_adaptive_config(
        self, site_type: str, environmental_conditions: dict[str, Any]
    ) -> dict[str, Any]:
        """Create adaptive configuration based on site type and environmental conditions.

        Args:
            site_type: Type of site ('kelp_farm', 'open_ocean', 'coastal')
            environmental_conditions: Environmental factors (cloud_cover, turbidity, etc.)

        Returns:
            Adaptive configuration dictionary

        """
        # Base configuration optimized for different site types
        base_configs = {
            "kelp_farm": {
                "ndre_threshold": 0.08,
                "kelp_fai_threshold": 0.03,
                "min_detection_threshold": 0.05,
                "apply_morphology": True,
                "min_kelp_cluster_size": 8,
                "require_water_context": True,
            },
            "open_ocean": {
                "ndre_threshold": 0.12,
                "kelp_fai_threshold": 0.05,
                "min_detection_threshold": 0.08,
                "apply_morphology": True,
                "min_kelp_cluster_size": 12,
                "require_water_context": True,
            },
            "coastal": {
                "ndre_threshold": 0.06,
                "kelp_fai_threshold": 0.02,
                "min_detection_threshold": 0.04,
                "apply_morphology": True,
                "min_kelp_cluster_size": 6,
                "require_water_context": False,
            },
        }

        config = base_configs.get(site_type, base_configs["coastal"]).copy()

        # Adjust for environmental conditions
        cloud_cover = environmental_conditions.get("cloud_cover", 0.0)
        turbidity = environmental_conditions.get("turbidity", "low")

        # Increase thresholds for high cloud cover
        if cloud_cover > 0.3:
            config["ndre_threshold"] *= 1.2
            config["kelp_fai_threshold"] *= 1.3
            config["min_detection_threshold"] *= 1.15

        # Adjust for turbidity
        if turbidity == "high":
            config["ndre_threshold"] *= 0.9  # Slightly lower for turbid water
            config["kelp_fai_threshold"] *= 1.4  # Higher FAI threshold
            config["min_kelp_cluster_size"] += 3  # Larger clusters required
        elif turbidity == "low":
            config["ndre_threshold"] *= 1.1  # Slightly higher for clear water
            config["kelp_fai_threshold"] *= 0.8  # Lower FAI threshold

        logger.info(f"Created adaptive config for {site_type} site:")
        logger.info(
            f"  Environmental conditions: cloud_cover={cloud_cover:.1%}, turbidity={turbidity}"
        )

        return config

    def optimize_for_real_time(
        self, target_processing_time: float = 15.0
    ) -> dict[str, Any]:
        """Optimize configuration for real-time processing.

        Args:
            target_processing_time: Target processing time in seconds

        Returns:
            Real-time optimized configuration

        """
        # Fast processing configuration
        fast_config = {
            "apply_waf": True,  # Keep WAF but use fast mode
            "waf_fast_mode": True,
            "combine_with_ndre": True,
            "detection_combination": "intersection",  # Faster than union
            "apply_morphology": False,  # Skip morphology for speed
            "min_kelp_cluster_size": 3,  # Smaller minimum size
            "ndre_threshold": 0.1,  # Higher threshold for fewer false positives
            "kelp_fai_threshold": 0.04,
            "min_detection_threshold": 0.06,
            "require_water_context": False,  # Skip for speed
            "max_processing_resolution": 20,  # Lower resolution for speed
        }

        logger.info(
            f"Created real-time optimized config (target: {target_processing_time}s)"
        )
        return fast_config

    def save_optimization_results(
        self, output_path: str, optimization_type: str = "threshold_tuning"
    ):
        """Save optimization results to JSON file.

        Args:
            output_path: Path to save optimization results
            optimization_type: Type of optimization performed

        """
        self.optimization_results["optimization_type"] = optimization_type
        self.optimization_results["timestamp"] = datetime.now().isoformat()
        self.optimization_results["total_validations_analyzed"] = len(
            self.validation_history
        )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(self.optimization_results, f, indent=2)

        logger.info(f"Optimization results saved to {output_file}")

    def run_comprehensive_optimization(
        self, validation_results_path: str, output_dir: str = "results/optimization/"
    ) -> dict[str, Any]:
        """Run comprehensive optimization analysis.

        Args:
            validation_results_path: Path to validation results JSON
            output_dir: Directory to save optimization results

        Returns:
            Comprehensive optimization results

        """
        logger.info("Starting comprehensive threshold optimization...")

        # Load and analyze validation results
        results = self.load_validation_results(validation_results_path)
        analysis = self.analyze_detection_rates(results)
        optimal_thresholds = self.calculate_optimal_thresholds(results)

        # Generate configurations for different scenarios
        scenarios = {
            "optimal_accuracy": optimal_thresholds,
            "kelp_farm_tuned": self.create_adaptive_config(
                "kelp_farm", {"cloud_cover": 0.15, "turbidity": "medium"}
            ),
            "open_ocean_tuned": self.create_adaptive_config(
                "open_ocean", {"cloud_cover": 0.20, "turbidity": "low"}
            ),
            "coastal_tuned": self.create_adaptive_config(
                "coastal", {"cloud_cover": 0.25, "turbidity": "high"}
            ),
            "real_time_optimized": self.optimize_for_real_time(15.0),
        }

        # Store comprehensive results
        self.optimization_results = {
            "current_analysis": analysis,
            "optimized_scenarios": scenarios,
            "recommendations": self._generate_recommendations(analysis),
            "validation_source": validation_results_path,
        }

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_optimization_results(
            f"{output_dir}/optimization_results_{timestamp}.json"
        )

        logger.info("Comprehensive optimization completed successfully!")
        return self.optimization_results

    def _generate_recommendations(self, analysis: dict[str, float]) -> list[str]:
        """Generate optimization recommendations based on analysis.

        Args:
            analysis: Detection rate analysis

        Returns:
            List of optimization recommendations

        """
        recommendations = []

        if analysis["over_detection_ratio"] > 3.0:
            recommendations.append(
                "CRITICAL: Detection thresholds are too low, causing severe over-detection"
            )
            recommendations.append(
                "Increase NDRE threshold to at least 0.1 and FAI threshold to 0.05"
            )

        elif analysis["over_detection_ratio"] > 2.0:
            recommendations.append("HIGH: Significant over-detection detected")
            recommendations.append(
                "Increase detection thresholds by 2-3x current values"
            )

        elif analysis["over_detection_ratio"] > 1.5:
            recommendations.append("MEDIUM: Moderate over-detection")
            recommendations.append("Fine-tune thresholds upward by 50-100%")

        elif analysis["over_detection_ratio"] < 0.7:
            recommendations.append(
                "Under-detection detected - consider lowering thresholds"
            )

        if analysis["accuracy_score"] < 0.3:
            recommendations.append(
                "Poor accuracy score - comprehensive threshold review needed"
            )
        elif analysis["accuracy_score"] < 0.7:
            recommendations.append("Moderate accuracy - fine-tuning recommended")
        else:
            recommendations.append("Good accuracy achieved")

        # Performance recommendations
        recommendations.append(
            "Consider adaptive thresholding based on environmental conditions"
        )
        recommendations.append(
            "Implement real-time optimization for production deployments"
        )

        return recommendations


# Utility functions for integration with validation pipeline


def optimize_detection_pipeline(
    validation_results_path: str, output_dir: str = "results/optimization/"
) -> dict[str, Any]:
    """Optimize detection pipeline based on validation results.

    Args:
        validation_results_path: Path to validation results JSON
        output_dir: Directory to save optimization results

    Returns:
        Optimization results dictionary

    """
    optimizer = ThresholdOptimizer()
    return optimizer.run_comprehensive_optimization(validation_results_path, output_dir)


def get_optimized_config_for_site(
    site_type: str, environmental_conditions: dict[str, Any]
) -> dict[str, Any]:
    """Get optimized configuration for specific site and conditions.

    Args:
        site_type: Type of site ('kelp_farm', 'open_ocean', 'coastal')
        environmental_conditions: Environmental factors

    Returns:
        Optimized configuration dictionary

    """
    optimizer = ThresholdOptimizer()
    return optimizer.create_adaptive_config(site_type, environmental_conditions)
