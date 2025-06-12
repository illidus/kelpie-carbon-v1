"""
Environmental Robustness Testing for SKEMA Kelp Detection.

This module implements comprehensive environmental condition testing for
SKEMA kelp detection algorithms, validating performance across:
- Tidal effects and current variations
- Water clarity conditions (turbid vs clear)
- Seasonal variations and temporal consistency
- Real-world environmental parameter influences

Based on research from Timmer et al. (2024) and SKEMA validation studies.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import xarray as xr

from ...core.fetch import fetch_sentinel_tiles
from ...core.mask import create_skema_kelp_detection_mask
from ...core.processing.derivative_features import DerivativeFeatures
from ...core.processing.water_anomaly_filter import WaterAnomalyFilter

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentalCondition:
    """Represents a specific environmental condition for testing."""

    name: str
    description: str
    parameters: dict[str, Any]
    expected_behavior: str
    tolerance: float = 0.1


@dataclass
class EnvironmentalTestResult:
    """Results from environmental condition testing."""

    condition: EnvironmentalCondition
    detection_rate: float
    consistency_score: float
    performance_metrics: dict[str, float]
    success: bool
    timestamp: datetime
    metadata: dict[str, Any]


class EnvironmentalRobustnessValidator:
    """
    Validates SKEMA kelp detection robustness across environmental conditions.

    Implements testing framework for:
    - Tidal state variations with correction factors
    - Water clarity impact on detection
    - Seasonal consistency validation
    - Multi-date temporal robustness
    """

    def __init__(self):
        self.waf = WaterAnomalyFilter()
        self.derivative_features = DerivativeFeatures()
        self.results: list[EnvironmentalTestResult] = []

    def get_environmental_conditions(self) -> list[EnvironmentalCondition]:
        """Define environmental conditions for testing."""
        return [
            # Tidal Effect Conditions
            EnvironmentalCondition(
                name="low_tide_low_current",
                description="Low tide with low current conditions (<10 cm/s)",
                parameters={
                    "tidal_height": -1.0,  # 1m below mean
                    "current_speed": 5.0,  # cm/s
                    "correction_factor": -0.225,  # 22.5% per meter (Timmer et al. 2024)
                },
                expected_behavior="Reduced kelp extent with low current correction",
                tolerance=0.15,
            ),
            EnvironmentalCondition(
                name="low_tide_high_current",
                description="Low tide with high current conditions (>10 cm/s)",
                parameters={
                    "tidal_height": -1.0,
                    "current_speed": 15.0,  # cm/s
                    "correction_factor": -0.355,  # 35.5% per meter (Timmer et al. 2024)
                },
                expected_behavior="Significantly reduced kelp extent with high current correction",
                tolerance=0.20,
            ),
            EnvironmentalCondition(
                name="high_tide_low_current",
                description="High tide with low current conditions",
                parameters={
                    "tidal_height": 1.0,  # 1m above mean
                    "current_speed": 5.0,
                    "correction_factor": -0.225,
                },
                expected_behavior="Increased kelp extent with low current correction",
                tolerance=0.15,
            ),
            EnvironmentalCondition(
                name="high_tide_high_current",
                description="High tide with high current conditions",
                parameters={
                    "tidal_height": 1.0,
                    "current_speed": 15.0,
                    "correction_factor": -0.355,
                },
                expected_behavior="Increased kelp extent with high current correction",
                tolerance=0.20,
            ),
            # Water Clarity Conditions
            EnvironmentalCondition(
                name="turbid_water",
                description="Turbid water conditions (Secchi depth <4m)",
                parameters={
                    "secchi_depth": 2.0,  # meters
                    "turbidity_factor": 0.8,  # reduced detection capability
                    "waf_intensity": 1.2,  # enhanced WAF application
                },
                expected_behavior="Reduced detection with enhanced WAF processing",
                tolerance=0.25,
            ),
            EnvironmentalCondition(
                name="clear_water",
                description="Clear water conditions (Secchi depth >7m)",
                parameters={
                    "secchi_depth": 10.0,  # meters
                    "turbidity_factor": 1.0,  # normal detection capability
                    "waf_intensity": 1.0,  # standard WAF application
                },
                expected_behavior="Optimal detection with standard processing",
                tolerance=0.10,
            ),
            # Seasonal Conditions
            EnvironmentalCondition(
                name="peak_season",
                description="Peak kelp growth season (July-September)",
                parameters={
                    "growth_factor": 1.2,  # enhanced kelp presence
                    "density_factor": 1.1,  # increased canopy density
                    "biomass_factor": 1.3,  # peak biomass
                },
                expected_behavior="Maximum detection rates and consistency",
                tolerance=0.10,
            ),
            EnvironmentalCondition(
                name="off_season",
                description="Off-season kelp presence (October-April)",
                parameters={
                    "growth_factor": 0.7,  # reduced kelp presence
                    "density_factor": 0.8,  # decreased canopy density
                    "biomass_factor": 0.6,  # reduced biomass
                },
                expected_behavior="Reduced but consistent detection",
                tolerance=0.20,
            ),
        ]

    def apply_tidal_correction(
        self, detection_mask: np.ndarray, condition: EnvironmentalCondition
    ) -> np.ndarray:
        """
        Apply tidal height correction based on Timmer et al. (2024) research.

        Research findings:
        - Low current (<10 cm/s): 22.5% extent decrease per meter
        - High current (>10 cm/s): 35.5% extent decrease per meter
        """
        if "tidal_height" not in condition.parameters:
            return detection_mask

        tidal_height = condition.parameters["tidal_height"]
        correction_factor = condition.parameters["correction_factor"]

        # Apply tidal correction to detection extent
        # Positive tidal_height increases extent, negative decreases
        correction = 1 + (correction_factor * abs(tidal_height))
        if tidal_height > 0:
            # High tide: increase extent (invert correction)
            correction = 1 + abs(correction_factor * tidal_height)
        else:
            # Low tide: decrease extent
            correction = 1 + (correction_factor * abs(tidal_height))

        # Apply correction to detection probability
        corrected_mask = detection_mask * correction

        # Ensure values stay within valid range [0, 1]
        return np.clip(corrected_mask, 0, 1)

    def apply_water_clarity_correction(
        self, dataset: xr.Dataset, condition: EnvironmentalCondition
    ) -> xr.Dataset:
        """Apply water clarity corrections based on Secchi depth."""
        if "secchi_depth" not in condition.parameters:
            return dataset

        secchi_depth = condition.parameters["secchi_depth"]
        turbidity_factor = condition.parameters["turbidity_factor"]
        waf_intensity = condition.parameters["waf_intensity"]

        # Adjust spectral response based on water clarity
        corrected_dataset = dataset.copy()

        # Turbid water reduces spectral contrast
        if secchi_depth < 4.0:  # Turbid conditions
            # Reduce contrast in visible bands
            for band in ["red", "green", "blue"]:
                if band in corrected_dataset:
                    corrected_dataset[band] *= turbidity_factor

            # Apply enhanced WAF processing for turbid conditions
            if waf_intensity > 1.0:
                # Enhanced artifact removal in turbid conditions
                corrected_dataset = self._apply_enhanced_waf(
                    corrected_dataset, waf_intensity
                )

        return corrected_dataset

    def apply_seasonal_correction(
        self, detection_mask: np.ndarray, condition: EnvironmentalCondition
    ) -> np.ndarray:
        """Apply seasonal growth pattern corrections."""
        if "growth_factor" not in condition.parameters:
            return detection_mask

        growth_factor = condition.parameters["growth_factor"]
        density_factor = condition.parameters.get("density_factor", 1.0)

        # Apply seasonal growth and density corrections
        corrected_mask = detection_mask * growth_factor * density_factor

        return np.clip(corrected_mask, 0, 1)

    async def test_environmental_condition(
        self,
        condition: EnvironmentalCondition,
        lat: float,
        lng: float,
        start_date: str,
        end_date: str,
    ) -> EnvironmentalTestResult:
        """Test SKEMA detection under specific environmental conditions."""
        logger.info(f"Testing environmental condition: {condition.name}")

        try:
            # Fetch satellite data for the location
            data_result = fetch_sentinel_tiles(
                lat=lat, lng=lng, start_date=start_date, end_date=end_date
            )
            dataset = data_result.get("data") if data_result else None

            if dataset is None or len(dataset.data_vars) == 0:
                logger.warning(f"No suitable data found for {condition.name}")
                return self._create_failed_result(
                    condition, "No satellite data available"
                )

            # Apply environmental corrections
            corrected_dataset = self.apply_water_clarity_correction(dataset, condition)

            # Generate SKEMA detection mask
            detection_mask = create_skema_kelp_detection_mask(
                corrected_dataset,
                config={
                    "apply_waf": True,
                    "combine_with_ndre": True,
                    "detection_combination": "union",
                    "apply_morphology": True,
                    "min_kelp_cluster_size": 5,
                },
            )

            # Apply environmental corrections to detection
            corrected_mask = self.apply_tidal_correction(detection_mask, condition)
            corrected_mask = self.apply_seasonal_correction(corrected_mask, condition)

            # Calculate performance metrics
            detection_rate = float(np.mean(corrected_mask))
            consistency_score = self._calculate_consistency_score(corrected_mask)

            # Evaluate success based on expected behavior
            success = self._evaluate_condition_success(
                detection_rate, condition, consistency_score
            )

            # Create comprehensive result
            result = EnvironmentalTestResult(
                condition=condition,
                detection_rate=detection_rate,
                consistency_score=consistency_score,
                performance_metrics={
                    "mean_detection": detection_rate,
                    "std_detection": float(np.std(corrected_mask)),
                    "max_detection": float(np.max(corrected_mask)),
                    "min_detection": float(np.min(corrected_mask)),
                    "spatial_consistency": consistency_score,
                },
                success=success,
                timestamp=datetime.now(),
                metadata={
                    "location": {"lat": lat, "lng": lng},
                    "date_range": {"start": start_date, "end": end_date},
                    "dataset_info": {
                        "bands": list(dataset.data_vars.keys()),
                        "spatial_dims": dict(dataset.sizes),
                    },
                    "condition_parameters": condition.parameters,
                },
            )

            self.results.append(result)
            logger.info(
                f"Completed testing {condition.name}: {'PASS' if success else 'FAIL'}"
            )
            return result

        except Exception as e:
            logger.error(f"Error testing condition {condition.name}: {str(e)}")
            return self._create_failed_result(condition, str(e))

    def _apply_enhanced_waf(self, dataset: xr.Dataset, intensity: float) -> xr.Dataset:
        """Apply enhanced Water Anomaly Filter for turbid conditions."""
        # Enhanced WAF processing for challenging conditions
        # Create enhanced config based on intensity factor
        enhanced_config = {
            "sunglint_threshold": 0.15
            / intensity,  # Lower threshold for higher intensity
            "kernel_size": max(
                3, int(5 * intensity)
            ),  # Larger kernel for higher intensity
            "spectral_smoothing": True,
            "artifact_fill_method": "interpolation",
        }
        # Apply WAF with enhanced configuration
        from ...processing.water_anomaly_filter import apply_water_anomaly_filter

        waf_result = apply_water_anomaly_filter(dataset, enhanced_config)
        return waf_result

    def _calculate_consistency_score(self, detection_mask: np.ndarray) -> float:
        """Calculate spatial consistency score for detection."""
        # Measure spatial consistency using coefficient of variation
        if np.mean(detection_mask) == 0:
            return 0.0

        # Lower CV indicates higher spatial consistency
        cv = np.std(detection_mask) / np.mean(detection_mask)
        consistency_score = max(0.0, 1.0 - cv)  # Invert so higher is better

        return float(consistency_score)

    def _evaluate_condition_success(
        self,
        detection_rate: float,
        condition: EnvironmentalCondition,
        consistency_score: float,
    ) -> bool:
        """Evaluate if the condition test was successful."""
        # Define expected detection rate ranges based on condition type
        if "tidal" in condition.name:
            # Tidal conditions should show predictable variations
            if "low_tide" in condition.name:
                expected_range = (0.05, 0.25)  # Reduced extent at low tide
            else:
                expected_range = (0.10, 0.35)  # Increased extent at high tide
        elif "turbid" in condition.name:
            expected_range = (0.05, 0.20)  # Reduced detection in turbid water
        elif "clear" in condition.name:
            expected_range = (0.10, 0.30)  # Better detection in clear water
        elif "peak_season" in condition.name:
            expected_range = (0.15, 0.40)  # Maximum detection in peak season
        elif "off_season" in condition.name:
            expected_range = (0.05, 0.20)  # Reduced detection off-season
        else:
            expected_range = (0.05, 0.30)  # General range

        # Check if detection rate is within expected range
        rate_ok = expected_range[0] <= detection_rate <= expected_range[1]

        # Check if consistency is acceptable (>0.3 is reasonable)
        consistency_ok = consistency_score > 0.3

        return bool(rate_ok and consistency_ok)

    def _create_failed_result(
        self, condition: EnvironmentalCondition, error_message: str
    ) -> EnvironmentalTestResult:
        """Create a failed test result."""
        return EnvironmentalTestResult(
            condition=condition,
            detection_rate=0.0,
            consistency_score=0.0,
            performance_metrics={},
            success=False,
            timestamp=datetime.now(),
            metadata={"error": error_message},
        )

    async def run_comprehensive_environmental_testing(
        self, lat: float, lng: float, base_date: str, date_range_days: int = 30
    ) -> dict[str, Any]:
        """
        Run comprehensive environmental robustness testing.

        Tests all environmental conditions and provides summary report.
        """
        logger.info("Starting comprehensive environmental robustness testing")

        # Calculate date range
        base_datetime = datetime.fromisoformat(base_date)
        start_date = (base_datetime - timedelta(days=date_range_days)).strftime(
            "%Y-%m-%d"
        )
        end_date = base_datetime.strftime("%Y-%m-%d")

        # Get all environmental conditions
        conditions = self.get_environmental_conditions()

        # Test each condition
        results = []
        for condition in conditions:
            result = await self.test_environmental_condition(
                condition, lat, lng, start_date, end_date
            )
            results.append(result)

        # Generate comprehensive report
        report = self._generate_environmental_report(results)

        logger.info(
            f"Environmental testing complete: {report['summary']['success_rate'] * 100:.1f}% success rate"
        )
        return report

    # Convenience method aliases for test compatibility
    async def run_comprehensive_testing(
        self, lat: float, lng: float, base_date: str
    ) -> dict[str, Any]:
        """Convenience method alias for test compatibility."""
        return await self.run_comprehensive_environmental_testing(lat, lng, base_date)

    def _generate_report(
        self, results: list[EnvironmentalTestResult]
    ) -> dict[str, Any]:
        """Convenience method alias for test compatibility."""
        return self._generate_environmental_report(results)

    def _generate_environmental_report(
        self, results: list[EnvironmentalTestResult]
    ) -> dict[str, Any]:
        """Generate comprehensive environmental testing report."""
        successful_tests = [r for r in results if r.success]
        failed_tests = [r for r in results if not r.success]

        # Calculate summary statistics
        detection_rates = [r.detection_rate for r in successful_tests]
        consistency_scores = [r.consistency_score for r in successful_tests]

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_conditions": len(results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(successful_tests) / len(results) if results else 0,
                "average_detection_rate": (
                    np.mean(detection_rates) if detection_rates else 0
                ),
                "average_consistency": (
                    np.mean(consistency_scores) if consistency_scores else 0
                ),
            },
            "condition_categories": {
                "tidal_effects": {
                    "tested": len(
                        [
                            r
                            for r in results
                            if "tidal" in r.condition.name or "tide" in r.condition.name
                        ]
                    ),
                    "successful": len(
                        [
                            r
                            for r in successful_tests
                            if "tidal" in r.condition.name or "tide" in r.condition.name
                        ]
                    ),
                },
                "water_clarity": {
                    "tested": len(
                        [
                            r
                            for r in results
                            if "turbid" in r.condition.name
                            or "clear" in r.condition.name
                        ]
                    ),
                    "successful": len(
                        [
                            r
                            for r in successful_tests
                            if "turbid" in r.condition.name
                            or "clear" in r.condition.name
                        ]
                    ),
                },
                "seasonal_variation": {
                    "tested": len([r for r in results if "season" in r.condition.name]),
                    "successful": len(
                        [r for r in successful_tests if "season" in r.condition.name]
                    ),
                },
            },
            "detailed_results": [
                {
                    "condition_name": r.condition.name,
                    "description": r.condition.description,
                    "detection_rate": r.detection_rate,
                    "consistency_score": r.consistency_score,
                    "success": r.success,
                    "performance_metrics": r.performance_metrics,
                }
                for r in results
            ],
            "recommendations": self._generate_recommendations(results),
        }

    def _generate_recommendations(
        self, results: list[EnvironmentalTestResult]
    ) -> list[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Analyze tidal effects
        tidal_results = [r for r in results if "tide" in r.condition.name]
        if tidal_results:
            successful_tidal = [r for r in tidal_results if r.success]
            if len(successful_tidal) < len(tidal_results):
                recommendations.append(
                    "Consider implementing more robust tidal correction factors for improved accuracy"
                )

        # Analyze water clarity effects
        clarity_results = [
            r
            for r in results
            if "turbid" in r.condition.name or "clear" in r.condition.name
        ]
        turbid_success = any(
            r.success for r in clarity_results if "turbid" in r.condition.name
        )
        if not turbid_success:
            recommendations.append(
                "Enhance Water Anomaly Filter (WAF) processing for turbid water conditions"
            )

        # Analyze seasonal effects
        seasonal_results = [r for r in results if "season" in r.condition.name]
        if seasonal_results:
            off_season_success = any(
                r.success for r in seasonal_results if "off_season" in r.condition.name
            )
            if not off_season_success:
                recommendations.append(
                    "Optimize detection thresholds for off-season kelp detection"
                )

        # General performance recommendations
        overall_success_rate = len([r for r in results if r.success]) / len(results)
        if overall_success_rate < 0.8:
            recommendations.append(
                "Consider recalibrating SKEMA detection parameters for better environmental robustness"
            )

        return recommendations


# Convenience functions for testing specific environmental aspects


async def validate_tidal_effects(
    lat: float, lng: float, base_date: str
) -> dict[str, Any]:
    """Test tidal effect variations on kelp detection."""
    validator = EnvironmentalRobustnessValidator()

    # Get only tidal conditions
    all_conditions = validator.get_environmental_conditions()
    tidal_conditions = [c for c in all_conditions if "tide" in c.name]

    results = []
    for condition in tidal_conditions:
        result = await validator.test_environmental_condition(
            condition, lat, lng, base_date, base_date
        )
        results.append(result)

    return validator._generate_environmental_report(results)


async def validate_water_clarity_effects(
    lat: float, lng: float, base_date: str
) -> dict[str, Any]:
    """Test water clarity variations on kelp detection."""
    validator = EnvironmentalRobustnessValidator()

    # Get only water clarity conditions
    all_conditions = validator.get_environmental_conditions()
    clarity_conditions = [
        c for c in all_conditions if "turbid" in c.name or "clear" in c.name
    ]

    results = []
    for condition in clarity_conditions:
        result = await validator.test_environmental_condition(
            condition, lat, lng, base_date, base_date
        )
        results.append(result)

    return validator._generate_environmental_report(results)


async def validate_seasonal_variations(
    lat: float, lng: float, peak_season_date: str, off_season_date: str
) -> dict[str, Any]:
    """Test seasonal variations on kelp detection."""
    validator = EnvironmentalRobustnessValidator()

    # Get only seasonal conditions
    all_conditions = validator.get_environmental_conditions()
    seasonal_conditions = [c for c in all_conditions if "season" in c.name]

    results = []
    for condition in seasonal_conditions:
        # Use appropriate date for each season
        test_date = peak_season_date if "peak" in condition.name else off_season_date
        result = await validator.test_environmental_condition(
            condition, lat, lng, test_date, test_date
        )
        results.append(result)

    return validator._generate_environmental_report(results)


def create_environmental_validator() -> EnvironmentalRobustnessValidator:
    """Create a configured environmental robustness validator instance.

    Returns:
        Configured EnvironmentalRobustnessValidator instance
    """
    return EnvironmentalRobustnessValidator()


async def run_comprehensive_environmental_testing(
    lat: float, lng: float, base_date: str, date_range_days: int = 30
) -> dict[str, Any]:
    """Run comprehensive environmental robustness testing.

    Args:
        lat: Latitude for testing location
        lng: Longitude for testing location
        base_date: Base date for testing in YYYY-MM-DD format
        date_range_days: Number of days to search for imagery

    Returns:
        Comprehensive environmental testing report
    """
    validator = create_environmental_validator()
    return await validator.run_comprehensive_environmental_testing(
        lat, lng, base_date, date_range_days
    )
