"""Temporal Validation & Environmental Drivers for SKEMA Kelp Detection.

This module implements comprehensive temporal validation following UVic's
Broughton Archipelago methodology for multi-year kelp detection persistence,
seasonal trend analysis, and environmental driver integration.

Key Features:
- Time-series validation across multiple years
- Seasonal consistency analysis
- Environmental driver correlation
- Temporal persistence metrics
- Long-term trend detection

Based on UVic SKEMA research and Timmer et al. (2024) temporal studies.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from ...core.fetch import fetch_sentinel_tiles
from ...core.mask import create_skema_kelp_detection_mask
from ...core.processing.derivative_features import DerivativeFeatures
from ...core.processing.water_anomaly_filter import WaterAnomalyFilter
from .environmental_testing import (
    EnvironmentalRobustnessValidator,
)
from .real_world_validation import RealWorldValidator, ValidationSite

logger = logging.getLogger(__name__)


@dataclass
class TemporalDataPoint:
    """Represents a single temporal observation for kelp detection."""

    timestamp: datetime
    detection_rate: float
    kelp_area_km2: float
    confidence_score: float
    environmental_conditions: dict[str, float]
    processing_metadata: dict[str, Any] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)


@dataclass
class SeasonalPattern:
    """Represents seasonal patterns in kelp detection."""

    season: str  # spring, summer, fall, winter
    average_detection_rate: float
    peak_month: int
    trough_month: int
    variability_coefficient: float
    trend_slope: float
    statistical_significance: float  # p-value for trend


@dataclass
class TemporalValidationResult:
    """Results from temporal validation analysis."""

    site_name: str
    validation_period: tuple[datetime, datetime]
    data_points: list[TemporalDataPoint]
    seasonal_patterns: dict[str, SeasonalPattern]
    persistence_metrics: dict[str, float]
    environmental_correlations: dict[str, float]
    trend_analysis: dict[str, Any]
    quality_assessment: dict[str, Any]
    recommendations: list[str]
    timestamp: datetime = field(default_factory=datetime.now)


class TemporalValidator:
    """Comprehensive temporal validation framework for SKEMA kelp detection.

    Implements UVic's Broughton Archipelago time-series approach for:
    - Multi-year persistence validation
    - Seasonal trend analysis
    - Environmental driver correlation
    - Long-term change detection
    """

    def __init__(self):
        """Initialize temporal validator."""
        self.env_validator = EnvironmentalRobustnessValidator()
        self.real_world_validator = RealWorldValidator()
        self.waf = WaterAnomalyFilter()
        self.derivative_features = DerivativeFeatures()
        self.results: list[TemporalValidationResult] = []

    def get_broughton_validation_config(self) -> dict[str, Any]:
        """Get UVic Broughton Archipelago validation configuration.

        Based on UVic SKEMA research methodology for temporal consistency.
        """
        return {
            "site": ValidationSite(
                name="Broughton Archipelago - Temporal",
                lat=50.0833,
                lng=-126.1667,
                species="Nereocystis luetkeana",
                expected_detection_rate=0.20,  # Higher expectation for temporal validation
                water_depth="7.5m Secchi depth",
                optimal_season="July-September",
                site_type="kelp_farm",
                description="UVic primary temporal validation site for bull kelp",
            ),
            "temporal_parameters": {
                "validation_years": 3,  # Multi-year validation
                "sampling_frequency_days": 15,  # Bi-weekly sampling
                "seasonal_windows": {
                    "spring": (3, 5),  # March-May
                    "summer": (6, 8),  # June-August
                    "fall": (9, 11),  # September-November
                    "winter": (12, 2),  # December-February
                },
                "environmental_drivers": [
                    "tidal_height",
                    "current_speed",
                    "water_temperature",
                    "secchi_depth",
                    "wind_speed",
                    "precipitation",
                ],
            },
            "persistence_thresholds": {
                "minimum_detection_rate": 0.10,  # 10% minimum for persistence
                "consistency_threshold": 0.75,  # 75% consistency required
                "seasonal_variation_max": 0.40,  # Max 40% seasonal variation
                "inter_annual_variation_max": 0.30,  # Max 30% year-to-year variation
            },
        }

    async def validate_temporal_persistence(
        self,
        site: ValidationSite,
        start_date: datetime,
        end_date: datetime,
        sampling_interval_days: int = 15,
    ) -> TemporalValidationResult:
        """Validate temporal persistence following UVic methodology.

        Args:
            site: Validation site configuration
            start_date: Start of validation period
            end_date: End of validation period
            sampling_interval_days: Days between sampling points

        Returns:
            Comprehensive temporal validation results

        """
        logger.info(f"Starting temporal persistence validation for {site.name}")
        logger.info(
            f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        # Generate sampling dates
        sampling_dates = self._generate_sampling_dates(
            start_date, end_date, sampling_interval_days
        )

        # Collect temporal data points
        data_points = []
        for sample_date in sampling_dates:
            try:
                data_point = await self._collect_temporal_data_point(site, sample_date)
                if data_point:
                    data_points.append(data_point)

            except Exception as e:
                logger.warning(f"Failed to collect data for {sample_date}: {e}")

        # Analyze temporal patterns
        seasonal_patterns = self._analyze_seasonal_patterns(data_points)
        persistence_metrics = self._calculate_persistence_metrics(data_points)
        environmental_correlations = self._analyze_environmental_correlations(
            data_points
        )
        trend_analysis = self._perform_trend_analysis(data_points)
        quality_assessment = self._assess_temporal_quality(data_points, site)
        recommendations = self._generate_temporal_recommendations(
            persistence_metrics, seasonal_patterns, trend_analysis
        )

        result = TemporalValidationResult(
            site_name=site.name,
            validation_period=(start_date, end_date),
            data_points=data_points,
            seasonal_patterns=seasonal_patterns,
            persistence_metrics=persistence_metrics,
            environmental_correlations=environmental_correlations,
            trend_analysis=trend_analysis,
            quality_assessment=quality_assessment,
            recommendations=recommendations,
        )

        self.results.append(result)
        logger.info(
            f"Completed temporal validation with {len(data_points)} data points"
        )

        return result

    def _generate_sampling_dates(
        self, start_date: datetime, end_date: datetime, interval_days: int
    ) -> list[datetime]:
        """Generate sampling dates with specified interval."""
        dates = []
        current_date = start_date

        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=interval_days)

        return dates

    async def _collect_temporal_data_point(
        self, site: ValidationSite, sample_date: datetime
    ) -> TemporalDataPoint | None:
        """Collect a single temporal data point with environmental conditions."""
        try:
            # Format date for satellite data fetch
            date_str = sample_date.strftime("%Y-%m-%d")

            # Fetch satellite data
            tiles = await fetch_sentinel_tiles(
                lat=site.lat,
                lng=site.lng,
                start_date=date_str,
                end_date=date_str,
                processing_level="l2a",
            )

            if not tiles:
                logger.warning(f"No satellite data available for {date_str}")
                return None

            # Process first available tile
            dataset = tiles[0]

            # Apply SKEMA kelp detection
            kelp_mask = create_skema_kelp_detection_mask(dataset)

            # Calculate detection metrics
            detection_rate = float(np.mean(kelp_mask))
            kelp_area_km2 = self._calculate_kelp_area(kelp_mask, dataset)
            confidence_score = self._calculate_confidence_score(kelp_mask, dataset)

            # Simulate environmental conditions (in production, fetch from external APIs)
            environmental_conditions = self._simulate_environmental_conditions(
                site, sample_date
            )

            # Quality assessment
            quality_flags = self._assess_data_quality(dataset, kelp_mask)

            return TemporalDataPoint(
                timestamp=sample_date,
                detection_rate=detection_rate,
                kelp_area_km2=kelp_area_km2,
                confidence_score=confidence_score,
                environmental_conditions=environmental_conditions,
                processing_metadata={
                    "satellite_scene_id": dataset.attrs.get("scene_id", "unknown"),
                    "cloud_coverage": dataset.attrs.get("cloud_coverage", 0.0),
                    "processing_level": "l2a",
                },
                quality_flags=quality_flags,
            )

        except Exception as e:
            logger.error(f"Error collecting temporal data point: {e}")
            return None

    def _calculate_kelp_area(self, kelp_mask: np.ndarray, dataset: xr.Dataset) -> float:
        """Calculate kelp area in km² from detection mask."""
        # Get pixel area (simplified - in production use proper geospatial calculations)
        pixel_area_m2 = 100  # Sentinel-2 10m pixels = 100 m²
        kelp_pixels = np.sum(kelp_mask > 0.3)  # 30% threshold for kelp presence
        kelp_area_m2 = kelp_pixels * pixel_area_m2
        kelp_area_km2 = kelp_area_m2 / 1_000_000  # Convert to km²

        return float(kelp_area_km2)

    def _calculate_confidence_score(
        self, kelp_mask: np.ndarray, dataset: xr.Dataset
    ) -> float:
        """Calculate confidence score for detection."""
        # Factors affecting confidence
        mean_detection = np.mean(kelp_mask)
        detection_variance = np.var(kelp_mask)

        # Higher mean detection and lower variance = higher confidence
        confidence = mean_detection * (1.0 - min(detection_variance, 0.5))

        return float(np.clip(confidence, 0.0, 1.0))

    def _simulate_environmental_conditions(
        self, site: ValidationSite, date: datetime
    ) -> dict[str, float]:
        """Simulate environmental conditions for temporal analysis.

        In production, this would fetch real data from:
        - NOAA tidal data
        - Weather APIs
        - Ocean condition databases
        """
        # Seasonal patterns
        day_of_year = date.timetuple().tm_yday

        # Tidal height (simplified sinusoidal pattern)
        tidal_height = np.sin(2 * np.pi * day_of_year / 365.25) * 2.0  # ±2m range

        # Current speed (correlated with tidal activity)
        current_speed = abs(tidal_height) * 5.0 + np.random.normal(8.0, 2.0)  # cm/s

        # Water temperature (seasonal)
        temp_base = 10.0 if site.lat > 45 else 15.0  # Colder for northern sites
        water_temperature = temp_base + 8.0 * np.sin(
            2 * np.pi * (day_of_year - 60) / 365.25
        )

        # Secchi depth (water clarity)
        secchi_depth = 6.0 + 2.0 * np.sin(2 * np.pi * (day_of_year - 120) / 365.25)

        # Wind speed (random with seasonal bias)
        wind_speed = np.random.lognormal(2.0, 0.5) + 2.0 * np.sin(
            2 * np.pi * day_of_year / 365.25
        )

        # Precipitation (higher in winter)
        precip_factor = 1.0 + 0.5 * np.sin(2 * np.pi * (day_of_year + 180) / 365.25)
        precipitation = np.random.exponential(2.0) * precip_factor

        return {
            "tidal_height": float(tidal_height),
            "current_speed": float(current_speed),
            "water_temperature": float(water_temperature),
            "secchi_depth": float(secchi_depth),
            "wind_speed": float(wind_speed),
            "precipitation": float(precipitation),
        }

    def _assess_data_quality(
        self, dataset: xr.Dataset, kelp_mask: np.ndarray
    ) -> list[str]:
        """Assess data quality and flag potential issues."""
        flags = []

        # Cloud coverage check
        cloud_coverage = dataset.attrs.get("cloud_coverage", 0.0)
        if cloud_coverage > 0.2:
            flags.append("high_cloud_coverage")

        # Detection consistency check
        if np.std(kelp_mask) < 0.05:
            flags.append("low_spatial_variation")

        # Extreme values check
        if np.mean(kelp_mask) > 0.8:
            flags.append("extremely_high_detection")
        elif np.mean(kelp_mask) < 0.02:
            flags.append("extremely_low_detection")

        return flags

    def _analyze_seasonal_patterns(
        self, data_points: list[TemporalDataPoint]
    ) -> dict[str, SeasonalPattern]:
        """Analyze seasonal patterns in kelp detection."""
        if not data_points:
            return {}

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(
            [
                {
                    "timestamp": dp.timestamp,
                    "detection_rate": dp.detection_rate,
                    "month": dp.timestamp.month,
                    "season": self._get_season(dp.timestamp.month),
                }
                for dp in data_points
            ]
        )

        seasonal_patterns = {}

        for season in ["spring", "summer", "fall", "winter"]:
            season_data = df[df["season"] == season]

            if len(season_data) < 2:
                continue

            # Calculate seasonal statistics
            avg_detection = season_data["detection_rate"].mean()
            detection_values = season_data["detection_rate"].values

            # Find peak and trough months
            monthly_avg = season_data.groupby("month")["detection_rate"].mean()
            peak_month = int(monthly_avg.idxmax()) if len(monthly_avg) > 0 else 6
            trough_month = int(monthly_avg.idxmin()) if len(monthly_avg) > 0 else 12

            # Calculate variability
            variability_coeff = season_data["detection_rate"].std() / max(
                avg_detection, 0.001
            )

            # Trend analysis (simplified)
            if len(detection_values) > 1:
                slope, _, _, p_value, _ = stats.linregress(
                    range(len(detection_values)), detection_values
                )
            else:
                slope, p_value = 0.0, 1.0

            seasonal_patterns[season] = SeasonalPattern(
                season=season,
                average_detection_rate=float(avg_detection),
                peak_month=peak_month,
                trough_month=trough_month,
                variability_coefficient=float(variability_coeff),
                trend_slope=float(slope),
                statistical_significance=float(p_value),
            )

        return seasonal_patterns

    def _get_season(self, month: int) -> str:
        """Map month to season."""
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "fall"
        else:
            return "winter"

    def _calculate_persistence_metrics(
        self, data_points: list[TemporalDataPoint]
    ) -> dict[str, float]:
        """Calculate temporal persistence metrics."""
        if not data_points:
            return {}

        detection_rates = [dp.detection_rate for dp in data_points]

        # Basic statistics
        mean_detection = np.mean(detection_rates)
        std_detection = np.std(detection_rates)
        coefficient_of_variation = std_detection / max(mean_detection, 0.001)

        # Persistence measures
        detection_threshold = 0.1  # 10% detection threshold
        persistent_detections = sum(
            1 for rate in detection_rates if rate >= detection_threshold
        )
        persistence_rate = persistent_detections / len(detection_rates)

        # Temporal consistency (how often consecutive measurements are similar)
        consecutive_similar = 0
        similarity_threshold = 0.2  # 20% difference threshold

        for i in range(1, len(detection_rates)):
            if abs(detection_rates[i] - detection_rates[i - 1]) <= similarity_threshold:
                consecutive_similar += 1

        consistency_rate = consecutive_similar / max(len(detection_rates) - 1, 1)

        # Trend stability (low absolute slope indicates stable trend)
        if len(detection_rates) > 1:
            slope, _, r_value, p_value, _ = stats.linregress(
                range(len(detection_rates)), detection_rates
            )
            trend_stability = 1.0 - min(abs(slope), 1.0)  # Inverse of slope magnitude
        else:
            slope, r_value, p_value, trend_stability = 0.0, 0.0, 1.0, 1.0

        return {
            "mean_detection_rate": float(mean_detection),
            "std_detection_rate": float(std_detection),
            "coefficient_of_variation": float(coefficient_of_variation),
            "persistence_rate": float(persistence_rate),
            "consistency_rate": float(consistency_rate),
            "trend_slope": float(slope),
            "trend_r_squared": float(r_value**2),
            "trend_p_value": float(p_value),
            "trend_stability": float(trend_stability),
            "temporal_coverage": float(len(data_points)),
            "data_quality_score": self._calculate_data_quality_score(data_points),
        }

    def _calculate_data_quality_score(
        self, data_points: list[TemporalDataPoint]
    ) -> float:
        """Calculate overall data quality score (0-1)."""
        if not data_points:
            return 0.0

        # Factors affecting quality
        total_flags = sum(len(dp.quality_flags) for dp in data_points)
        avg_confidence = np.mean([dp.confidence_score for dp in data_points])

        # Quality score: high confidence, low flags
        flag_penalty = min(total_flags / len(data_points) * 0.1, 0.3)
        quality_score = avg_confidence - flag_penalty

        return float(np.clip(quality_score, 0.0, 1.0))

    def _analyze_environmental_correlations(
        self, data_points: list[TemporalDataPoint]
    ) -> dict[str, float]:
        """Analyze correlations between detection rates and environmental drivers."""
        if len(data_points) < 3:
            return {}

        detection_rates = [dp.detection_rate for dp in data_points]
        correlations = {}

        # Get all environmental variables
        env_vars = set()
        for dp in data_points:
            env_vars.update(dp.environmental_conditions.keys())

        for var in env_vars:
            env_values = []
            for dp in data_points:
                if var in dp.environmental_conditions:
                    env_values.append(dp.environmental_conditions[var])
                else:
                    env_values.append(0.0)

            if len(env_values) == len(detection_rates) and np.std(env_values) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    correlation, p_value = stats.pearsonr(detection_rates, env_values)

                if not np.isnan(correlation):
                    correlations[var] = float(correlation)
                    correlations[f"{var}_p_value"] = float(p_value)

        return correlations

    def _perform_trend_analysis(
        self, data_points: list[TemporalDataPoint]
    ) -> dict[str, Any]:
        """Perform comprehensive trend analysis."""
        if len(data_points) < 3:
            return {}

        # Convert timestamps to numeric for trend analysis
        timestamps = [dp.timestamp.timestamp() for dp in data_points]
        detection_rates = [dp.detection_rate for dp in data_points]

        # Linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            timestamps, detection_rates
        )

        # Seasonal decomposition (simplified)
        # In production, use more sophisticated seasonal decomposition
        monthly_trends = {}
        df = pd.DataFrame(
            [
                {
                    "timestamp": dp.timestamp,
                    "detection_rate": dp.detection_rate,
                    "month": dp.timestamp.month,
                }
                for dp in data_points
            ]
        )

        for month in range(1, 13):
            month_data = df[df["month"] == month]["detection_rate"]
            if len(month_data) > 1:
                monthly_trends[f"month_{month}"] = float(month_data.mean())

        # Change point detection (simplified)
        change_points = []
        if len(detection_rates) > 5:
            # Simple change point detection based on significant jumps
            for i in range(2, len(detection_rates) - 2):
                before_avg = np.mean(detection_rates[max(0, i - 2) : i])
                after_avg = np.mean(
                    detection_rates[i + 1 : min(len(detection_rates), i + 3)]
                )

                if abs(after_avg - before_avg) > 0.15:  # 15% change threshold
                    change_points.append(
                        {
                            "index": i,
                            "timestamp": data_points[i].timestamp.isoformat(),
                            "magnitude": float(after_avg - before_avg),
                        }
                    )

        return {
            "linear_trend": {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "std_error": float(std_err),
                "trend_direction": "increasing" if slope > 0 else "decreasing",
                "trend_strength": (
                    "strong"
                    if abs(r_value) > 0.7
                    else "moderate"
                    if abs(r_value) > 0.3
                    else "weak"
                ),
            },
            "monthly_patterns": monthly_trends,
            "change_points": change_points,
            "variability_analysis": {
                "overall_variance": float(np.var(detection_rates)),
                "trend_adjusted_variance": float(
                    np.var(detection_rates) - slope**2 * np.var(timestamps)
                ),
                "periodic_component": self._estimate_periodic_component(data_points),
            },
        }

    def _estimate_periodic_component(
        self, data_points: list[TemporalDataPoint]
    ) -> float:
        """Estimate strength of periodic/seasonal component."""
        if len(data_points) < 12:
            return 0.0

        # Simple seasonal strength estimation
        monthly_avgs = {}
        for dp in data_points:
            month = dp.timestamp.month
            if month not in monthly_avgs:
                monthly_avgs[month] = []
            monthly_avgs[month].append(dp.detection_rate)

        # Calculate variance across monthly averages
        month_means = [np.mean(values) for values in monthly_avgs.values()]
        if len(month_means) > 1:
            seasonal_variance = np.var(month_means)
            total_variance = np.var([dp.detection_rate for dp in data_points])
            periodic_strength = seasonal_variance / max(total_variance, 0.001)
            return float(min(periodic_strength, 1.0))

        return 0.0

    def _assess_temporal_quality(
        self, data_points: list[TemporalDataPoint], site: ValidationSite
    ) -> dict[str, Any]:
        """Assess overall temporal validation quality."""
        config = self.get_broughton_validation_config()
        thresholds = config["persistence_thresholds"]

        if not data_points:
            return {
                "overall_quality": "insufficient_data",
                "data_coverage": 0.0,
                "temporal_gaps": [],
                "quality_score": 0.0,
            }

        # Calculate quality metrics
        mean_detection = np.mean([dp.detection_rate for dp in data_points])
        detection_consistency = self._calculate_detection_consistency(data_points)

        # Assess against thresholds
        quality_checks = {
            "meets_minimum_detection": mean_detection
            >= thresholds["minimum_detection_rate"],
            "meets_consistency_threshold": detection_consistency
            >= thresholds["consistency_threshold"],
            "sufficient_data_points": len(data_points) >= 10,
            "good_temporal_coverage": self._assess_temporal_coverage(data_points),
        }

        # Calculate overall quality score
        quality_score = sum(quality_checks.values()) / len(quality_checks)

        # Quality assessment
        if quality_score >= 0.8:
            overall_quality = "excellent"
        elif quality_score >= 0.6:
            overall_quality = "good"
        elif quality_score >= 0.4:
            overall_quality = "moderate"
        else:
            overall_quality = "limited"

        return {
            "overall_quality": overall_quality,
            "quality_score": float(quality_score),
            "quality_checks": quality_checks,
            "data_coverage": self._calculate_data_coverage(data_points),
            "temporal_gaps": self._identify_temporal_gaps(data_points),
            "recommendations": self._generate_quality_recommendations(quality_checks),
        }

    def _calculate_detection_consistency(
        self, data_points: list[TemporalDataPoint]
    ) -> float:
        """Calculate detection consistency across temporal samples."""
        if len(data_points) < 2:
            return 0.0

        detection_rates = [dp.detection_rate for dp in data_points]
        coefficient_of_variation = np.std(detection_rates) / max(
            np.mean(detection_rates), 0.001
        )

        # Convert CV to consistency score (inverse relationship)
        consistency = 1.0 / (1.0 + coefficient_of_variation)
        return float(consistency)

    def _assess_temporal_coverage(self, data_points: list[TemporalDataPoint]) -> bool:
        """Assess if temporal coverage spans sufficient time periods."""
        if len(data_points) < 4:
            return False

        # Check if we have data across multiple seasons
        months = {dp.timestamp.month for dp in data_points}
        seasons_covered = len({self._get_season(month) for month in months})

        return seasons_covered >= 3  # At least 3 seasons covered

    def _calculate_data_coverage(self, data_points: list[TemporalDataPoint]) -> float:
        """Calculate temporal data coverage percentage."""
        if not data_points:
            return 0.0

        # Calculate expected vs actual data points
        start_date = min(dp.timestamp for dp in data_points)
        end_date = max(dp.timestamp for dp in data_points)
        total_days = (end_date - start_date).days

        if total_days == 0:
            return 1.0

        # Assume 15-day sampling interval
        expected_points = max(total_days // 15, 1)
        coverage = min(len(data_points) / expected_points, 1.0)

        return float(coverage)

    def _identify_temporal_gaps(
        self, data_points: list[TemporalDataPoint]
    ) -> list[dict[str, str]]:
        """Identify significant gaps in temporal coverage."""
        if len(data_points) < 2:
            return []

        # Sort by timestamp
        sorted_points = sorted(data_points, key=lambda x: x.timestamp)
        gaps = []

        for i in range(1, len(sorted_points)):
            gap_days = (
                sorted_points[i].timestamp - sorted_points[i - 1].timestamp
            ).days

            if gap_days > 30:  # Gap longer than 30 days
                gaps.append(
                    {
                        "start_date": sorted_points[i - 1].timestamp.strftime(
                            "%Y-%m-%d"
                        ),
                        "end_date": sorted_points[i].timestamp.strftime("%Y-%m-%d"),
                        "gap_days": gap_days,
                    }
                )

        return gaps

    def _generate_quality_recommendations(
        self, quality_checks: dict[str, bool]
    ) -> list[str]:
        """Generate recommendations based on quality assessment."""
        recommendations = []

        if not quality_checks.get("meets_minimum_detection", True):
            recommendations.append(
                "Increase detection sensitivity or verify site has sufficient kelp presence"
            )

        if not quality_checks.get("meets_consistency_threshold", True):
            recommendations.append(
                "Investigate sources of temporal inconsistency in detection"
            )

        if not quality_checks.get("sufficient_data_points", True):
            recommendations.append(
                "Collect additional temporal samples for robust analysis"
            )

        if not quality_checks.get("good_temporal_coverage", True):
            recommendations.append("Extend temporal coverage to include all seasons")

        if not recommendations:
            recommendations.append(
                "Temporal validation quality is acceptable for production use"
            )

        return recommendations

    def _generate_temporal_recommendations(
        self,
        persistence_metrics: dict[str, float],
        seasonal_patterns: dict[str, SeasonalPattern],
        trend_analysis: dict[str, Any],
    ) -> list[str]:
        """Generate comprehensive temporal validation recommendations."""
        recommendations = []

        # Persistence recommendations
        if persistence_metrics.get("persistence_rate", 0) < 0.6:
            recommendations.append(
                "Low persistence rate - consider site-specific calibration"
            )

        if persistence_metrics.get("coefficient_of_variation", 1) > 0.5:
            recommendations.append(
                "High temporal variability - implement adaptive thresholding"
            )

        # Seasonal recommendations
        if seasonal_patterns:
            summer_pattern = seasonal_patterns.get("summer")
            if summer_pattern and summer_pattern.average_detection_rate < 0.15:
                recommendations.append(
                    "Low summer detection rates - verify optimal processing parameters"
                )

        # Trend recommendations
        trend = trend_analysis.get("linear_trend", {})
        if trend.get("trend_strength") == "strong" and trend.get("p_value", 1) < 0.05:
            if trend.get("slope", 0) < -0.01:
                recommendations.append(
                    "Significant declining trend detected - investigate environmental changes"
                )
            elif trend.get("slope", 0) > 0.01:
                recommendations.append(
                    "Significant increasing trend detected - validate against ground truth"
                )

        # Quality recommendations
        data_quality = persistence_metrics.get("data_quality_score", 0)
        if data_quality < 0.7:
            recommendations.append(
                "Improve data quality through enhanced preprocessing"
            )

        if not recommendations:
            recommendations.append(
                "Temporal validation shows good stability for operational deployment"
            )

        return recommendations

    async def run_broughton_archipelago_validation(
        self, validation_years: int = 3
    ) -> TemporalValidationResult:
        """Run comprehensive Broughton Archipelago temporal validation.

        Implements UVic's multi-year validation methodology.
        """
        config = self.get_broughton_validation_config()
        site = config["site"]

        # Calculate validation period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * validation_years)

        logger.info(
            f"Starting Broughton Archipelago {validation_years}-year temporal validation"
        )

        result = await self.validate_temporal_persistence(
            site=site,
            start_date=start_date,
            end_date=end_date,
            sampling_interval_days=config["temporal_parameters"][
                "sampling_frequency_days"
            ],
        )

        return result

    def generate_comprehensive_temporal_report(
        self, results: list[TemporalValidationResult]
    ) -> dict[str, Any]:
        """Generate comprehensive temporal validation report."""
        if not results:
            return {"error": "No temporal validation results available"}

        # Aggregate metrics across all results
        all_persistence_metrics = []
        all_seasonal_patterns = {}
        all_recommendations = []

        for result in results:
            all_persistence_metrics.append(result.persistence_metrics)
            all_recommendations.extend(result.recommendations)

            for season, pattern in result.seasonal_patterns.items():
                if season not in all_seasonal_patterns:
                    all_seasonal_patterns[season] = []
                all_seasonal_patterns[season].append(pattern)

        # Generate summary statistics
        summary_metrics = self._calculate_summary_metrics(all_persistence_metrics)
        seasonal_summary = self._summarize_seasonal_patterns(all_seasonal_patterns)

        # Determine overall assessment
        overall_assessment = self._determine_overall_assessment(
            summary_metrics, seasonal_summary
        )

        return {
            "executive_summary": {
                "total_sites_validated": len(results),
                "validation_period_days": sum(
                    (r.validation_period[1] - r.validation_period[0]).days
                    for r in results
                )
                // len(results),
                "overall_assessment": overall_assessment,
                "mean_persistence_rate": summary_metrics.get(
                    "mean_persistence_rate", 0.0
                ),
                "temporal_stability": summary_metrics.get("mean_trend_stability", 0.0),
            },
            "detailed_metrics": {
                "persistence_metrics_summary": summary_metrics,
                "seasonal_patterns_summary": seasonal_summary,
                "site_specific_results": [
                    {
                        "site_name": r.site_name,
                        "data_points": len(r.data_points),
                        "persistence_rate": r.persistence_metrics.get(
                            "persistence_rate", 0.0
                        ),
                        "quality_score": r.quality_assessment.get("quality_score", 0.0),
                    }
                    for r in results
                ],
            },
            "recommendations": {
                "high_priority": self._extract_high_priority_recommendations(
                    all_recommendations
                ),
                "operational": self._extract_operational_recommendations(
                    all_recommendations
                ),
                "research": self._extract_research_recommendations(all_recommendations),
            },
            "validation_quality": {
                "data_coverage": np.mean(
                    [r.quality_assessment.get("data_coverage", 0.0) for r in results]
                ),
                "overall_quality_distribution": self._calculate_quality_distribution(
                    results
                ),
            },
            "timestamp": datetime.now().isoformat(),
            "methodology": "UVic Broughton Archipelago Multi-Year Temporal Validation",
        }

    def _calculate_summary_metrics(
        self, all_persistence_metrics: list[dict[str, float]]
    ) -> dict[str, float]:
        """Calculate summary statistics across all persistence metrics."""
        if not all_persistence_metrics:
            return {}

        # Aggregate all metrics
        aggregated = {}
        for metric_name in all_persistence_metrics[0]:
            values = [
                metrics.get(metric_name, 0.0) for metrics in all_persistence_metrics
            ]
            aggregated[f"mean_{metric_name}"] = float(np.mean(values))
            aggregated[f"std_{metric_name}"] = float(np.std(values))
            aggregated[f"min_{metric_name}"] = float(np.min(values))
            aggregated[f"max_{metric_name}"] = float(np.max(values))

        return aggregated

    def _summarize_seasonal_patterns(
        self, all_seasonal_patterns: dict[str, list[SeasonalPattern]]
    ) -> dict[str, dict[str, float]]:
        """Summarize seasonal patterns across sites."""
        summary = {}

        for season, patterns in all_seasonal_patterns.items():
            if patterns:
                summary[season] = {
                    "mean_detection_rate": float(
                        np.mean([p.average_detection_rate for p in patterns])
                    ),
                    "mean_variability": float(
                        np.mean([p.variability_coefficient for p in patterns])
                    ),
                    "sites_count": len(patterns),
                    "significant_trends": sum(
                        1 for p in patterns if p.statistical_significance < 0.05
                    ),
                }

        return summary

    def _determine_overall_assessment(
        self,
        summary_metrics: dict[str, float],
        seasonal_summary: dict[str, dict[str, float]],
    ) -> str:
        """Determine overall temporal validation assessment."""
        persistence_rate = summary_metrics.get("mean_persistence_rate", 0.0)
        stability = summary_metrics.get("mean_trend_stability", 0.0)

        if persistence_rate >= 0.8 and stability >= 0.8:
            return "Excellent - High temporal stability and persistence"
        elif persistence_rate >= 0.6 and stability >= 0.6:
            return "Good - Acceptable temporal characteristics for production"
        elif persistence_rate >= 0.4 and stability >= 0.4:
            return "Moderate - Requires monitoring and potential calibration"
        else:
            return "Limited - Significant temporal inconsistencies detected"

    def _extract_high_priority_recommendations(
        self, all_recommendations: list[str]
    ) -> list[str]:
        """Extract high-priority recommendations."""
        high_priority_keywords = [
            "declining trend",
            "insufficient data",
            "low detection",
            "calibration",
        ]

        high_priority = []
        for rec in set(all_recommendations):  # Remove duplicates
            if any(keyword in rec.lower() for keyword in high_priority_keywords):
                high_priority.append(rec)

        return high_priority[:5]  # Top 5 high-priority

    def _extract_operational_recommendations(
        self, all_recommendations: list[str]
    ) -> list[str]:
        """Extract operational recommendations."""
        operational_keywords = ["adaptive", "threshold", "processing", "parameter"]

        operational = []
        for rec in set(all_recommendations):
            if any(keyword in rec.lower() for keyword in operational_keywords):
                operational.append(rec)

        return operational[:5]  # Top 5 operational

    def _extract_research_recommendations(
        self, all_recommendations: list[str]
    ) -> list[str]:
        """Extract research-oriented recommendations."""
        research_keywords = [
            "investigate",
            "environmental",
            "ground truth",
            "validation",
        ]

        research = []
        for rec in set(all_recommendations):
            if any(keyword in rec.lower() for keyword in research_keywords):
                research.append(rec)

        return research[:5]  # Top 5 research

    def _calculate_quality_distribution(
        self, results: list[TemporalValidationResult]
    ) -> dict[str, int]:
        """Calculate distribution of validation quality levels."""
        quality_counts = {"excellent": 0, "good": 0, "moderate": 0, "limited": 0}

        for result in results:
            quality = result.quality_assessment.get("overall_quality", "limited")
            if quality in quality_counts:
                quality_counts[quality] += 1

        return quality_counts


# Factory functions for easy access
def create_temporal_validator() -> TemporalValidator:
    """Create a temporal validator instance."""
    return TemporalValidator()


async def run_broughton_temporal_validation(
    validation_years: int = 3,
) -> TemporalValidationResult:
    """Run Broughton Archipelago temporal validation."""
    validator = create_temporal_validator()
    return await validator.run_broughton_archipelago_validation(validation_years)


async def run_comprehensive_temporal_analysis(
    sites: list[ValidationSite], validation_years: int = 2
) -> dict[str, Any]:
    """Run comprehensive temporal analysis across multiple sites."""
    validator = create_temporal_validator()
    results = []

    for site in sites:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * validation_years)

        result = await validator.validate_temporal_persistence(
            site=site, start_date=start_date, end_date=end_date
        )
        results.append(result)

    return validator.generate_comprehensive_temporal_report(results)
