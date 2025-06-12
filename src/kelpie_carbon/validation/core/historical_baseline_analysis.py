"""
Historical Baseline Analysis for Kelp Detection

This module implements comprehensive historical baseline analysis capabilities,
including historical data digitization, change detection algorithms, and temporal
trend analysis following UVic methodology (1858-1956 historical charts).

Classes:
    HistoricalSite: Represents a historical kelp monitoring site
    HistoricalDataset: Manages historical kelp extent datasets
    ChangeDetectionAnalyzer: Implements change detection algorithms
    TemporalTrendAnalyzer: Provides temporal trend analysis
    HistoricalBaselineAnalysis: Main analysis framework
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class HistoricalSite:
    """Represents a historical kelp monitoring site with metadata."""

    name: str
    latitude: float
    longitude: float
    region: str
    historical_period: tuple[int, int]  # (start_year, end_year)
    data_sources: list[str]
    species: list[str]
    chart_references: list[str] = field(default_factory=list)
    digitization_quality: str = "high"  # high, medium, low
    notes: str = ""

    def __post_init__(self):
        """Validate historical site data."""
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"Invalid longitude: {self.longitude}")
        if self.historical_period[0] > self.historical_period[1]:
            raise ValueError("Start year must be <= end year")
        if self.digitization_quality not in ["high", "medium", "low"]:
            raise ValueError("Quality must be 'high', 'medium', or 'low'")


@dataclass
class HistoricalDataset:
    """Represents a complete historical kelp extent dataset."""

    site: HistoricalSite
    temporal_data: dict[int, dict[str, float]]  # year -> {extent, confidence, etc}
    baseline_extent: float
    confidence_intervals: dict[int, tuple[float, float]]
    data_quality_metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate historical dataset."""
        if not self.temporal_data:
            raise ValueError("Temporal data cannot be empty")
        if self.baseline_extent < 0:
            raise ValueError("Baseline extent must be >= 0")


class ChangeDetectionAnalyzer:
    """Implements change detection algorithms for historical vs. current kelp extent."""

    def __init__(self):
        self.change_threshold = 0.15  # 15% change significance threshold
        self.trend_window = 10  # years for trend analysis

    def detect_significant_changes(
        self,
        historical_data: dict[int, float],
        current_data: dict[int, float],
        method: str = "mann_kendall",
    ) -> dict[str, Any]:
        """
        Detect statistically significant changes between historical and current data.

        Args:
            historical_data: Year -> extent mapping for historical period
            current_data: Year -> extent mapping for current period
            method: Statistical test method ('mann_kendall', 't_test', 'wilcoxon')

        Returns:
            Dictionary with change detection results
        """
        try:
            # Calculate basic change metrics
            hist_mean = np.mean(list(historical_data.values()))
            curr_mean = np.mean(list(current_data.values()))

            absolute_change = curr_mean - hist_mean
            relative_change = (
                (absolute_change / hist_mean) * 100 if hist_mean > 0 else 0
            )

            # Perform statistical test
            hist_values = list(historical_data.values())
            curr_values = list(current_data.values())

            if method == "mann_kendall":
                statistic, p_value = self._mann_kendall_test(hist_values + curr_values)
            elif method == "t_test":
                statistic, p_value = stats.ttest_ind(hist_values, curr_values)
            elif method == "wilcoxon":
                # For paired comparison, pad shorter series
                min_len = min(len(hist_values), len(curr_values))
                statistic, p_value = stats.wilcoxon(
                    hist_values[:min_len], curr_values[:min_len]
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            # Determine significance
            is_significant = p_value < 0.05
            change_magnitude = (
                "large"
                if abs(relative_change) > 25
                else "medium"
                if abs(relative_change) > 10
                else "small"
            )

            return {
                "historical_mean": hist_mean,
                "current_mean": curr_mean,
                "absolute_change": absolute_change,
                "relative_change_percent": relative_change,
                "statistical_test": method,
                "test_statistic": statistic,
                "p_value": p_value,
                "is_significant": is_significant,
                "change_magnitude": change_magnitude,
                "confidence_level": 0.95,
                "sample_sizes": {
                    "historical": len(hist_values),
                    "current": len(curr_values),
                },
            }

        except Exception as e:
            logger.error(f"Change detection failed: {e}")
            return {"error": str(e)}

    def _mann_kendall_test(self, data: list[float]) -> tuple[float, float]:
        """Implement Mann-Kendall trend test."""
        n = len(data)
        if n < 3:
            return 0.0, 1.0

        # Calculate S statistic
        S = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if data[j] > data[i]:
                    S += 1
                elif data[j] < data[i]:
                    S -= 1

        # Calculate variance
        var_S = n * (n - 1) * (2 * n + 5) / 18

        # Calculate Z statistic
        if S > 0:
            Z = (S - 1) / np.sqrt(var_S)
        elif S < 0:
            Z = (S + 1) / np.sqrt(var_S)
        else:
            Z = 0

        # Calculate p-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(Z)))

        return Z, p_value

    def analyze_change_patterns(
        self,
        historical_dataset: HistoricalDataset,
        current_extent: float,
        analysis_year: int = 2024,
    ) -> dict[str, Any]:
        """
        Analyze patterns in kelp extent changes over time.

        Args:
            historical_dataset: Historical kelp dataset
            current_extent: Current kelp extent measurement
            analysis_year: Year of current measurement

        Returns:
            Comprehensive change pattern analysis
        """
        try:
            temporal_data = historical_dataset.temporal_data
            years = sorted(temporal_data.keys())
            extents = [temporal_data[year]["extent"] for year in years]

            # Add current measurement
            all_years = years + [analysis_year]
            all_extents = extents + [current_extent]

            # Calculate trend metrics
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                all_years, all_extents
            )

            # Identify change points using DBSCAN clustering
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(np.array(all_extents).reshape(-1, 1))
            clustering = DBSCAN(eps=0.5, min_samples=2).fit(data_scaled)

            # Calculate decadal averages
            decades = {}
            for year, extent in zip(all_years, all_extents, strict=False):
                decade = (year // 10) * 10
                if decade not in decades:
                    decades[decade] = []
                decades[decade].append(extent)

            decade_averages = {d: np.mean(extents) for d, extents in decades.items()}

            return {
                "trend_analysis": {
                    "slope": slope,
                    "slope_units": "hectares_per_year",
                    "r_squared": r_value**2,
                    "p_value": p_value,
                    "trend_direction": (
                        "increasing"
                        if slope > 0
                        else "decreasing"
                        if slope < 0
                        else "stable"
                    ),
                    "trend_significance": (
                        "significant" if p_value < 0.05 else "not_significant"
                    ),
                },
                "decadal_averages": decade_averages,
                "change_points": {
                    "n_clusters": len(set(clustering.labels_))
                    - (1 if -1 in clustering.labels_ else 0),
                    "cluster_labels": clustering.labels_.tolist(),
                },
                "variability_metrics": {
                    "historical_std": np.std(extents),
                    "historical_cv": (
                        np.std(extents) / np.mean(extents)
                        if np.mean(extents) > 0
                        else 0
                    ),
                    "range": max(all_extents) - min(all_extents),
                },
                "baseline_comparison": {
                    "baseline_extent": historical_dataset.baseline_extent,
                    "current_vs_baseline": current_extent
                    - historical_dataset.baseline_extent,
                    "percent_change_from_baseline": (
                        (
                            (current_extent - historical_dataset.baseline_extent)
                            / historical_dataset.baseline_extent
                        )
                        * 100
                        if historical_dataset.baseline_extent > 0
                        else 0
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Change pattern analysis failed: {e}")
            return {"error": str(e)}


class TemporalTrendAnalyzer:
    """Provides comprehensive temporal trend analysis for kelp extent data."""

    def __init__(self):
        self.seasonal_window = 5  # years for seasonal analysis
        self.forecast_years = 10  # years to forecast

    def analyze_temporal_trends(
        self, historical_dataset: HistoricalDataset, include_forecast: bool = True
    ) -> dict[str, Any]:
        """
        Perform comprehensive temporal trend analysis.

        Args:
            historical_dataset: Historical kelp dataset
            include_forecast: Whether to include future projections

        Returns:
            Comprehensive temporal trend analysis results
        """
        try:
            temporal_data = historical_dataset.temporal_data
            years = sorted(temporal_data.keys())
            extents = [temporal_data[year]["extent"] for year in years]

            # Basic trend analysis
            trend_results = self._calculate_trend_metrics(years, extents)

            # Seasonal analysis (if enough data)
            seasonal_results = self._analyze_seasonal_patterns(temporal_data)

            # Cyclical analysis
            cyclical_results = self._detect_cyclical_patterns(years, extents)

            # Forecast (if requested)
            forecast_results = {}
            if include_forecast:
                forecast_results = self._generate_forecast(years, extents)

            # Risk assessment
            risk_assessment = self._assess_trend_risks(trend_results, forecast_results)

            return {
                "trend_metrics": trend_results,
                "seasonal_patterns": seasonal_results,
                "cyclical_patterns": cyclical_results,
                "forecast": forecast_results,
                "risk_assessment": risk_assessment,
                "data_quality": {
                    "temporal_coverage_years": len(years),
                    "data_completeness": len([x for x in extents if x > 0])
                    / len(extents),
                    "average_confidence": np.mean(
                        [temporal_data[year].get("confidence", 0.8) for year in years]
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Temporal trend analysis failed: {e}")
            return {"error": str(e)}

    def _calculate_trend_metrics(
        self, years: list[int], extents: list[float]
    ) -> dict[str, Any]:
        """Calculate comprehensive trend metrics."""
        if len(years) < 3:
            return {"error": "Insufficient data for trend analysis"}

        # Linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, extents)

        # Polynomial trend (quadratic)
        poly_coeffs = np.polyfit(years, extents, 2)
        poly_r2 = 1 - (
            np.sum((extents - np.polyval(poly_coeffs, years)) ** 2)
            / np.sum((extents - np.mean(extents)) ** 2)
        )

        # Rate of change analysis
        rates = np.diff(extents) / np.diff(years)

        return {
            "linear_trend": {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value**2,
                "p_value": p_value,
                "standard_error": std_err,
            },
            "polynomial_trend": {
                "coefficients": poly_coeffs.tolist(),
                "r_squared": poly_r2,
            },
            "rate_of_change": {
                "mean_rate": np.mean(rates),
                "rate_std": np.std(rates),
                "acceleration": np.mean(np.diff(rates)),
            },
        }

    def _analyze_seasonal_patterns(
        self, temporal_data: dict[int, dict[str, float]]
    ) -> dict[str, Any]:
        """Analyze seasonal patterns if data includes sub-annual resolution."""
        # For now, assume annual data - could be extended for monthly/seasonal data
        years = list(temporal_data.keys())
        if len(years) < 10:
            return {"note": "Insufficient data for robust seasonal analysis"}

        # Look for periodic patterns
        extents = [temporal_data[year]["extent"] for year in sorted(years)]

        # Simple period detection using FFT
        fft = np.fft.fft(extents)
        freqs = np.fft.fftfreq(len(extents))

        # Find dominant frequency
        dominant_freq_idx = np.argmax(np.abs(fft[1 : len(fft) // 2])) + 1
        dominant_period = (
            1 / freqs[dominant_freq_idx] if freqs[dominant_freq_idx] != 0 else 0
        )

        return {
            "dominant_period_years": abs(dominant_period),
            "periodicity_strength": np.abs(fft[dominant_freq_idx])
            / np.sum(np.abs(fft)),
            "note": "Basic periodicity analysis - extend for seasonal data",
        }

    def _detect_cyclical_patterns(
        self, years: list[int], extents: list[float]
    ) -> dict[str, Any]:
        """Detect cyclical patterns in kelp extent data."""
        if len(years) < 10:
            return {"note": "Insufficient data for cyclical analysis"}

        # Detrend data
        slope, intercept, _, _, _ = stats.linregress(years, extents)
        detrended = extents - (slope * np.array(years) + intercept)

        # Autocorrelation analysis
        max_lag = min(len(years) // 3, 20)
        autocorr = [
            np.corrcoef(detrended[:-lag], detrended[lag:])[0, 1]
            for lag in range(1, max_lag + 1)
        ]

        # Find significant cycles
        significant_lags = [
            lag + 1 for lag, corr in enumerate(autocorr) if abs(corr) > 0.3
        ]

        return {
            "autocorrelation": autocorr,
            "significant_cycles": significant_lags,
            "max_autocorr": max(autocorr) if autocorr else 0,
            "cycle_strength": np.std(autocorr) if autocorr else 0,
        }

    def _generate_forecast(
        self, years: list[int], extents: list[float]
    ) -> dict[str, Any]:
        """Generate future extent forecasts."""
        if len(years) < 5:
            return {"note": "Insufficient data for forecasting"}

        # Simple linear extrapolation
        slope, intercept, r_value, _, std_err = stats.linregress(years, extents)

        future_years = list(range(max(years) + 1, max(years) + self.forecast_years + 1))
        linear_forecast = [slope * year + intercept for year in future_years]

        # Confidence intervals (basic approach)
        forecast_std = std_err * np.sqrt(
            1
            + 1 / len(years)
            + (np.array(future_years) - np.mean(years)) ** 2
            / np.sum((np.array(years) - np.mean(years)) ** 2)
        )

        confidence_intervals = [
            (pred - 1.96 * std, pred + 1.96 * std)
            for pred, std in zip(linear_forecast, forecast_std, strict=False)
        ]

        return {
            "method": "linear_regression",
            "forecast_years": future_years,
            "predicted_extents": linear_forecast,
            "confidence_intervals_95": confidence_intervals,
            "model_r_squared": r_value**2,
            "forecast_uncertainty": np.mean(forecast_std),
        }

    def _assess_trend_risks(
        self, trend_results: dict, forecast_results: dict
    ) -> dict[str, Any]:
        """Assess risks based on trend analysis."""
        risk_level = "low"
        risk_factors = []

        # Check trend direction and significance
        if "linear_trend" in trend_results:
            slope = trend_results["linear_trend"]["slope"]
            p_value = trend_results["linear_trend"]["p_value"]

            if slope < -0.5 and p_value < 0.05:
                risk_level = "high"
                risk_factors.append("Significant declining trend")
            elif slope < -0.1:
                risk_level = "medium"
                risk_factors.append("Mild declining trend")

        # Check forecast predictions
        if forecast_results and "predicted_extents" in forecast_results:
            future_extents = forecast_results["predicted_extents"]
            if any(extent < 0 for extent in future_extents[:5]):  # Next 5 years
                risk_level = "high"
                risk_factors.append("Forecast predicts potential extinction")

        # Check variability
        if "rate_of_change" in trend_results:
            rate_std = trend_results["rate_of_change"]["rate_std"]
            if rate_std > 1.0:  # High variability
                if risk_level == "low":
                    risk_level = "medium"
                risk_factors.append("High variability in extent changes")

        return {
            "overall_risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommendations": self._generate_risk_recommendations(
                risk_level, risk_factors
            ),
        }

    def _generate_risk_recommendations(
        self, risk_level: str, risk_factors: list[str]
    ) -> list[str]:
        """Generate management recommendations based on risk assessment."""
        recommendations = []

        if risk_level == "high":
            recommendations.extend(
                [
                    "Implement immediate conservation measures",
                    "Increase monitoring frequency",
                    "Investigate environmental stressors",
                    "Consider restoration interventions",
                ]
            )
        elif risk_level == "medium":
            recommendations.extend(
                [
                    "Enhanced monitoring program",
                    "Assess environmental conditions",
                    "Develop contingency plans",
                ]
            )
        else:
            recommendations.extend(
                ["Continue regular monitoring", "Maintain current conservation status"]
            )

        return recommendations


class HistoricalBaselineAnalysis:
    """
    Main framework for historical baseline analysis of kelp extent.

    This class integrates historical data digitization, change detection,
    and temporal trend analysis following UVic methodology for kelp mapping.
    """

    def __init__(self):
        self.change_analyzer = ChangeDetectionAnalyzer()
        self.trend_analyzer = TemporalTrendAnalyzer()
        self.historical_sites: dict[str, HistoricalSite] = {}
        self.historical_datasets: dict[str, HistoricalDataset] = {}

    def create_historical_site(
        self,
        name: str,
        latitude: float,
        longitude: float,
        region: str,
        historical_period: tuple[int, int],
        data_sources: list[str],
        species: list[str],
        **kwargs,
    ) -> HistoricalSite:
        """Create and register a historical monitoring site."""
        site = HistoricalSite(
            name=name,
            latitude=latitude,
            longitude=longitude,
            region=region,
            historical_period=historical_period,
            data_sources=data_sources,
            species=species,
            **kwargs,
        )

        self.historical_sites[name] = site
        logger.info(f"Created historical site: {name}")
        return site

    def digitize_historical_data(
        self,
        site_name: str,
        chart_data: dict[int, dict[str, float | str]],
        quality_control_params: dict[str, Any] | None = None,
    ) -> HistoricalDataset:
        """
        Digitize historical chart data for a site.

        Args:
            site_name: Name of the historical site
            chart_data: Year -> {extent, confidence, source, notes}
            quality_control_params: Quality control parameters

        Returns:
            Validated historical dataset
        """
        if site_name not in self.historical_sites:
            raise ValueError(f"Site {site_name} not found")

        site = self.historical_sites[site_name]

        # Apply quality control
        qc_params = quality_control_params or {}
        processed_data = self._apply_quality_control(chart_data, qc_params)

        # Calculate baseline extent (typically early period average)
        baseline_years = [
            year
            for year in processed_data.keys()
            if site.historical_period[0] <= year <= site.historical_period[0] + 10
        ]
        baseline_extent = np.mean(
            [processed_data[year]["extent"] for year in baseline_years]
        )

        # Calculate confidence intervals
        confidence_intervals = {}
        for year, data in processed_data.items():
            conf = data.get("confidence", 0.8)
            extent = data["extent"]
            margin = extent * (1 - conf) * 0.5  # Simple confidence margin
            confidence_intervals[year] = (extent - margin, extent + margin)

        # Calculate data quality metrics
        quality_metrics = self._calculate_data_quality_metrics(processed_data, site)

        dataset = HistoricalDataset(
            site=site,
            temporal_data=processed_data,
            baseline_extent=baseline_extent,
            confidence_intervals=confidence_intervals,
            data_quality_metrics=quality_metrics,
        )

        self.historical_datasets[site_name] = dataset
        logger.info(f"Digitized historical data for site: {site_name}")
        return dataset

    def _apply_quality_control(
        self, chart_data: dict[int, dict[str, float | str]], qc_params: dict[str, Any]
    ) -> dict[int, dict[str, float]]:
        """Apply quality control procedures to historical chart data."""
        processed = {}

        min_confidence = qc_params.get("min_confidence", 0.5)
        max_extent_change = qc_params.get("max_extent_change", 5.0)  # 5x change filter

        years = sorted(chart_data.keys())
        last_processed_year = None

        for year in years:
            data = chart_data[year].copy()

            # Convert to numeric if needed
            if isinstance(data["extent"], str):
                try:
                    data["extent"] = float(data["extent"])
                except ValueError:
                    logger.warning(f"Skipping year {year}: invalid extent data")
                    continue

            # Confidence filter
            confidence = data.get("confidence", 0.8)
            if confidence < min_confidence:
                logger.warning(f"Low confidence data for year {year}: {confidence}")
                continue

            # Outlier detection (basic)
            if last_processed_year is not None:
                prev_extent = processed[last_processed_year]["extent"]
                change_ratio = data["extent"] / prev_extent if prev_extent > 0 else 1
                if (
                    change_ratio > max_extent_change
                    or change_ratio < 1 / max_extent_change
                ):
                    logger.warning(
                        f"Large extent change detected for year {year}: {change_ratio:.2f}x"
                    )
                    # Could flag for manual review rather than exclude

            processed[year] = {
                "extent": float(data["extent"]),
                "confidence": float(confidence),
                "source": data.get("source", "unknown"),
                "notes": data.get("notes", ""),
            }
            last_processed_year = year

        return processed

    def _calculate_data_quality_metrics(
        self, processed_data: dict[int, dict[str, float]], site: HistoricalSite
    ) -> dict[str, float]:
        """Calculate comprehensive data quality metrics."""
        years = list(processed_data.keys())

        # Temporal coverage
        expected_years = site.historical_period[1] - site.historical_period[0] + 1
        actual_years = len(years)
        temporal_coverage = actual_years / expected_years

        # Confidence statistics
        confidences = [data["confidence"] for data in processed_data.values()]
        mean_confidence = np.mean(confidences)
        min_confidence = np.min(confidences)

        # Temporal regularity (gaps)
        if len(years) > 1:
            gaps = np.diff(sorted(years))
            max_gap = np.max(gaps)
            mean_gap = np.mean(gaps)
        else:
            max_gap = mean_gap = 0

        return {
            "temporal_coverage": temporal_coverage,
            "mean_confidence": mean_confidence,
            "min_confidence": min_confidence,
            "max_temporal_gap": max_gap,
            "mean_temporal_gap": mean_gap,
            "data_completeness": len(
                [d for d in processed_data.values() if d["extent"] > 0]
            )
            / len(processed_data),
        }

    def perform_comprehensive_analysis(
        self,
        site_name: str,
        current_extent: float,
        current_year: int = 2024,
        analysis_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Perform comprehensive historical baseline analysis.

        Args:
            site_name: Name of the site to analyze
            current_extent: Current kelp extent measurement
            current_year: Year of current measurement
            analysis_options: Analysis configuration options

        Returns:
            Comprehensive analysis results
        """
        if site_name not in self.historical_datasets:
            raise ValueError(f"No historical dataset found for site: {site_name}")

        dataset = self.historical_datasets[site_name]
        options = analysis_options or {}

        # Extract historical data
        historical_data = {
            year: data["extent"] for year, data in dataset.temporal_data.items()
        }
        current_data = {current_year: current_extent}

        # Change detection analysis
        change_results = self.change_analyzer.detect_significant_changes(
            historical_data,
            current_data,
            method=options.get("change_method", "mann_kendall"),
        )

        # Change pattern analysis
        pattern_results = self.change_analyzer.analyze_change_patterns(
            dataset, current_extent, current_year
        )

        # Temporal trend analysis
        trend_results = self.trend_analyzer.analyze_temporal_trends(
            dataset, include_forecast=options.get("include_forecast", True)
        )

        # Site metadata
        site_info = {
            "name": dataset.site.name,
            "region": dataset.site.region,
            "species": dataset.site.species,
            "historical_period": dataset.site.historical_period,
            "data_sources": dataset.site.data_sources,
            "coordinates": [dataset.site.latitude, dataset.site.longitude],
        }

        return {
            "site_information": site_info,
            "change_detection": change_results,
            "change_patterns": pattern_results,
            "temporal_trends": trend_results,
            "data_quality": dataset.data_quality_metrics,
            "analysis_metadata": {
                "analysis_date": datetime.now().isoformat(),
                "current_measurement": {"year": current_year, "extent": current_extent},
                "baseline_extent": dataset.baseline_extent,
                "total_historical_years": len(dataset.temporal_data),
            },
        }

    def generate_comparison_report(
        self, site_names: list[str], output_format: str = "dict"
    ) -> dict[str, Any] | str:
        """
        Generate comparative analysis report across multiple sites.

        Args:
            site_names: List of site names to compare
            output_format: Output format ('dict', 'json', 'markdown')

        Returns:
            Comparative analysis report
        """
        if not all(name in self.historical_datasets for name in site_names):
            missing = [
                name for name in site_names if name not in self.historical_datasets
            ]
            raise ValueError(f"Missing datasets for sites: {missing}")

        comparative_results = {}

        for site_name in site_names:
            dataset = self.historical_datasets[site_name]

            # Basic site statistics
            temporal_data = dataset.temporal_data
            extents = [data["extent"] for data in temporal_data.values()]
            years = list(temporal_data.keys())

            site_stats = {
                "baseline_extent": dataset.baseline_extent,
                "mean_extent": np.mean(extents),
                "extent_range": [np.min(extents), np.max(extents)],
                "temporal_span": [min(years), max(years)],
                "data_years": len(years),
                "region": dataset.site.region,
                "species": dataset.site.species,
                "data_quality_score": np.mean(
                    list(dataset.data_quality_metrics.values())
                ),
            }

            comparative_results[site_name] = site_stats

        # Cross-site analysis
        all_baselines = [
            results["baseline_extent"] for results in comparative_results.values()
        ]
        all_means = [results["mean_extent"] for results in comparative_results.values()]

        summary_stats = {
            "total_sites": len(site_names),
            "baseline_extent_range": [np.min(all_baselines), np.max(all_baselines)],
            "mean_extent_range": [np.min(all_means), np.max(all_means)],
            "regional_diversity": len(
                set(results["region"] for results in comparative_results.values())
            ),
            "species_diversity": len(
                set(
                    tuple(results["species"])
                    for results in comparative_results.values()
                )
            ),
        }

        final_report = {
            "summary_statistics": summary_stats,
            "site_comparisons": comparative_results,
            "report_metadata": {
                "generated_date": datetime.now().isoformat(),
                "analysis_type": "historical_baseline_comparison",
                "sites_analyzed": site_names,
            },
        }

        if output_format == "json":
            return json.dumps(final_report, indent=2)
        elif output_format == "markdown":
            return self._format_markdown_report(final_report)
        else:
            return final_report

    def _format_markdown_report(self, report: dict[str, Any]) -> str:
        """Format comparison report as markdown."""
        md_lines = [
            "# Historical Baseline Analysis - Comparative Report\n",
            f"**Generated:** {report['report_metadata']['generated_date']}\n",
            f"**Sites Analyzed:** {len(report['report_metadata']['sites_analyzed'])}\n",
            "\n## Summary Statistics\n",
        ]

        summary = report["summary_statistics"]
        md_lines.extend(
            [
                f"- **Total Sites:** {summary['total_sites']}",
                f"- **Baseline Extent Range:** {summary['baseline_extent_range'][0]:.1f} - {summary['baseline_extent_range'][1]:.1f} hectares",
                f"- **Regional Diversity:** {summary['regional_diversity']} regions",
                f"- **Species Diversity:** {summary['species_diversity']} species groups",
                "\n## Site Details\n",
            ]
        )

        for site_name, stats in report["site_comparisons"].items():
            md_lines.extend(
                [
                    f"\n### {site_name}",
                    f"- **Region:** {stats['region']}",
                    f"- **Species:** {', '.join(stats['species'])}",
                    f"- **Baseline Extent:** {stats['baseline_extent']:.1f} hectares",
                    f"- **Mean Historical Extent:** {stats['mean_extent']:.1f} hectares",
                    f"- **Temporal Span:** {stats['temporal_span'][0]} - {stats['temporal_span'][1]}",
                    f"- **Data Quality Score:** {stats['data_quality_score']:.2f}",
                ]
            )

        return "\n".join(md_lines)

    def _convert_numpy_types(self, obj):
        """Convert NumPy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj

    def export_results(
        self,
        site_name: str,
        output_path: str | Path,
        include_visualizations: bool = False,
    ) -> None:
        """
        Export analysis results to file.

        Args:
            site_name: Site name to export
            output_path: Output file path
            include_visualizations: Whether to generate plots
        """
        if site_name not in self.historical_datasets:
            raise ValueError(f"No dataset found for site: {site_name}")

        dataset = self.historical_datasets[site_name]
        output_path = Path(output_path)

        # Prepare export data
        export_data = {
            "site_metadata": {
                "name": dataset.site.name,
                "location": [dataset.site.latitude, dataset.site.longitude],
                "region": dataset.site.region,
                "species": dataset.site.species,
                "historical_period": dataset.site.historical_period,
                "data_sources": dataset.site.data_sources,
            },
            "temporal_data": dataset.temporal_data,
            "baseline_extent": dataset.baseline_extent,
            "confidence_intervals": dataset.confidence_intervals,
            "data_quality_metrics": dataset.data_quality_metrics,
            "export_metadata": {
                "export_date": datetime.now().isoformat(),
                "software_version": "kelpie_carbon",
            },
        }

        # Convert NumPy types to native Python types for JSON serialization
        export_data = self._convert_numpy_types(export_data)

        # Write JSON file
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(export_data, f, indent=2)

        # Generate visualizations if requested
        if include_visualizations:
            self._create_visualization_plots(dataset, output_path.parent)

        logger.info(f"Exported results for {site_name} to {output_path}")

    def _create_visualization_plots(
        self, dataset: HistoricalDataset, output_dir: Path
    ) -> None:
        """Create visualization plots for the dataset."""
        # Prepare data
        years = sorted(dataset.temporal_data.keys())
        extents = [dataset.temporal_data[year]["extent"] for year in years]
        confidences = [dataset.temporal_data[year]["confidence"] for year in years]

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Extent time series
        ax1.plot(years, extents, "b-o", linewidth=2, markersize=4)
        ax1.axhline(
            y=dataset.baseline_extent,
            color="r",
            linestyle="--",
            alpha=0.7,
            label="Baseline",
        )
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Kelp Extent (hectares)")
        ax1.set_title("Historical Kelp Extent")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Confidence levels
        ax2.plot(years, confidences, "g-s", linewidth=2, markersize=4)
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Data Confidence")
        ax2.set_title("Data Quality Over Time")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # Extent distribution
        ax3.hist(extents, bins=min(10, len(extents) // 2), alpha=0.7, edgecolor="black")
        ax3.axvline(
            x=dataset.baseline_extent,
            color="r",
            linestyle="--",
            alpha=0.7,
            label="Baseline",
        )
        ax3.set_xlabel("Kelp Extent (hectares)")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Extent Distribution")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Rate of change
        if len(years) > 1:
            rates = np.array(extents[1:]) - np.array(extents[:-1])
            rate_years = years[1:]
            ax4.bar(rate_years, rates, alpha=0.7, edgecolor="black")
            ax4.axhline(y=0, color="r", linestyle="-", alpha=0.7)
            ax4.set_xlabel("Year")
            ax4.set_ylabel("Annual Change (hectares)")
            ax4.set_title("Year-over-Year Changes")
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{dataset.site.name}_historical_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        logger.info(f"Created visualization plots for {dataset.site.name}")


# Factory functions for easy creation
def create_uvic_historical_sites() -> dict[str, HistoricalSite]:
    """Create historical sites following UVic methodology."""
    analyzer = HistoricalBaselineAnalysis()

    # Create key historical sites from UVic research
    sites = {}

    # Broughton Archipelago - Primary UVic site
    sites["broughton"] = analyzer.create_historical_site(
        name="Broughton Archipelago",
        latitude=50.0833,
        longitude=-126.1667,
        region="British Columbia",
        historical_period=(1858, 1956),
        data_sources=[
            "British Admiralty Charts",
            "Canadian Hydrographic Service",
            "UVic SKEMA",
        ],
        species=["Nereocystis luetkeana", "Macrocystis pyrifera"],
        chart_references=["Chart 1917 (1858)", "Chart 3576 (1956)"],
        digitization_quality="high",
        notes="Primary UVic SKEMA validation site with extensive historical documentation",
    )

    # Saanich Inlet - Multi-species site
    sites["saanich"] = analyzer.create_historical_site(
        name="Saanich Inlet",
        latitude=48.5830,
        longitude=-123.5000,
        region="British Columbia",
        historical_period=(1859, 1952),
        data_sources=["British Admiralty Charts", "UVic Marine Science"],
        species=["Nereocystis luetkeana", "Macrocystis pyrifera", "Mixed species"],
        chart_references=["Chart 1912 (1859)", "Chart 3313 (1952)"],
        digitization_quality="high",
        notes="Multi-species validation site near UVic campus",
    )

    # Monterey Bay - California comparison site
    sites["monterey"] = analyzer.create_historical_site(
        name="Monterey Bay",
        latitude=36.8000,
        longitude=-121.9000,
        region="California",
        historical_period=(1851, 1950),
        data_sources=["US Coast Survey", "NOAA Charts", "MBARI"],
        species=["Macrocystis pyrifera"],
        chart_references=["T-Sheet 1851", "Chart 18685"],
        digitization_quality="medium",
        notes="Historic giant kelp forests with MBARI validation data",
    )

    return sites


def create_sample_historical_dataset() -> HistoricalDataset:
    """Create a sample historical dataset for testing."""
    analyzer = HistoricalBaselineAnalysis()

    # Create Broughton site
    site = analyzer.create_historical_site(
        name="Sample Broughton Site",
        latitude=50.0833,
        longitude=-126.1667,
        region="British Columbia",
        historical_period=(1858, 1956),
        data_sources=["Sample Charts"],
        species=["Nereocystis luetkeana"],
    )

    # Create sample temporal data with realistic variation
    np.random.seed(42)  # For reproducible results
    base_extent = 150.0  # hectares
    years = list(range(1858, 1957, 5))  # Every 5 years

    temporal_data = {}
    for i, year in enumerate(years):
        # Add realistic trend and variation
        trend = -0.3 * i  # Slight decline over time
        noise = np.random.normal(0, 10)  # Random variation
        extent = max(0, base_extent + trend + noise)

        temporal_data[year] = {
            "extent": extent,
            "confidence": np.random.uniform(0.7, 0.95),
            "source": f"Chart_{year}",
            "notes": f"Sample data for year {year}",
        }

    return analyzer.digitize_historical_data("Sample Broughton Site", temporal_data)
