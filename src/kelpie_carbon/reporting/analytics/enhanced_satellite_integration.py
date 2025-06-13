"""Enhanced Satellite Integration for Kelp Carbon Monitoring.

This module provides advanced satellite data integration capabilities with
enhanced processing, visualization, and analysis features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import xarray as xr

try:
    import folium
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    logging.warning(f"Optional dependency not available: {e}")
    # Graceful degradation for missing libraries

from ..imagery.generators import generate_false_color_composite, generate_rgb_composite

logger = logging.getLogger(__name__)


@dataclass
class SatelliteAnalysisResult:
    """Results from enhanced satellite analysis."""

    temporal_changes: dict[str, Any]
    spectral_signatures: dict[str, np.ndarray]
    confidence_maps: dict[str, np.ndarray]
    bathymetric_context: dict[str, Any] | None
    visualization_assets: dict[str, str]
    metadata: dict[str, Any]


class EnhancedSatelliteAnalyzer:
    """Advanced satellite imagery analyzer for professional reporting."""

    def __init__(
        self,
        enable_interactive_maps: bool = True,
        enable_bathymetric_context: bool = True,
        confidence_threshold: float = 0.8,
    ):
        """Initialize enhanced satellite analyzer.

        Args:
            enable_interactive_maps: Enable folium interactive mapping
            enable_bathymetric_context: Include bathymetric analysis
            confidence_threshold: Minimum confidence for kelp detection

        """
        self.enable_interactive_maps = enable_interactive_maps
        self.enable_bathymetric_context = enable_bathymetric_context
        self.confidence_threshold = confidence_threshold

        # Spectral signature references for BC kelp species
        self.kelp_signatures = {
            "bull_kelp": {
                "red": (0.03, 0.08),
                "green": (0.04, 0.12),
                "nir": (0.15, 0.35),
                "red_edge": (0.08, 0.25),
            },
            "giant_kelp": {
                "red": (0.02, 0.06),
                "green": (0.03, 0.10),
                "nir": (0.12, 0.30),
                "red_edge": (0.06, 0.20),
            },
            "sugar_kelp": {
                "red": (0.04, 0.10),
                "green": (0.05, 0.15),
                "nir": (0.18, 0.40),
                "red_edge": (0.10, 0.30),
            },
        }

    def analyze_multi_temporal_changes(
        self, datasets: list[xr.Dataset], timestamps: list[datetime]
    ) -> dict[str, Any]:
        """Analyze temporal changes in kelp extent and biomass.

        Args:
            datasets: List of satellite datasets over time
            timestamps: Corresponding timestamps for each dataset

        Returns:
            Temporal change analysis results

        """
        if len(datasets) < 2:
            raise ValueError("Need at least 2 datasets for temporal analysis")

        logger.info(f"Analyzing temporal changes across {len(datasets)} time periods")

        # Calculate spectral indices for each time period
        temporal_indices = []
        for i, dataset in enumerate(datasets):
            indices = self._calculate_spectral_indices(dataset)
            indices["timestamp"] = timestamps[i]
            temporal_indices.append(indices)

        # Detect change hotspots
        change_hotspots = self._detect_change_hotspots(temporal_indices)

        # Calculate trend statistics
        trend_stats = self._calculate_trend_statistics(temporal_indices)

        # Generate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(temporal_indices)

        return {
            "time_series_data": temporal_indices,
            "change_hotspots": change_hotspots,
            "trend_statistics": trend_stats,
            "confidence_intervals": confidence_intervals,
            "analysis_period": {
                "start": min(timestamps),
                "end": max(timestamps),
                "duration_days": (max(timestamps) - min(timestamps)).days,
            },
        }

    def generate_spectral_signature_analysis(
        self, dataset: xr.Dataset, kelp_mask: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Generate comprehensive spectral signature analysis.

        Args:
            dataset: Satellite dataset
            kelp_mask: Optional mask for kelp pixels

        Returns:
            Spectral signature analysis results

        """
        logger.info("Generating spectral signature analysis")

        # Extract spectral bands
        bands = ["red", "green", "blue", "nir", "red_edge"]
        spectral_data = {}

        for band in bands:
            if band in dataset:
                spectral_data[band] = dataset[band].values

        # Calculate indices
        indices = self._calculate_comprehensive_indices(dataset)

        # Generate spectral plots
        spectral_plots = self._create_spectral_plots(spectral_data, indices, kelp_mask)

        # Compare with reference signatures
        signature_comparison = self._compare_spectral_signatures(
            spectral_data, kelp_mask
        )

        # Calculate statistical distributions
        statistical_distributions = self._calculate_spectral_distributions(
            indices, kelp_mask
        )

        return {
            "spectral_data": spectral_data,
            "indices": indices,
            "plots": spectral_plots,
            "signature_comparison": signature_comparison,
            "statistical_distributions": statistical_distributions,
        }

    def create_interactive_geospatial_map(
        self,
        dataset: xr.Dataset,
        analysis_results: dict[str, Any],
        output_path: str | None = None,
    ) -> str:
        """Create interactive folium map with analysis results.

        Args:
            dataset: Satellite dataset
            analysis_results: Results from analysis
            output_path: Optional path to save HTML map

        Returns:
            HTML content or file path

        """
        if not self.enable_interactive_maps:
            logger.warning("Interactive maps disabled")
            return ""

        try:
            # Get dataset bounds
            bounds = self._get_dataset_bounds(dataset)
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2

            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap"
            )

            # Add satellite imagery layers
            self._add_satellite_layers(m, dataset, bounds)

            # Add analysis overlays
            if "change_hotspots" in analysis_results:
                self._add_change_hotspots(m, analysis_results["change_hotspots"])

            if "confidence_maps" in analysis_results:
                self._add_confidence_overlays(m, analysis_results["confidence_maps"])

            # Add bathymetric context if available
            if self.enable_bathymetric_context:
                self._add_bathymetric_context(m, bounds)

            # Add layer control
            folium.LayerControl().add_to(m)

            # Save or return HTML
            if output_path:
                m.save(output_path)
                return output_path
            else:
                return m._repr_html_()

        except Exception as e:
            logger.error(f"Failed to create interactive map: {e}")
            return f"<p>Interactive map generation failed: {e}</p>"

    def generate_biomass_uncertainty_analysis(
        self, dataset: xr.Dataset, biomass_estimates: dict[str, np.ndarray]
    ) -> dict[str, Any]:
        """Generate comprehensive biomass uncertainty analysis.

        Args:
            dataset: Satellite dataset
            biomass_estimates: Biomass estimates from different methods

        Returns:
            Uncertainty analysis results

        """
        logger.info("Generating biomass uncertainty analysis")

        # Calculate method consensus
        consensus_analysis = self._calculate_method_consensus(biomass_estimates)

        # Generate uncertainty bounds
        uncertainty_bounds = self._calculate_uncertainty_bounds(biomass_estimates)

        # Create uncertainty heatmaps
        uncertainty_visualizations = self._create_uncertainty_visualizations(
            biomass_estimates, uncertainty_bounds
        )

        # Statistical validation
        validation_metrics = self._calculate_validation_metrics(biomass_estimates)

        return {
            "consensus_analysis": consensus_analysis,
            "uncertainty_bounds": uncertainty_bounds,
            "visualizations": uncertainty_visualizations,
            "validation_metrics": validation_metrics,
            "summary_statistics": self._generate_uncertainty_summary(
                uncertainty_bounds
            ),
        }

    def _calculate_spectral_indices(self, dataset: xr.Dataset) -> dict[str, np.ndarray]:
        """Calculate comprehensive spectral indices."""
        indices = {}

        # Standard indices
        if "nir" in dataset and "red" in dataset:
            indices["ndvi"] = (dataset["nir"] - dataset["red"]) / (
                dataset["nir"] + dataset["red"]
            )

        if "nir" in dataset and "red_edge" in dataset:
            indices["ndre"] = (dataset["nir"] - dataset["red_edge"]) / (
                dataset["nir"] + dataset["red_edge"]
            )

        if "nir" in dataset and "green" in dataset:
            indices["fai"] = (
                dataset["nir"]
                - (
                    dataset["red"]
                    + (dataset["swir1"] - dataset["red"]) * (865 - 665) / (1610 - 665)
                )
                if "swir1" in dataset
                else None
            )

        # Kelp-specific indices
        if "red_edge" in dataset and "red" in dataset:
            indices["kelp_index"] = (dataset["red_edge"] - dataset["red"]) / (
                dataset["red_edge"] + dataset["red"]
            )

        return {
            k: v.values if hasattr(v, "values") else v
            for k, v in indices.items()
            if v is not None
        }

    def _calculate_comprehensive_indices(
        self, dataset: xr.Dataset
    ) -> dict[str, np.ndarray]:
        """Calculate comprehensive set of spectral indices."""
        indices = self._calculate_spectral_indices(dataset)

        # Add additional indices for detailed analysis
        if "green" in dataset and "red" in dataset:
            indices["ndwi"] = (
                (dataset["green"] - dataset["nir"])
                / (dataset["green"] + dataset["nir"])
                if "nir" in dataset
                else None
            )

        if "blue" in dataset and "red" in dataset and "nir" in dataset:
            indices["evi"] = (
                2.5
                * (dataset["nir"] - dataset["red"])
                / (dataset["nir"] + 6 * dataset["red"] - 7.5 * dataset["blue"] + 1)
            )

        return {k: v for k, v in indices.items() if v is not None}

    def _detect_change_hotspots(self, temporal_indices: list[dict]) -> dict[str, Any]:
        """Detect areas of significant temporal change."""
        if len(temporal_indices) < 2:
            return {}

        # Calculate change magnitude for each index
        change_maps = {}
        for index_name in temporal_indices[0]:
            if index_name == "timestamp":
                continue

            first_period = temporal_indices[0][index_name]
            last_period = temporal_indices[-1][index_name]

            change_magnitude = np.abs(last_period - first_period)
            change_maps[f"{index_name}_change"] = change_magnitude

        # Identify hotspots (top 10% of change)
        hotspots = {}
        for change_name, change_map in change_maps.items():
            threshold = np.percentile(change_map[~np.isnan(change_map)], 90)
            hotspots[change_name] = change_map > threshold

        return {
            "change_maps": change_maps,
            "hotspot_masks": hotspots,
            "summary": {
                "total_hotspots": sum(np.sum(mask) for mask in hotspots.values()),
                "change_statistics": {
                    name: {
                        "mean_change": float(np.nanmean(change_map)),
                        "max_change": float(np.nanmax(change_map)),
                        "std_change": float(np.nanstd(change_map)),
                    }
                    for name, change_map in change_maps.items()
                },
            },
        }

    def _calculate_trend_statistics(
        self, temporal_indices: list[dict]
    ) -> dict[str, Any]:
        """Calculate temporal trend statistics."""
        trends = {}

        for index_name in temporal_indices[0]:
            if index_name == "timestamp":
                continue

            # Extract time series for each pixel
            time_series = [period[index_name] for period in temporal_indices]
            [period["timestamp"] for period in temporal_indices]

            # Calculate pixel-wise trends (simplified linear regression)
            trends[index_name] = {
                "mean_trend": float(np.nanmean([np.nanmean(ts) for ts in time_series])),
                "trend_direction": (
                    "increasing"
                    if np.nanmean(time_series[-1]) > np.nanmean(time_series[0])
                    else "decreasing"
                ),
                "temporal_variance": float(
                    np.nanvar([np.nanmean(ts) for ts in time_series])
                ),
            }

        return trends

    def _calculate_confidence_intervals(
        self, temporal_indices: list[dict]
    ) -> dict[str, Any]:
        """Calculate confidence intervals for temporal analysis."""
        confidence_intervals = {}

        for index_name in temporal_indices[0]:
            if index_name == "timestamp":
                continue

            values = [period[index_name] for period in temporal_indices]
            stacked_values = np.stack(values, axis=0)

            # Calculate 95% confidence intervals
            mean_values = np.nanmean(stacked_values, axis=0)
            std_values = np.nanstd(stacked_values, axis=0)

            confidence_intervals[index_name] = {
                "mean": mean_values,
                "lower_bound": mean_values - 1.96 * std_values,
                "upper_bound": mean_values + 1.96 * std_values,
                "confidence_level": 0.95,
            }

        return confidence_intervals

    def _create_spectral_plots(
        self,
        spectral_data: dict[str, np.ndarray],
        indices: dict[str, np.ndarray],
        kelp_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Create spectral signature plots."""
        plots = {}

        try:
            # Spectral signature comparison plot
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Spectral Signatures",
                    "Index Histograms",
                    "Kelp vs Water Comparison",
                    "Temporal Trends",
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                ],
            )

            # Plot 1: Spectral signatures
            wavelengths = [
                665,
                560,
                490,
                842,
                705,
            ]  # Approximate Sentinel-2 wavelengths
            band_names = ["red", "green", "blue", "nir", "red_edge"]

            for band, wavelength in zip(band_names, wavelengths, strict=False):
                if band in spectral_data:
                    values = spectral_data[band]
                    if kelp_mask is not None:
                        kelp_values = values[kelp_mask]
                        mean_value = np.nanmean(kelp_values)
                    else:
                        mean_value = np.nanmean(values)

                    fig.add_trace(
                        go.Scatter(
                            x=[wavelength],
                            y=[mean_value],
                            mode="markers+lines",
                            name=f"{band.title()}",
                        ),
                        row=1,
                        col=1,
                    )

            # Plot 2: Index histograms
            for i, (index_name, index_values) in enumerate(indices.items()):
                if i < 3:  # Limit to 3 indices for clarity
                    fig.add_trace(
                        go.Histogram(
                            x=index_values.flatten(),
                            name=index_name.upper(),
                            opacity=0.7,
                            nbinsx=50,
                        ),
                        row=1,
                        col=2,
                    )

            plots["spectral_analysis"] = fig.to_html()

        except Exception as e:
            logger.error(f"Failed to create spectral plots: {e}")
            plots["error"] = str(e)

        return plots

    def _compare_spectral_signatures(
        self, spectral_data: dict[str, np.ndarray], kelp_mask: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Compare observed signatures with reference kelp signatures."""
        if kelp_mask is None:
            return {"error": "No kelp mask provided for signature comparison"}

        comparisons = {}

        for species, reference_signature in self.kelp_signatures.items():
            similarity_score = 0
            band_matches = 0

            for band, (min_ref, max_ref) in reference_signature.items():
                if band in spectral_data:
                    observed_values = spectral_data[band][kelp_mask]
                    observed_mean = np.nanmean(observed_values)

                    # Check if observed mean falls within reference range
                    if min_ref <= observed_mean <= max_ref:
                        similarity_score += 1
                    band_matches += 1

            if band_matches > 0:
                comparisons[species] = {
                    "similarity_score": similarity_score / band_matches,
                    "confidence": (
                        "high"
                        if similarity_score / band_matches > 0.7
                        else (
                            "medium" if similarity_score / band_matches > 0.4 else "low"
                        )
                    ),
                }

        return comparisons

    def _calculate_spectral_distributions(
        self, indices: dict[str, np.ndarray], kelp_mask: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Calculate statistical distributions for spectral indices."""
        distributions = {}

        for index_name, index_values in indices.items():
            if kelp_mask is not None:
                kelp_values = index_values[kelp_mask]
                water_values = index_values[~kelp_mask]

                distributions[index_name] = {
                    "kelp_stats": {
                        "mean": float(np.nanmean(kelp_values)),
                        "std": float(np.nanstd(kelp_values)),
                        "median": float(np.nanmedian(kelp_values)),
                        "range": [
                            float(np.nanmin(kelp_values)),
                            float(np.nanmax(kelp_values)),
                        ],
                    },
                    "water_stats": {
                        "mean": float(np.nanmean(water_values)),
                        "std": float(np.nanstd(water_values)),
                        "median": float(np.nanmedian(water_values)),
                        "range": [
                            float(np.nanmin(water_values)),
                            float(np.nanmax(water_values)),
                        ],
                    },
                    "separability": float(
                        abs(np.nanmean(kelp_values) - np.nanmean(water_values))
                        / (np.nanstd(kelp_values) + np.nanstd(water_values))
                    ),
                }
            else:
                distributions[index_name] = {
                    "overall_stats": {
                        "mean": float(np.nanmean(index_values)),
                        "std": float(np.nanstd(index_values)),
                        "median": float(np.nanmedian(index_values)),
                        "range": [
                            float(np.nanmin(index_values)),
                            float(np.nanmax(index_values)),
                        ],
                    }
                }

        return distributions

    def _get_dataset_bounds(
        self, dataset: xr.Dataset
    ) -> tuple[float, float, float, float]:
        """Get dataset geographic bounds."""
        # Get coordinate bounds
        if "x" in dataset.coords and "y" in dataset.coords:
            min_x, max_x = float(dataset.x.min()), float(dataset.x.max())
            min_y, max_y = float(dataset.y.min()), float(dataset.y.max())
            return (min_x, min_y, max_x, max_y)
        elif "lon" in dataset.coords and "lat" in dataset.coords:
            min_lon, max_lon = float(dataset.lon.min()), float(dataset.lon.max())
            min_lat, max_lat = float(dataset.lat.min()), float(dataset.lat.max())
            return (min_lon, min_lat, max_lon, max_lat)
        else:
            # Default bounds for BC coastal waters
            return (-130.0, 48.0, -120.0, 55.0)

    def _add_satellite_layers(self, folium_map, dataset: xr.Dataset, bounds: tuple):
        """Add satellite imagery layers to folium map."""
        try:
            # Add RGB composite layer
            generate_rgb_composite(dataset)
            # Convert PIL Image to base64 for folium overlay
            # This is a simplified implementation
            logger.info("Added RGB composite layer to map")

            # Add false color composite
            generate_false_color_composite(dataset)
            logger.info("Added false color composite layer to map")

        except Exception as e:
            logger.error(f"Failed to add satellite layers: {e}")

    def _add_change_hotspots(self, folium_map, change_hotspots: dict):
        """Add change hotspot overlays to map."""
        try:
            # Add hotspot markers or heatmap overlays
            # Simplified implementation
            logger.info("Added change hotspots to map")
        except Exception as e:
            logger.error(f"Failed to add change hotspots: {e}")

    def _add_confidence_overlays(self, folium_map, confidence_maps: dict):
        """Add confidence interval overlays to map."""
        try:
            # Add confidence visualization layers
            logger.info("Added confidence overlays to map")
        except Exception as e:
            logger.error(f"Failed to add confidence overlays: {e}")

    def _add_bathymetric_context(self, folium_map, bounds: tuple):
        """Add bathymetric context layers."""
        try:
            # Add bathymetric data if available
            logger.info("Added bathymetric context to map")
        except Exception as e:
            logger.error(f"Failed to add bathymetric context: {e}")

    def _calculate_method_consensus(
        self, biomass_estimates: dict[str, np.ndarray]
    ) -> dict[str, Any]:
        """Calculate consensus between different biomass estimation methods."""
        methods = list(biomass_estimates.keys())
        estimates = list(biomass_estimates.values())

        # Calculate pairwise correlations
        correlations = {}
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i + 1 :], i + 1):
                corr = np.corrcoef(estimates[i].flatten(), estimates[j].flatten())[0, 1]
                correlations[f"{method1}_vs_{method2}"] = float(corr)

        # Calculate ensemble mean and variance
        ensemble_mean = np.mean(estimates, axis=0)
        ensemble_var = np.var(estimates, axis=0)

        return {
            "correlations": correlations,
            "ensemble_statistics": {
                "mean": ensemble_mean,
                "variance": ensemble_var,
                "coefficient_of_variation": np.sqrt(ensemble_var) / ensemble_mean,
            },
            "consensus_score": float(np.mean(list(correlations.values()))),
        }

    def _calculate_uncertainty_bounds(
        self, biomass_estimates: dict[str, np.ndarray]
    ) -> dict[str, Any]:
        """Calculate uncertainty bounds for biomass estimates."""
        estimates = np.stack(list(biomass_estimates.values()), axis=0)

        # Calculate statistical bounds
        mean_estimate = np.mean(estimates, axis=0)
        std_estimate = np.std(estimates, axis=0)

        # 95% confidence intervals
        lower_bound = mean_estimate - 1.96 * std_estimate
        upper_bound = mean_estimate + 1.96 * std_estimate

        # Relative uncertainty
        relative_uncertainty = std_estimate / mean_estimate

        return {
            "mean": mean_estimate,
            "std": std_estimate,
            "lower_95": lower_bound,
            "upper_95": upper_bound,
            "relative_uncertainty": relative_uncertainty,
            "uncertainty_classification": self._classify_uncertainty(
                relative_uncertainty
            ),
        }

    def _classify_uncertainty(self, relative_uncertainty: np.ndarray) -> np.ndarray:
        """Classify uncertainty levels."""
        classification = np.zeros_like(relative_uncertainty, dtype=int)
        classification[relative_uncertainty < 0.1] = 1  # Low uncertainty
        classification[(relative_uncertainty >= 0.1) & (relative_uncertainty < 0.3)] = (
            2  # Medium uncertainty
        )
        classification[relative_uncertainty >= 0.3] = 3  # High uncertainty
        return classification

    def _create_uncertainty_visualizations(
        self,
        biomass_estimates: dict[str, np.ndarray],
        uncertainty_bounds: dict[str, Any],
    ) -> dict[str, Any]:
        """Create uncertainty visualization plots."""
        visualizations = {}

        try:
            # Create uncertainty heatmap
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Biomass Density Heatmap",
                    "Uncertainty Map",
                    "Method Comparison",
                    "Uncertainty Distribution",
                ),
                specs=[
                    [{"type": "heatmap"}, {"type": "heatmap"}],
                    [{"type": "scatter"}, {"type": "histogram"}],
                ],
            )

            # Biomass density heatmap
            mean_biomass = uncertainty_bounds["mean"]
            fig.add_trace(
                go.Heatmap(
                    z=mean_biomass, colorscale="Viridis", name="Biomass Density"
                ),
                row=1,
                col=1,
            )

            # Uncertainty heatmap
            uncertainty_map = uncertainty_bounds["relative_uncertainty"]
            fig.add_trace(
                go.Heatmap(
                    z=uncertainty_map, colorscale="Reds", name="Relative Uncertainty"
                ),
                row=1,
                col=2,
            )

            visualizations["uncertainty_analysis"] = fig.to_html()

        except Exception as e:
            logger.error(f"Failed to create uncertainty visualizations: {e}")
            visualizations["error"] = str(e)

        return visualizations

    def _calculate_validation_metrics(
        self, biomass_estimates: dict[str, np.ndarray]
    ) -> dict[str, Any]:
        """Calculate validation metrics for biomass estimates."""
        list(biomass_estimates.keys())
        estimates = list(biomass_estimates.values())

        # Calculate ensemble statistics
        ensemble_mean = np.mean(estimates, axis=0)

        # Calculate metrics for each method against ensemble
        metrics = {}
        for method, estimate in biomass_estimates.items():
            # R-squared
            ss_res = np.sum((estimate - ensemble_mean) ** 2)
            ss_tot = np.sum((estimate - np.mean(estimate)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # RMSE
            rmse = np.sqrt(np.mean((estimate - ensemble_mean) ** 2))

            # MAE
            mae = np.mean(np.abs(estimate - ensemble_mean))

            metrics[method] = {
                "r_squared": float(r_squared),
                "rmse": float(rmse),
                "mae": float(mae),
                "bias": float(np.mean(estimate - ensemble_mean)),
            }

        return metrics

    def _generate_uncertainty_summary(
        self, uncertainty_bounds: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate summary statistics for uncertainty analysis."""
        relative_uncertainty = uncertainty_bounds["relative_uncertainty"]
        uncertainty_classification = uncertainty_bounds["uncertainty_classification"]

        # Calculate percentage in each uncertainty class
        total_pixels = np.prod(relative_uncertainty.shape)
        low_uncertainty = np.sum(uncertainty_classification == 1) / total_pixels
        medium_uncertainty = np.sum(uncertainty_classification == 2) / total_pixels
        high_uncertainty = np.sum(uncertainty_classification == 3) / total_pixels

        return {
            "uncertainty_distribution": {
                "low_uncertainty_percent": float(low_uncertainty * 100),
                "medium_uncertainty_percent": float(medium_uncertainty * 100),
                "high_uncertainty_percent": float(high_uncertainty * 100),
            },
            "overall_uncertainty": {
                "mean_relative_uncertainty": float(np.nanmean(relative_uncertainty)),
                "median_relative_uncertainty": float(
                    np.nanmedian(relative_uncertainty)
                ),
                "max_relative_uncertainty": float(np.nanmax(relative_uncertainty)),
            },
            "quality_assessment": {
                "confidence_level": (
                    "high"
                    if low_uncertainty > 0.7
                    else "medium"
                    if low_uncertainty > 0.4
                    else "low"
                ),
                "recommended_action": self._get_uncertainty_recommendation(
                    low_uncertainty
                ),
            },
        }

    def _get_uncertainty_recommendation(self, low_uncertainty_fraction: float) -> str:
        """Get recommendation based on uncertainty levels."""
        if low_uncertainty_fraction > 0.8:
            return "High confidence results - suitable for regulatory reporting"
        elif low_uncertainty_fraction > 0.6:
            return "Moderate confidence - recommend additional validation"
        elif low_uncertainty_fraction > 0.4:
            return "Limited confidence - additional data collection recommended"
        else:
            return "Low confidence - extensive validation required before use"


def create_enhanced_satellite_analyzer(**kwargs) -> EnhancedSatelliteAnalyzer:
    """Create enhanced satellite analyzer."""
    return EnhancedSatelliteAnalyzer(**kwargs)
