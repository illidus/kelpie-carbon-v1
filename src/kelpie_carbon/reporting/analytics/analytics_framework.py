"""
Core Analytics Framework for Kelpie Carbon v1

This module provides the central analytics engine that integrates all analysis types
(validation, temporal, species, historical, deep learning) into unified analytics
and performance metrics.

Classes:
    AnalyticsFramework: Main analytics engine
    AnalysisRequest: Request specification for analytics
    AnalysisResult: Standardized analysis result container
    MetricCalculator: Performance and accuracy metrics
    TrendAnalyzer: Temporal trend analysis integration
    PerformanceMetrics: System performance tracking
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of analysis available in the framework."""
    VALIDATION = "validation"
    TEMPORAL = "temporal"
    SPECIES = "species"
    HISTORICAL = "historical"
    DEEP_LEARNING = "deep_learning"
    SUBMERGED = "submerged"
    COMPREHENSIVE = "comprehensive"

class OutputFormat(Enum):
    """Output formats for analysis results."""
    JSON = "json"
    DATAFRAME = "dataframe"
    SUMMARY = "summary"
    FULL = "full"

@dataclass
class AnalysisRequest:
    """Request specification for analytics framework."""
    
    analysis_types: list[AnalysisType]
    site_coordinates: tuple[float, float]  # (lat, lon)
    time_range: tuple[datetime, datetime]
    output_format: OutputFormat = OutputFormat.FULL
    include_confidence: bool = True
    include_uncertainty: bool = True
    stakeholder_type: str = "scientific"
    quality_threshold: float = 0.7
    options: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate analysis request."""
        if not self.analysis_types:
            raise ValueError("At least one analysis type must be specified")
        
        lat, lon = self.site_coordinates
        if not (-90 <= lat <= 90):
            raise ValueError(f"Invalid latitude: {lat}")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Invalid longitude: {lon}")
        
        if self.time_range[0] >= self.time_range[1]:
            raise ValueError("Start time must be before end time")

@dataclass
class AnalysisResult:
    """Standardized container for analysis results."""
    
    request: AnalysisRequest
    results: dict[str, Any]
    metrics: dict[str, float]
    confidence_scores: dict[str, float]
    uncertainty_estimates: dict[str, tuple[float, float]]
    execution_time: float
    data_quality: dict[str, float]
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get_summary(self) -> dict[str, Any]:
        """Get concise summary of analysis results."""
        return {
            "analysis_types": [t.value for t in self.request.analysis_types],
            "site_location": self.request.site_coordinates,
            "time_range": [t.isoformat() for t in self.request.time_range],
            "overall_confidence": np.mean(list(self.confidence_scores.values())),
            "data_quality_score": np.mean(list(self.data_quality.values())),
            "execution_time_seconds": self.execution_time,
            "key_findings": self._extract_key_findings(),
            "recommendations": self.recommendations[:3]  # Top 3
        }
    
    def _extract_key_findings(self) -> list[str]:
        """Extract key findings from analysis results."""
        findings = []
        
        # Kelp extent findings
        if "kelp_extent" in self.results:
            extent = self.results["kelp_extent"]
            findings.append(f"Current kelp extent: {extent:.1f} hectares")
        
        # Trend findings
        if "trend_analysis" in self.results:
            trend = self.results["trend_analysis"]
            if "direction" in trend:
                findings.append(f"Kelp extent trend: {trend['direction']}")
        
        # Risk findings
        if "risk_assessment" in self.results:
            risk = self.results["risk_assessment"]
            if "level" in risk:
                findings.append(f"Conservation risk level: {risk['level']}")
        
        return findings

class MetricCalculator:
    """Calculate performance and accuracy metrics across analysis types."""
    
    def __init__(self):
        self.metric_weights = {
            "accuracy": 0.3,
            "precision": 0.2,
            "recall": 0.2,
            "confidence": 0.15,
            "data_quality": 0.15
        }
    
    def calculate_composite_score(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        confidence: float,
        data_quality: float
    ) -> float:
        """Calculate weighted composite performance score."""
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "confidence": confidence,
            "data_quality": data_quality
        }
        
        # Validate inputs
        for name, value in metrics.items():
            if not (0 <= value <= 1):
                logger.warning(f"Metric {name} outside valid range [0,1]: {value}")
                metrics[name] = max(0, min(1, value))
        
        # Calculate weighted score
        score = sum(metrics[name] * self.metric_weights[name] for name in metrics)
        return round(score, 3)
    
    def calculate_detection_metrics(
        self,
        true_positives: int,
        false_positives: int,
        false_negatives: int,
        true_negatives: int
    ) -> dict[str, float]:
        """Calculate standard detection performance metrics."""
        
        # Avoid division by zero
        total = true_positives + false_positives + false_negatives + true_negatives
        if total == 0:
            return {"error": "No detection data available"}
        
        # Basic metrics
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        
        # F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "specificity": round(specificity, 3),
            "f1_score": round(f1_score, 3),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives
        }
    
    def calculate_temporal_metrics(
        self,
        time_series_data: list[tuple[datetime, float]],
        reference_data: list[tuple[datetime, float]] | None = None
    ) -> dict[str, float]:
        """Calculate temporal analysis performance metrics."""
        
        if len(time_series_data) < 3:
            return {"error": "Insufficient temporal data"}
        
        # Extract values and calculate statistics
        values = [value for _, value in time_series_data]
        
        metrics = {
            "temporal_coverage": len(time_series_data),
            "mean_value": np.mean(values),
            "std_value": np.std(values),
            "coefficient_variation": np.std(values) / np.mean(values) if np.mean(values) > 0 else 0,
            "trend_strength": self._calculate_trend_strength(values),
            "seasonality_score": self._calculate_seasonality_score(values)
        }
        
        # Compare with reference data if available
        if reference_data:
            ref_values = [value for _, value in reference_data]
            if len(ref_values) > 0:
                correlation = np.corrcoef(values[:min(len(values), len(ref_values))], 
                                       ref_values[:min(len(values), len(ref_values))])[0, 1]
                metrics["reference_correlation"] = correlation if not np.isnan(correlation) else 0
        
        return {k: round(v, 3) for k, v in metrics.items()}
    
    def _calculate_trend_strength(self, values: list[float]) -> float:
        """Calculate strength of temporal trend."""
        if len(values) < 3:
            return 0
        
        x = np.arange(len(values))
        correlation = np.corrcoef(x, values)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0
    
    def _calculate_seasonality_score(self, values: list[float]) -> float:
        """Calculate seasonality score for time series."""
        if len(values) < 12:  # Need at least a year of monthly data
            return 0
        
        # Simple seasonality detection using autocorrelation
        try:
            autocorr_12 = np.corrcoef(values[:-12], values[12:])[0, 1]
            return abs(autocorr_12) if not np.isnan(autocorr_12) else 0
        except:
            return 0

class TrendAnalyzer:
    """Integrate temporal trend analysis across all analysis types."""
    
    def __init__(self):
        self.trend_methods = ["linear", "polynomial", "seasonal"]
    
    def analyze_kelp_trends(
        self,
        temporal_data: dict[datetime, float],
        analysis_type: str = "comprehensive"
    ) -> dict[str, Any]:
        """Analyze kelp extent trends using multiple methods."""
        
        if len(temporal_data) < 3:
            return {"error": "Insufficient data for trend analysis"}
        
        # Prepare data
        dates = sorted(temporal_data.keys())
        values = [temporal_data[date] for date in dates]
        x = np.arange(len(dates))
        
        results = {}
        
        # Linear trend
        if len(values) >= 2:
            linear_coef = np.polyfit(x, values, 1)
            linear_r2 = 1 - (np.sum((values - np.polyval(linear_coef, x))**2) / 
                           np.sum((values - np.mean(values))**2))
            
            results["linear_trend"] = {
                "slope": linear_coef[0],
                "intercept": linear_coef[1],
                "r_squared": linear_r2,
                "direction": "increasing" if linear_coef[0] > 0 else "decreasing" if linear_coef[0] < 0 else "stable"
            }
        
        # Polynomial trend (if enough data)
        if len(values) >= 4:
            poly_coef = np.polyfit(x, values, 2)
            poly_r2 = 1 - (np.sum((values - np.polyval(poly_coef, x))**2) / 
                          np.sum((values - np.mean(values))**2))
            
            results["polynomial_trend"] = {
                "coefficients": poly_coef.tolist(),
                "r_squared": poly_r2,
                "curvature": "concave_up" if poly_coef[0] > 0 else "concave_down" if poly_coef[0] < 0 else "linear"
            }
        
        # Change point detection
        if len(values) >= 6:
            change_points = self._detect_change_points(values)
            results["change_points"] = change_points
        
        # Rate of change analysis
        if len(values) >= 2:
            rates = np.diff(values)
            results["rate_analysis"] = {
                "mean_rate": np.mean(rates),
                "rate_variability": np.std(rates),
                "acceleration": np.mean(np.diff(rates)) if len(rates) > 1 else 0
            }
        
        # Risk assessment
        results["risk_assessment"] = self._assess_trend_risk(results)
        
        return results
    
    def _detect_change_points(self, values: list[float]) -> list[int]:
        """Simple change point detection using moving averages."""
        if len(values) < 6:
            return []
        
        change_points = []
        window = max(2, len(values) // 6)
        
        for i in range(window, len(values) - window):
            before_mean = np.mean(values[i-window:i])
            after_mean = np.mean(values[i:i+window])
            
            # Detect significant change (>20% difference)
            if abs(after_mean - before_mean) / before_mean > 0.2:
                change_points.append(i)
        
        return change_points
    
    def _assess_trend_risk(self, trend_results: dict[str, Any]) -> dict[str, Any]:
        """Assess conservation risk based on trend analysis."""
        risk_factors = []
        risk_score = 0
        
        # Linear trend risk
        if "linear_trend" in trend_results:
            slope = trend_results["linear_trend"]["slope"]
            r2 = trend_results["linear_trend"]["r_squared"]
            
            if slope < -0.5 and r2 > 0.5:  # Strong declining trend
                risk_factors.append("Strong declining trend")
                risk_score += 0.4
            elif slope < -0.1:  # Mild declining trend
                risk_factors.append("Mild declining trend")
                risk_score += 0.2
        
        # Change points risk
        if "change_points" in trend_results and len(trend_results["change_points"]) > 0:
            risk_factors.append("Abrupt changes detected")
            risk_score += 0.2
        
        # Rate variability risk
        if "rate_analysis" in trend_results:
            variability = trend_results["rate_analysis"]["rate_variability"]
            if variability > 1.0:  # High variability
                risk_factors.append("High variability in changes")
                risk_score += 0.1
        
        # Classify risk level
        if risk_score >= 0.4:
            risk_level = "HIGH"
        elif risk_score >= 0.2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "recommendations": self._generate_risk_recommendations(risk_level, risk_factors)
        }
    
    def _generate_risk_recommendations(self, risk_level: str, risk_factors: list[str]) -> list[str]:
        """Generate management recommendations based on risk assessment."""
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.extend([
                "Implement immediate conservation measures",
                "Increase monitoring frequency to monthly",
                "Investigate environmental stressors",
                "Consider active restoration interventions"
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "Enhance monitoring program",
                "Assess environmental conditions",
                "Develop conservation contingency plans",
                "Engage with local stakeholders"
            ])
        else:
            recommendations.extend([
                "Continue regular monitoring",
                "Maintain current conservation status",
                "Monitor for emerging threats"
            ])
        
        return recommendations

class PerformanceMetrics:
    """Track system performance across all analysis components."""
    
    def __init__(self):
        self.performance_history = []
        self.benchmark_targets = {
            "processing_time": 30.0,  # seconds
            "accuracy": 0.85,
            "data_quality": 0.8,
            "system_availability": 0.99
        }
    
    def record_performance(
        self,
        analysis_type: str,
        processing_time: float,
        accuracy: float,
        data_quality: float,
        timestamp: datetime | None = None
    ) -> None:
        """Record performance metrics for an analysis."""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        performance_record = {
            "timestamp": timestamp,
            "analysis_type": analysis_type,
            "processing_time": processing_time,
            "accuracy": accuracy,
            "data_quality": data_quality,
            "meets_targets": self._check_targets(processing_time, accuracy, data_quality)
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only last 1000 records
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def _check_targets(self, processing_time: float, accuracy: float, data_quality: float) -> dict[str, bool]:
        """Check if performance meets benchmark targets."""
        return {
            "processing_time": processing_time <= self.benchmark_targets["processing_time"],
            "accuracy": accuracy >= self.benchmark_targets["accuracy"],
            "data_quality": data_quality >= self.benchmark_targets["data_quality"]
        }
    
    def get_performance_summary(self, days: int = 30) -> dict[str, Any]:
        """Get performance summary for the last N days."""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_records = [r for r in self.performance_history if r["timestamp"] >= cutoff_date]
        
        if not recent_records:
            return {"error": "No recent performance data available"}
        
        # Calculate aggregated metrics
        processing_times = [r["processing_time"] for r in recent_records]
        accuracies = [r["accuracy"] for r in recent_records]
        data_qualities = [r["data_quality"] for r in recent_records]
        
        target_compliance = {
            target: sum(1 for r in recent_records if r["meets_targets"][target]) / len(recent_records)
            for target in ["processing_time", "accuracy", "data_quality"]
        }
        
        return {
            "period_days": days,
            "total_analyses": len(recent_records),
            "performance_metrics": {
                "mean_processing_time": np.mean(processing_times),
                "mean_accuracy": np.mean(accuracies),
                "mean_data_quality": np.mean(data_qualities),
                "processing_time_95th": np.percentile(processing_times, 95),
                "accuracy_5th": np.percentile(accuracies, 5),
                "data_quality_5th": np.percentile(data_qualities, 5)
            },
            "target_compliance": target_compliance,
            "overall_system_health": np.mean(list(target_compliance.values()))
        }

class AnalyticsFramework:
    """
    Main analytics framework integrating all analysis types.
    
    This class provides the central engine for comprehensive kelp analysis,
    integrating validation, temporal, species, historical, and deep learning
    capabilities into unified analytics and reporting.
    """
    
    def __init__(self):
        self.metric_calculator = MetricCalculator()
        self.trend_analyzer = TrendAnalyzer()
        self.performance_metrics = PerformanceMetrics()
        
        # Analysis components (would be injected or imported)
        self.analysis_components = {}
        self._initialize_analysis_components()
    
    def _initialize_analysis_components(self):
        """Initialize analysis component references."""
        # These would normally be injected or imported from other modules
        self.analysis_components = {
            "validation": "ValidationFramework",  # Placeholder
            "temporal": "TemporalAnalyzer",
            "species": "SpeciesClassifier", 
            "historical": "HistoricalAnalyzer",
            "deep_learning": "DeepLearningDetector",
            "submerged": "SubmergedDetector"
        }
    
    def execute_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """Execute comprehensive analysis based on request."""
        
        start_time = datetime.now()
        results = {}
        confidence_scores = {}
        uncertainty_estimates = {}
        data_quality = {}
        recommendations = []
        
        try:
            # Execute each requested analysis type
            for analysis_type in request.analysis_types:
                logger.info(f"Executing {analysis_type.value} analysis")
                
                analysis_result = self._execute_single_analysis(
                    analysis_type, request
                )
                
                results[analysis_type.value] = analysis_result["data"]
                confidence_scores[analysis_type.value] = analysis_result["confidence"]
                uncertainty_estimates[analysis_type.value] = analysis_result["uncertainty"]
                data_quality[analysis_type.value] = analysis_result["quality"]
                recommendations.extend(analysis_result.get("recommendations", []))
            
            # Calculate integrated metrics
            integrated_metrics = self._calculate_integrated_metrics(results, confidence_scores)
            
            # Execute cross-analysis integration
            if len(request.analysis_types) > 1:
                integration_results = self._integrate_cross_analysis(results, request)
                results["integration"] = integration_results
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Record performance
            overall_accuracy = np.mean(list(confidence_scores.values()))
            overall_quality = np.mean(list(data_quality.values()))
            
            self.performance_metrics.record_performance(
                analysis_type="comprehensive",
                processing_time=execution_time,
                accuracy=overall_accuracy,
                data_quality=overall_quality
            )
            
            # Create result object
            result = AnalysisResult(
                request=request,
                results=results,
                metrics=integrated_metrics,
                confidence_scores=confidence_scores,
                uncertainty_estimates=uncertainty_estimates,
                execution_time=execution_time,
                data_quality=data_quality,
                recommendations=list(set(recommendations))  # Remove duplicates
            )
            
            logger.info(f"Analysis completed in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Analysis execution failed: {e}")
            
            # Return error result
            execution_time = (datetime.now() - start_time).total_seconds()
            return AnalysisResult(
                request=request,
                results={"error": str(e)},
                metrics={"error": True},
                confidence_scores={},
                uncertainty_estimates={},
                execution_time=execution_time,
                data_quality={},
                recommendations=["Review input parameters and try again"]
            )
    
    def _execute_single_analysis(
        self, 
        analysis_type: AnalysisType, 
        request: AnalysisRequest
    ) -> dict[str, Any]:
        """Execute a single analysis type."""
        
        # This is a simplified implementation - in practice would call actual analysis modules
        lat, lon = request.site_coordinates
        start_time, end_time = request.time_range
        
        if analysis_type == AnalysisType.VALIDATION:
            return self._mock_validation_analysis(lat, lon, start_time, end_time)
        elif analysis_type == AnalysisType.TEMPORAL:
            return self._mock_temporal_analysis(lat, lon, start_time, end_time)
        elif analysis_type == AnalysisType.SPECIES:
            return self._mock_species_analysis(lat, lon)
        elif analysis_type == AnalysisType.HISTORICAL:
            return self._mock_historical_analysis(lat, lon)
        elif analysis_type == AnalysisType.DEEP_LEARNING:
            return self._mock_deep_learning_analysis(lat, lon)
        elif analysis_type == AnalysisType.SUBMERGED:
            return self._mock_submerged_analysis(lat, lon)
        else:
            return {
                "data": {"error": f"Unknown analysis type: {analysis_type}"},
                "confidence": 0.0,
                "uncertainty": (0.0, 0.0),
                "quality": 0.0
            }
    
    def _mock_validation_analysis(self, lat: float, lon: float, start_time: datetime, end_time: datetime) -> dict[str, Any]:
        """Mock validation analysis - would integrate with actual validation framework."""
        return {
            "data": {
                "kelp_extent": 125.5,
                "detection_accuracy": 0.87,
                "validation_sites": 3,
                "data_points": 24
            },
            "confidence": 0.87,
            "uncertainty": (115.0, 136.0),
            "quality": 0.85,
            "recommendations": ["Continue regular monitoring", "Validate against field data"]
        }
    
    def _mock_temporal_analysis(self, lat: float, lon: float, start_time: datetime, end_time: datetime) -> dict[str, Any]:
        """Mock temporal analysis - would integrate with actual temporal framework."""
        return {
            "data": {
                "trend_direction": "decreasing",
                "annual_change_rate": -2.3,
                "seasonal_patterns": True,
                "trend_significance": 0.02
            },
            "confidence": 0.82,
            "uncertainty": (-3.1, -1.5),
            "quality": 0.80,
            "recommendations": ["Investigate causes of decline", "Increase monitoring frequency"]
        }
    
    def _mock_species_analysis(self, lat: float, lon: float) -> dict[str, Any]:
        """Mock species analysis - would integrate with actual species classifier."""
        return {
            "data": {
                "primary_species": "Nereocystis luetkeana",
                "species_confidence": 0.91,
                "biomass_estimate": 850.2,
                "species_diversity": 2
            },
            "confidence": 0.91,
            "uncertainty": (750.0, 950.0),
            "quality": 0.88,
            "recommendations": ["Monitor species composition changes"]
        }
    
    def _mock_historical_analysis(self, lat: float, lon: float) -> dict[str, Any]:
        """Mock historical analysis - would integrate with actual historical framework."""
        return {
            "data": {
                "historical_baseline": 180.5,
                "current_vs_historical": -30.5,
                "change_significance": 0.001,
                "risk_level": "MEDIUM"
            },
            "confidence": 0.78,
            "uncertainty": (-40.0, -21.0),
            "quality": 0.75,
            "recommendations": ["Compare with historical baselines", "Assess long-term trends"]
        }
    
    def _mock_deep_learning_analysis(self, lat: float, lon: float) -> dict[str, Any]:
        """Mock deep learning analysis - would integrate with actual DL framework."""
        return {
            "data": {
                "dl_detection_accuracy": 0.89,
                "sam_segments": 45,
                "unet_confidence": 0.85,
                "ensemble_prediction": 128.7
            },
            "confidence": 0.89,
            "uncertainty": (118.0, 139.0),
            "quality": 0.87,
            "recommendations": ["Validate DL predictions with ground truth"]
        }
    
    def _mock_submerged_analysis(self, lat: float, lon: float) -> dict[str, Any]:
        """Mock submerged kelp analysis - would integrate with actual submerged detector."""
        return {
            "data": {
                "submerged_extent": 45.2,
                "max_detection_depth": 85,
                "depth_profile": [30, 50, 70, 85],
                "red_edge_effectiveness": 0.74
            },
            "confidence": 0.74,
            "uncertainty": (38.0, 52.0),
            "quality": 0.72,
            "recommendations": ["Validate depth detection capabilities"]
        }
    
    def _calculate_integrated_metrics(
        self, 
        results: dict[str, Any], 
        confidence_scores: dict[str, float]
    ) -> dict[str, float]:
        """Calculate metrics that integrate across analysis types."""
        
        # Overall system performance
        overall_confidence = np.mean(list(confidence_scores.values())) if confidence_scores else 0
        
        # Kelp extent consistency
        extent_values = []
        for analysis_result in results.values():
            if isinstance(analysis_result, dict):
                if "kelp_extent" in analysis_result:
                    extent_values.append(analysis_result["kelp_extent"])
                elif "ensemble_prediction" in analysis_result:
                    extent_values.append(analysis_result["ensemble_prediction"])
        
        extent_consistency = 1 - (np.std(extent_values) / np.mean(extent_values)) if len(extent_values) > 1 and np.mean(extent_values) > 0 else 1
        
        return {
            "overall_confidence": round(overall_confidence, 3),
            "extent_consistency": round(extent_consistency, 3),
            "analysis_coverage": len(results),
            "integration_score": round((overall_confidence + extent_consistency) / 2, 3)
        }
    
    def _integrate_cross_analysis(
        self, 
        results: dict[str, Any], 
        request: AnalysisRequest
    ) -> dict[str, Any]:
        """Integrate results across multiple analysis types."""
        
        integration = {
            "cross_validation": {},
            "consensus_metrics": {},
            "disagreement_analysis": {}
        }
        
        # Extract kelp extent estimates from different methods
        extent_estimates = {}
        for analysis_type, result in results.items():
            if isinstance(result, dict) and not result.get("error"):
                if "kelp_extent" in result:
                    extent_estimates[analysis_type] = result["kelp_extent"]
                elif "ensemble_prediction" in result:
                    extent_estimates[analysis_type] = result["ensemble_prediction"]
        
        if len(extent_estimates) > 1:
            # Calculate consensus estimate
            estimates = list(extent_estimates.values())
            consensus_estimate = np.mean(estimates)
            estimate_std = np.std(estimates)
            
            integration["consensus_metrics"] = {
                "consensus_kelp_extent": round(consensus_estimate, 1),
                "estimate_uncertainty": round(estimate_std, 1),
                "coefficient_variation": round(estimate_std / consensus_estimate, 3) if consensus_estimate > 0 else 0,
                "contributing_methods": list(extent_estimates.keys())
            }
            
            # Disagreement analysis
            max_diff = max(estimates) - min(estimates)
            integration["disagreement_analysis"] = {
                "max_difference": round(max_diff, 1),
                "relative_disagreement": round(max_diff / consensus_estimate, 3) if consensus_estimate > 0 else 0,
                "agreement_level": "high" if max_diff / consensus_estimate < 0.1 else "medium" if max_diff / consensus_estimate < 0.2 else "low"
            }
        
        return integration
    
    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health and performance summary."""
        
        performance_summary = self.performance_metrics.get_performance_summary()
        
        if "error" in performance_summary:
            return {
                "status": "unknown",
                "message": "Insufficient performance data",
                "last_updated": datetime.now().isoformat()
            }
        
        # Determine system health status
        health_score = performance_summary["overall_system_health"]
        
        if health_score >= 0.9:
            status = "excellent"
        elif health_score >= 0.8:
            status = "good"
        elif health_score >= 0.7:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "status": status,
            "health_score": round(health_score, 3),
            "performance_summary": performance_summary,
            "recommendations": self._generate_system_recommendations(performance_summary),
            "last_updated": datetime.now().isoformat()
        }
    
    def _generate_system_recommendations(self, performance_summary: dict[str, Any]) -> list[str]:
        """Generate system-level recommendations based on performance."""
        recommendations = []
        
        compliance = performance_summary.get("target_compliance", {})
        
        if compliance.get("processing_time", 1) < 0.8:
            recommendations.append("Optimize processing performance")
        
        if compliance.get("accuracy", 1) < 0.8:
            recommendations.append("Review and improve detection algorithms")
        
        if compliance.get("data_quality", 1) < 0.8:
            recommendations.append("Enhance data quality control procedures")
        
        if not recommendations:
            recommendations.append("System operating within normal parameters")
        
        return recommendations

# Factory functions for easy usage
def create_analysis_request(
    analysis_types: list[str],
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    **kwargs
) -> AnalysisRequest:
    """Create an AnalysisRequest from simple parameters."""
    
    # Convert string analysis types to enum
    enum_types = []
    for type_str in analysis_types:
        try:
            enum_types.append(AnalysisType(type_str.lower()))
        except ValueError:
            logger.warning(f"Unknown analysis type: {type_str}")
    
    # Parse dates
    start_time = datetime.fromisoformat(start_date)
    end_time = datetime.fromisoformat(end_date)
    
    return AnalysisRequest(
        analysis_types=enum_types,
        site_coordinates=(latitude, longitude),
        time_range=(start_time, end_time),
        **kwargs
    )

def quick_analysis(
    dataset_path: str = None,
    analysis_types: list[str] = None,
    region: str = "default",
    latitude: float = None,
    longitude: float = None,
    analysis_type: str = None,
    **kwargs
) -> dict[str, Any]:
    """
    Quick analysis convenience function for basic usage.
    
    Args:
        dataset_path: Path to the dataset (optional if lat/lng provided)
        analysis_types: List of analysis types to run
        region: Region identifier
        latitude: Latitude for coordinate-based analysis
        longitude: Longitude for coordinate-based analysis  
        analysis_type: Single analysis type (alternative to analysis_types list)
        
    Returns:
        Dictionary containing analysis results
    """
    # Handle backward compatibility with test signature
    if latitude is not None and longitude is not None:
        # Coordinate-based analysis (test format)
        if analysis_type is not None:
            if analysis_type == "comprehensive":
                analysis_types = ['validation', 'temporal', 'species', 'historical']
            else:
                analysis_types = [analysis_type]
        
        if analysis_types is None:
            analysis_types = ['validation', 'temporal', 'species']
        
        # Create a request with coordinates
        start_time = datetime.now() - timedelta(days=365)
        end_time = datetime.now()
        
        request = create_analysis_request(
            analysis_types=analysis_types,
            latitude=latitude,
            longitude=longitude,
            start_date=start_time.isoformat(),
            end_date=end_time.isoformat()
        )
        
        framework = AnalyticsFramework()
        result = framework.execute_analysis(request)
        return result.get_summary()
    
    # Original dataset-path based analysis
    if analysis_types is None:
        analysis_types = ['validation', 'temporal', 'species']
    
    framework = AnalyticsFramework()
    
    # For dataset path mode, create a mock request
    request = AnalysisRequest(
        analysis_types=[AnalysisType(t.lower()) for t in analysis_types],
        site_coordinates=(48.5, -123.5),  # Default coordinates
        time_range=(datetime.now() - timedelta(days=365), datetime.now())
    )
    
    result = framework.execute_analysis(request)
    return result.get_summary()

def create_analytics_framework() -> AnalyticsFramework:
    """Create and return a configured analytics framework instance.
    
    Returns:
        Configured AnalyticsFramework instance ready for use
    """
    return AnalyticsFramework() 
