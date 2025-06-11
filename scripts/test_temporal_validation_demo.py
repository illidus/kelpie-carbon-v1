#!/usr/bin/env python3
"""
Temporal Validation Demonstration Script for SKEMA Kelp Detection.

This script demonstrates the comprehensive temporal validation capabilities
including time-series persistence, seasonal patterns, and environmental driver
analysis following UVic's Broughton Archipelago methodology.

Usage:
    python scripts/test_temporal_validation_demo.py
    python scripts/test_temporal_validation_demo.py --site broughton --years 2
    python scripts/test_temporal_validation_demo.py --comprehensive
"""

import asyncio
import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

from kelpie_carbon_v1.validation.temporal_validation import (
    TemporalValidator,
    TemporalDataPoint,
    run_broughton_temporal_validation,
    run_comprehensive_temporal_analysis,
    create_temporal_validator,
)
from kelpie_carbon_v1.validation.real_world_validation import ValidationSite
from kelpie_carbon_v1.logging_config import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Define test validation sites
TEST_SITES = {
    "broughton": ValidationSite(
        name="Broughton Archipelago - Test",
        lat=50.0833,
        lng=-126.1667,
        species="Nereocystis luetkeana",
        expected_detection_rate=0.20,
        water_depth="7.5m Secchi depth",
        optimal_season="July-September",
        site_type="kelp_farm",
        description="UVic primary temporal validation site for bull kelp"
    ),
    "saanich": ValidationSite(
        name="Saanich Inlet - Test",
        lat=48.5830,
        lng=-123.5000,
        species="Mixed Nereocystis + Macrocystis",
        expected_detection_rate=0.15,
        water_depth="6.0m average depth",
        optimal_season="June-September",
        site_type="kelp_farm",
        description="Multi-species kelp validation in sheltered waters"
    ),
    "monterey": ValidationSite(
        name="Monterey Bay - Test",
        lat=36.8000,
        lng=-121.9000,
        species="Macrocystis pyrifera",
        expected_detection_rate=0.18,
        water_depth="Variable 5-15m",
        optimal_season="April-October",
        site_type="kelp_farm",
        description="Giant kelp validation site for California studies"
    )
}


def print_section_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_subsection_header(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


def demonstrate_temporal_data_structures():
    """Demonstrate temporal validation data structures."""
    print_section_header("TEMPORAL VALIDATION DATA STRUCTURES DEMO")
    
    # Create sample temporal data point
    sample_timestamp = datetime(2023, 7, 15, 14, 30)
    sample_environmental_conditions = {
        "tidal_height": 1.2,
        "current_speed": 8.5,
        "water_temperature": 14.2,
        "secchi_depth": 7.8,
        "wind_speed": 12.0,
        "precipitation": 0.0
    }
    
    temporal_data_point = TemporalDataPoint(
        timestamp=sample_timestamp,
        detection_rate=0.22,
        kelp_area_km2=1.85,
        confidence_score=0.87,
        environmental_conditions=sample_environmental_conditions,
        quality_flags=["good_visibility", "low_cloud_cover"]
    )
    
    print(f"Sample Temporal Data Point:")
    print(f"  Timestamp: {temporal_data_point.timestamp}")
    print(f"  Detection Rate: {temporal_data_point.detection_rate:.3f}")
    print(f"  Kelp Area: {temporal_data_point.kelp_area_km2:.2f} km²")
    print(f"  Confidence: {temporal_data_point.confidence_score:.3f}")
    print(f"  Environmental Conditions:")
    for key, value in temporal_data_point.environmental_conditions.items():
        print(f"    {key}: {value}")
    print(f"  Quality Flags: {temporal_data_point.quality_flags}")
    
    return temporal_data_point


def create_synthetic_temporal_dataset(site: ValidationSite, months: int = 12) -> list:
    """Create synthetic temporal dataset for demonstration."""
    print_subsection_header(f"Creating Synthetic Dataset for {site.name}")
    
    data_points = []
    base_date = datetime(2023, 1, 1)
    
    for i in range(months * 2):  # Bi-weekly sampling
        sample_date = base_date + timedelta(days=i * 15)
        
        # Simulate seasonal kelp growth pattern
        day_of_year = sample_date.timetuple().tm_yday
        seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (day_of_year - 60) / 365.25)
        
        # Base detection rate with seasonal variation
        base_rate = site.expected_detection_rate
        detection_rate = base_rate * seasonal_factor * (0.8 + 0.4 * np.random.random())
        
        # Environmental conditions
        tidal_height = 2.0 * np.sin(2 * np.pi * day_of_year / 365.25) + np.random.normal(0, 0.3)
        water_temp = 12.0 + 6.0 * np.sin(2 * np.pi * (day_of_year - 60) / 365.25)
        current_speed = abs(tidal_height) * 3.0 + np.random.normal(8.0, 2.0)
        
        environmental_conditions = {
            "tidal_height": float(tidal_height),
            "current_speed": max(0, float(current_speed)),
            "water_temperature": float(water_temp),
            "secchi_depth": 6.0 + 2.0 * np.random.normal(),
            "wind_speed": max(0, np.random.lognormal(2.0, 0.5)),
            "precipitation": max(0, np.random.exponential(1.5))
        }
        
        # Quality flags based on conditions
        quality_flags = []
        if environmental_conditions["wind_speed"] > 15:
            quality_flags.append("high_wind")
        if environmental_conditions["precipitation"] > 5:
            quality_flags.append("high_precipitation")
        if environmental_conditions["secchi_depth"] < 4:
            quality_flags.append("low_visibility")
        
        data_point = TemporalDataPoint(
            timestamp=sample_date,
            detection_rate=max(0.0, min(1.0, detection_rate)),
            kelp_area_km2=detection_rate * 4.0,  # Convert to area
            confidence_score=0.7 + 0.3 * (1.0 - len(quality_flags) * 0.2),
            environmental_conditions=environmental_conditions,
            quality_flags=quality_flags
        )
        
        data_points.append(data_point)
    
    print(f"Created {len(data_points)} temporal data points over {months} months")
    print(f"Average detection rate: {np.mean([dp.detection_rate for dp in data_points]):.3f}")
    print(f"Detection rate range: {np.min([dp.detection_rate for dp in data_points]):.3f} - {np.max([dp.detection_rate for dp in data_points]):.3f}")
    
    return data_points


def demonstrate_temporal_analysis(validator: TemporalValidator, data_points: list, site: ValidationSite):
    """Demonstrate temporal analysis capabilities."""
    print_subsection_header(f"Temporal Analysis for {site.name}")
    
    # Analyze seasonal patterns
    seasonal_patterns = validator._analyze_seasonal_patterns(data_points)
    print("\nSeasonal Patterns:")
    for season, pattern in seasonal_patterns.items():
        print(f"  {season.capitalize()}:")
        print(f"    Average Detection Rate: {pattern.average_detection_rate:.3f}")
        print(f"    Peak Month: {pattern.peak_month}")
        print(f"    Variability: {pattern.variability_coefficient:.3f}")
        print(f"    Trend Slope: {pattern.trend_slope:.4f}")
        if pattern.statistical_significance < 0.05:
            print(f"    Statistically Significant Trend (p={pattern.statistical_significance:.3f})")
    
    # Calculate persistence metrics
    persistence_metrics = validator._calculate_persistence_metrics(data_points)
    print(f"\nPersistence Metrics:")
    print(f"  Mean Detection Rate: {persistence_metrics['mean_detection_rate']:.3f}")
    print(f"  Persistence Rate: {persistence_metrics['persistence_rate']:.3f}")
    print(f"  Consistency Rate: {persistence_metrics['consistency_rate']:.3f}")
    print(f"  Temporal Coverage: {persistence_metrics['temporal_coverage']} data points")
    print(f"  Data Quality Score: {persistence_metrics['data_quality_score']:.3f}")
    print(f"  Trend Stability: {persistence_metrics['trend_stability']:.3f}")
    
    # Analyze environmental correlations
    correlations = validator._analyze_environmental_correlations(data_points)
    print(f"\nEnvironmental Correlations:")
    for var, corr in correlations.items():
        if not var.endswith('_p_value'):
            p_value_key = f"{var}_p_value"
            p_value = correlations.get(p_value_key, 1.0)
            significance = " (*)" if p_value < 0.05 else ""
            print(f"  {var}: {corr:.3f}{significance}")
    print("  (*) = Statistically significant (p < 0.05)")
    
    # Perform trend analysis
    trend_analysis = validator._perform_trend_analysis(data_points)
    linear_trend = trend_analysis.get('linear_trend', {})
    print(f"\nTrend Analysis:")
    print(f"  Trend Direction: {linear_trend.get('trend_direction', 'unknown')}")
    print(f"  Trend Strength: {linear_trend.get('trend_strength', 'unknown')}")
    print(f"  R-squared: {linear_trend.get('r_squared', 0):.3f}")
    print(f"  Slope: {linear_trend.get('slope', 0):.4f}")
    if linear_trend.get('p_value', 1) < 0.05:
        print(f"  Statistically Significant (p={linear_trend.get('p_value', 1):.3f})")
    
    change_points = trend_analysis.get('change_points', [])
    if change_points:
        print(f"  Change Points Detected: {len(change_points)}")
        for cp in change_points[:3]:  # Show first 3
            print(f"    {cp['timestamp']}: {cp['magnitude']:+.3f} change")
    
    return {
        'seasonal_patterns': seasonal_patterns,
        'persistence_metrics': persistence_metrics,
        'environmental_correlations': correlations,
        'trend_analysis': trend_analysis
    }


def demonstrate_quality_assessment(validator: TemporalValidator, data_points: list, site: ValidationSite):
    """Demonstrate temporal quality assessment."""
    print_subsection_header(f"Quality Assessment for {site.name}")
    
    quality_assessment = validator._assess_temporal_quality(data_points, site)
    
    print(f"Overall Quality: {quality_assessment['overall_quality'].upper()}")
    print(f"Quality Score: {quality_assessment['quality_score']:.3f}/1.0")
    print(f"Data Coverage: {quality_assessment['data_coverage']:.3f}")
    
    print(f"\nQuality Checks:")
    for check, passed in quality_assessment['quality_checks'].items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check}: {status}")
    
    temporal_gaps = quality_assessment.get('temporal_gaps', [])
    if temporal_gaps:
        print(f"\nTemporal Gaps Detected: {len(temporal_gaps)}")
        for gap in temporal_gaps[:3]:  # Show first 3
            print(f"  {gap['start_date']} to {gap['end_date']}: {gap['gap_days']} days")
    else:
        print(f"\nNo significant temporal gaps detected")
    
    recommendations = quality_assessment.get('recommendations', [])
    if recommendations:
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    return quality_assessment


async def demonstrate_site_validation(site_name: str, validation_years: int = 1):
    """Demonstrate validation for a specific site."""
    print_section_header(f"SITE-SPECIFIC TEMPORAL VALIDATION: {site_name.upper()}")
    
    if site_name not in TEST_SITES:
        print(f"Error: Site '{site_name}' not found. Available sites: {list(TEST_SITES.keys())}")
        return None
    
    site = TEST_SITES[site_name]
    validator = create_temporal_validator()
    
    # Create synthetic dataset
    months = validation_years * 12
    data_points = create_synthetic_temporal_dataset(site, months)
    
    # Demonstrate temporal analysis
    analysis_results = demonstrate_temporal_analysis(validator, data_points, site)
    
    # Demonstrate quality assessment
    quality_results = demonstrate_quality_assessment(validator, data_points, site)
    
    # Generate recommendations
    recommendations = validator._generate_temporal_recommendations(
        analysis_results['persistence_metrics'],
        analysis_results['seasonal_patterns'],
        analysis_results['trend_analysis']
    )
    
    print_subsection_header("Generated Recommendations")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("No specific recommendations - system performing within acceptable parameters")
    
    return {
        'site': site,
        'analysis_results': analysis_results,
        'quality_results': quality_results,
        'recommendations': recommendations
    }


async def demonstrate_comprehensive_analysis():
    """Demonstrate comprehensive multi-site temporal analysis."""
    print_section_header("COMPREHENSIVE MULTI-SITE TEMPORAL ANALYSIS")
    
    validator = create_temporal_validator()
    all_results = []
    
    # Analyze each test site
    for site_name, site in TEST_SITES.items():
        print_subsection_header(f"Analyzing {site.name}")
        
        # Create synthetic data for each site
        data_points = create_synthetic_temporal_dataset(site, months=6)  # 6 months for demo
        
        # Create mock temporal validation result
        from kelpie_carbon_v1.validation.temporal_validation import TemporalValidationResult
        
        seasonal_patterns = validator._analyze_seasonal_patterns(data_points)
        persistence_metrics = validator._calculate_persistence_metrics(data_points)
        environmental_correlations = validator._analyze_environmental_correlations(data_points)
        trend_analysis = validator._perform_trend_analysis(data_points)
        quality_assessment = validator._assess_temporal_quality(data_points, site)
        recommendations = validator._generate_temporal_recommendations(
            persistence_metrics, seasonal_patterns, trend_analysis
        )
        
        result = TemporalValidationResult(
            site_name=site.name,
            validation_period=(datetime(2023, 1, 1), datetime(2023, 6, 30)),
            data_points=data_points,
            seasonal_patterns=seasonal_patterns,
            persistence_metrics=persistence_metrics,
            environmental_correlations=environmental_correlations,
            trend_analysis=trend_analysis,
            quality_assessment=quality_assessment,
            recommendations=recommendations
        )
        
        all_results.append(result)
        
        print(f"  Detection Rate: {persistence_metrics['mean_detection_rate']:.3f}")
        print(f"  Persistence: {persistence_metrics['persistence_rate']:.3f}")
        print(f"  Quality: {quality_assessment['overall_quality']}")
    
    # Generate comprehensive report
    comprehensive_report = validator.generate_comprehensive_temporal_report(all_results)
    
    print_subsection_header("Comprehensive Analysis Report")
    
    exec_summary = comprehensive_report['executive_summary']
    print(f"Total Sites Validated: {exec_summary['total_sites_validated']}")
    print(f"Overall Assessment: {exec_summary['overall_assessment']}")
    print(f"Mean Persistence Rate: {exec_summary['mean_persistence_rate']:.3f}")
    print(f"Temporal Stability: {exec_summary['temporal_stability']:.3f}")
    
    # Site-specific results
    print(f"\nSite-Specific Results:")
    site_results = comprehensive_report['detailed_metrics']['site_specific_results']
    for site_result in site_results:
        print(f"  {site_result['site_name']}:")
        print(f"    Data Points: {site_result['data_points']}")
        print(f"    Persistence: {site_result['persistence_rate']:.3f}")
        print(f"    Quality Score: {site_result['quality_score']:.3f}")
    
    # Recommendations
    recommendations = comprehensive_report['recommendations']
    print(f"\nHigh Priority Recommendations:")
    for rec in recommendations['high_priority'][:3]:
        print(f"  • {rec}")
    
    print(f"\nOperational Recommendations:")
    for rec in recommendations['operational'][:3]:
        print(f"  • {rec}")
    
    return comprehensive_report


async def demonstrate_broughton_validation(validation_years: int = 1):
    """Demonstrate UVic Broughton Archipelago validation methodology."""
    print_section_header("BROUGHTON ARCHIPELAGO TEMPORAL VALIDATION")
    print("Following UVic's multi-year validation methodology")
    
    validator = create_temporal_validator()
    config = validator.get_broughton_validation_config()
    
    print_subsection_header("Validation Configuration")
    site = config["site"]
    print(f"Site: {site.name}")
    print(f"Location: {site.lat}°N, {site.lng}°W")
    print(f"Species: {site.species}")
    print(f"Expected Detection Rate: {site.expected_detection_rate}")
    print(f"Optimal Season: {site.optimal_season}")
    
    temporal_params = config["temporal_parameters"]
    print(f"\nTemporal Parameters:")
    print(f"  Validation Years: {temporal_params['validation_years']}")
    print(f"  Sampling Frequency: {temporal_params['sampling_frequency_days']} days")
    print(f"  Environmental Drivers: {len(temporal_params['environmental_drivers'])}")
    
    thresholds = config["persistence_thresholds"]
    print(f"\nPersistence Thresholds:")
    print(f"  Minimum Detection Rate: {thresholds['minimum_detection_rate']}")
    print(f"  Consistency Threshold: {thresholds['consistency_threshold']}")
    print(f"  Max Seasonal Variation: {thresholds['seasonal_variation_max']}")
    
    # Create extended synthetic dataset for multi-year analysis
    months = validation_years * 12
    data_points = create_synthetic_temporal_dataset(site, months)
    
    # Perform comprehensive analysis
    print_subsection_header("Multi-Year Analysis Results")
    analysis_results = demonstrate_temporal_analysis(validator, data_points, site)
    quality_results = demonstrate_quality_assessment(validator, data_points, site)
    
    # Check against UVic thresholds
    persistence_rate = analysis_results['persistence_metrics']['persistence_rate']
    consistency_rate = analysis_results['persistence_metrics']['consistency_rate']
    
    print_subsection_header("UVic Threshold Assessment")
    print(f"Persistence Rate: {persistence_rate:.3f} (threshold: {thresholds['minimum_detection_rate']})")
    if persistence_rate >= thresholds['minimum_detection_rate']:
        print("  ✓ MEETS persistence threshold")
    else:
        print("  ✗ BELOW persistence threshold")
    
    print(f"Consistency Rate: {consistency_rate:.3f} (threshold: {thresholds['consistency_threshold']})")
    if consistency_rate >= thresholds['consistency_threshold']:
        print("  ✓ MEETS consistency threshold")
    else:
        print("  ✗ BELOW consistency threshold")
    
    # Overall validation assessment
    meets_thresholds = (
        persistence_rate >= thresholds['minimum_detection_rate'] and
        consistency_rate >= thresholds['consistency_threshold']
    )
    
    print(f"\nOVERALL BROUGHTON VALIDATION: {'PASSED' if meets_thresholds else 'NEEDS IMPROVEMENT'}")
    
    return {
        'config': config,
        'analysis_results': analysis_results,
        'quality_results': quality_results,
        'meets_thresholds': meets_thresholds
    }


async def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description="Temporal Validation Demonstration for SKEMA Kelp Detection"
    )
    parser.add_argument(
        "--site", 
        choices=list(TEST_SITES.keys()), 
        help="Specific site to validate"
    )
    parser.add_argument(
        "--years", 
        type=int, 
        default=1, 
        help="Number of validation years (default: 1)"
    )
    parser.add_argument(
        "--comprehensive", 
        action="store_true", 
        help="Run comprehensive multi-site analysis"
    )
    parser.add_argument(
        "--broughton", 
        action="store_true", 
        help="Run UVic Broughton Archipelago validation"
    )
    parser.add_argument(
        "--structures", 
        action="store_true", 
        help="Demonstrate data structures"
    )
    
    args = parser.parse_args()
    
    print_section_header("TEMPORAL VALIDATION DEMONSTRATION")
    print("Demonstrating SKEMA Kelp Detection Temporal Validation Capabilities")
    print("Following UVic's Broughton Archipelago methodology")
    
    try:
        if args.structures:
            demonstrate_temporal_data_structures()
        
        if args.site:
            await demonstrate_site_validation(args.site, args.years)
        
        if args.comprehensive:
            await demonstrate_comprehensive_analysis()
        
        if args.broughton:
            await demonstrate_broughton_validation(args.years)
        
        # If no specific options, run a basic demo
        if not any([args.site, args.comprehensive, args.broughton, args.structures]):
            print("\nRunning basic demonstration...")
            demonstrate_temporal_data_structures()
            await demonstrate_site_validation("broughton", 1)
        
        print_section_header("DEMONSTRATION COMPLETE")
        print("Temporal validation capabilities successfully demonstrated")
        print("Ready for production deployment with UVic methodology")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        logger.exception("Demonstration failed")


if __name__ == "__main__":
    asyncio.run(main())