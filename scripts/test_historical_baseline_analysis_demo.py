#!/usr/bin/env python3
"""
Historical Baseline Analysis Demonstration Script

This script demonstrates the comprehensive historical baseline analysis capabilities
for kelp detection, including:
- Historical site creation and data digitization
- Change detection algorithms
- Temporal trend analysis
- Risk assessment and forecasting
- Comparative analysis across multiple sites

Usage:
    python scripts/test_historical_baseline_analysis_demo.py [mode]
    
Modes:
    basic       - Basic historical analysis workflow
    uvic        - UVic-specific sites and methodology
    comparison  - Multi-site comparative analysis
    interactive - Interactive site exploration
    all         - Run all demonstration modes
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import tempfile
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kelpie_carbon_v1.validation.historical_baseline_analysis import (
    HistoricalBaselineAnalysis,
    create_uvic_historical_sites,
    create_sample_historical_dataset
)

def create_demo_data():
    """Create realistic demonstration data for various historical sites."""
    demo_sites = {}
    
    # 1. Broughton Archipelago - Declining trend due to warming
    broughton_data = {}
    np.random.seed(42)
    for year in range(1858, 1957, 3):  # Every 3 years
        # Base decline due to warming + random variation + cyclical pattern
        base_extent = 180.0  # Starting extent
        warming_trend = (year - 1858) * 0.4  # 0.4 ha/year decline
        cyclical = 15 * np.sin(2 * np.pi * (year - 1858) / 20)  # 20-year cycle
        noise = np.random.normal(0, 8)  # Random variation
        
        extent = max(0, base_extent - warming_trend + cyclical + noise)
        
        broughton_data[year] = {
            "extent": round(extent, 1),
            "confidence": np.random.uniform(0.75, 0.95),
            "source": f"Admiralty_Chart_{year}",
            "notes": f"Broughton Archipelago kelp extent for {year}"
        }
    
    demo_sites["broughton"] = broughton_data
    
    # 2. Monterey Bay - More stable with periodic El Niño effects
    monterey_data = {}
    np.random.seed(123)
    for year in range(1851, 1951, 4):  # Every 4 years
        base_extent = 220.0
        mild_decline = (year - 1851) * 0.15  # Mild decline
        el_nino_cycle = 25 * np.sin(2 * np.pi * (year - 1851) / 15)  # 15-year El Niño
        noise = np.random.normal(0, 12)
        
        extent = max(0, base_extent - mild_decline + el_nino_cycle + noise)
        
        monterey_data[year] = {
            "extent": round(extent, 1),
            "confidence": np.random.uniform(0.7, 0.9),
            "source": f"US_Coast_Survey_{year}",
            "notes": f"Monterey Bay giant kelp extent for {year}"
        }
    
    demo_sites["monterey"] = monterey_data
    
    # 3. Tasmania - Recovery pattern after initial decline
    tasmania_data = {}
    np.random.seed(456)
    for year in range(1860, 1961, 5):  # Every 5 years
        base_extent = 95.0
        
        # Initial decline then recovery
        if year < 1900:
            trend = (year - 1860) * 0.5  # Initial decline
        else:
            trend = (1900 - 1860) * 0.5 - (year - 1900) * 0.2  # Recovery
        
        seasonal = 8 * np.cos(2 * np.pi * (year - 1860) / 12)  # 12-year pattern
        noise = np.random.normal(0, 6)
        
        extent = max(0, base_extent - trend + seasonal + noise)
        
        tasmania_data[year] = {
            "extent": round(extent, 1),
            "confidence": np.random.uniform(0.65, 0.85),
            "source": f"Australian_Survey_{year}",
            "notes": f"Tasmania kelp extent for {year}"
        }
    
    demo_sites["tasmania"] = tasmania_data
    
    return demo_sites

def demo_basic_workflow():
    """Demonstrate basic historical baseline analysis workflow."""
    print("🔬 Basic Historical Baseline Analysis Workflow")
    print("=" * 60)
    
    analyzer = HistoricalBaselineAnalysis()
    
    # 1. Create historical site
    print("\n1. Creating historical site...")
    site = analyzer.create_historical_site(
        name="Demo Kelp Site",
        latitude=50.0833,
        longitude=-126.1667,
        region="British Columbia",
        historical_period=(1858, 1956),
        data_sources=["Admiralty Charts", "Hydrographic Surveys"],
        species=["Nereocystis luetkeana", "Macrocystis pyrifera"],
        digitization_quality="high",
        notes="Demonstration site for historical analysis"
    )
    
    print(f"   Created site: {site.name}")
    print(f"   Region: {site.region}")
    print(f"   Historical period: {site.historical_period[0]}-{site.historical_period[1]}")
    print(f"   Species: {', '.join(site.species)}")
    
    # 2. Generate and digitize historical data
    print("\n2. Digitizing historical data...")
    demo_data = create_demo_data()
    broughton_data = demo_data["broughton"]
    
    dataset = analyzer.digitize_historical_data("Demo Kelp Site", broughton_data)
    
    print(f"   Digitized {len(dataset.temporal_data)} historical observations")
    print(f"   Baseline extent: {dataset.baseline_extent:.1f} hectares")
    print(f"   Data quality metrics:")
    for metric, value in dataset.data_quality_metrics.items():
        print(f"     {metric}: {value:.3f}")
    
    # 3. Perform comprehensive analysis
    print("\n3. Performing comprehensive analysis...")
    current_extent = 85.0  # Current measurement
    current_year = 2024
    
    analysis = analyzer.perform_comprehensive_analysis(
        "Demo Kelp Site",
        current_extent=current_extent,
        current_year=current_year,
        analysis_options={"include_forecast": True}
    )
    
    # Display results
    print(f"   Analysis completed for {analysis['site_information']['name']}")
    
    # Change detection results
    change_detection = analysis["change_detection"]
    print(f"\n   📊 Change Detection Results:")
    print(f"     Historical mean: {change_detection['historical_mean']:.1f} ha")
    print(f"     Current extent: {current_extent:.1f} ha")
    print(f"     Relative change: {change_detection['relative_change_percent']:.1f}%")
    print(f"     Statistical significance: {'Yes' if change_detection['is_significant'] else 'No'}")
    print(f"     Change magnitude: {change_detection['change_magnitude']}")
    
    # Trend analysis results
    trends = analysis["temporal_trends"]
    if "trend_metrics" in trends and "linear_trend" in trends["trend_metrics"]:
        linear_trend = trends["trend_metrics"]["linear_trend"]
        print(f"\n   📈 Trend Analysis:")
        print(f"     Trend slope: {linear_trend['slope']:.3f} ha/year")
        print(f"     R-squared: {linear_trend['r_squared']:.3f}")
        print(f"     Trend direction: {trends['trend_metrics'].get('trend_direction', 'unknown')}")
    
    # Risk assessment
    if "risk_assessment" in trends:
        risk = trends["risk_assessment"]
        print(f"\n   ⚠️  Risk Assessment:")
        print(f"     Overall risk level: {risk['overall_risk_level'].upper()}")
        if risk["risk_factors"]:
            print(f"     Risk factors:")
            for factor in risk["risk_factors"]:
                print(f"       - {factor}")
        if risk["recommendations"]:
            print(f"     Recommendations:")
            for rec in risk["recommendations"]:
                print(f"       - {rec}")
    
    # Forecast results
    if "forecast" in trends and trends["forecast"]:
        forecast = trends["forecast"]
        print(f"\n   🔮 Forecast (next 5 years):")
        print(f"     Method: {forecast['method']}")
        if "predicted_extents" in forecast:
            for i, (year, extent) in enumerate(zip(forecast["forecast_years"][:5], forecast["predicted_extents"][:5])):
                conf_interval = forecast["confidence_intervals_95"][i]
                print(f"     {year}: {extent:.1f} ha (95% CI: {conf_interval[0]:.1f}-{conf_interval[1]:.1f})")
    
    return analyzer

def demo_uvic_methodology():
    """Demonstrate UVic-specific historical sites and methodology."""
    print("\n🏫 UVic Historical Methodology Demonstration")
    print("=" * 60)
    
    # Create UVic historical sites
    print("\n1. Creating UVic historical sites...")
    sites = create_uvic_historical_sites()
    
    for site_id, site in sites.items():
        print(f"   {site_id.title()}: {site.name}")
        print(f"     Location: {site.latitude:.4f}°N, {site.longitude:.4f}°W")
        print(f"     Region: {site.region}")
        print(f"     Historical period: {site.historical_period[0]}-{site.historical_period[1]}")
        print(f"     Species: {', '.join(site.species)}")
        print(f"     Data quality: {site.digitization_quality}")
        print()
    
    # Use the sample dataset
    print("2. Creating sample historical dataset...")
    sample_dataset = create_sample_historical_dataset()
    
    print(f"   Sample site: {sample_dataset.site.name}")
    print(f"   Data points: {len(sample_dataset.temporal_data)}")
    print(f"   Baseline extent: {sample_dataset.baseline_extent:.1f} hectares")
    print(f"   Years covered: {min(sample_dataset.temporal_data.keys())}-{max(sample_dataset.temporal_data.keys())}")
    
    # Show sample data characteristics
    extents = [data["extent"] for data in sample_dataset.temporal_data.values()]
    confidences = [data["confidence"] for data in sample_dataset.temporal_data.values()]
    
    print(f"\n   📊 Dataset characteristics:")
    print(f"     Extent range: {min(extents):.1f} - {max(extents):.1f} hectares")
    print(f"     Mean extent: {np.mean(extents):.1f} hectares")
    print(f"     Mean confidence: {np.mean(confidences):.3f}")
    print(f"     Data completeness: {sample_dataset.data_quality_metrics['data_completeness']:.3f}")
    
    return sites, sample_dataset

def demo_multi_site_comparison():
    """Demonstrate multi-site comparative analysis."""
    print("\n🌍 Multi-Site Comparative Analysis")
    print("=" * 60)
    
    analyzer = HistoricalBaselineAnalysis()
    demo_data = create_demo_data()
    
    # Create multiple sites with different characteristics
    sites_config = [
        {
            "name": "Broughton Archipelago",
            "latitude": 50.0833,
            "longitude": -126.1667,
            "region": "British Columbia",
            "species": ["Nereocystis luetkeana"],
            "data_key": "broughton"
        },
        {
            "name": "Monterey Bay",
            "latitude": 36.8000,
            "longitude": -121.9000,
            "region": "California",
            "species": ["Macrocystis pyrifera"],
            "data_key": "monterey"
        },
        {
            "name": "Tasmania Coast",
            "latitude": -42.8821,
            "longitude": 147.3272,
            "region": "Tasmania",
            "species": ["Macrocystis pyrifera"],
            "data_key": "tasmania"
        }
    ]
    
    # Create sites and digitize data
    print("1. Creating multiple historical sites...")
    site_names = []
    
    for config in sites_config:
        print(f"   Creating {config['name']}...")
        
        analyzer.create_historical_site(
            name=config["name"],
            latitude=config["latitude"],
            longitude=config["longitude"],
            region=config["region"],
            historical_period=(1850, 1960),
            data_sources=["Historical Charts"],
            species=config["species"]
        )
        
        # Digitize corresponding data
        site_data = demo_data[config["data_key"]]
        dataset = analyzer.digitize_historical_data(config["name"], site_data)
        
        site_names.append(config["name"])
        
        print(f"     Data points: {len(dataset.temporal_data)}")
        print(f"     Baseline extent: {dataset.baseline_extent:.1f} ha")
    
    # Generate comparative report
    print("\n2. Generating comparative analysis report...")
    report = analyzer.generate_comparison_report(site_names, output_format="dict")
    
    # Display summary statistics
    summary = report["summary_statistics"]
    print(f"\n   📊 Comparative Summary:")
    print(f"     Total sites analyzed: {summary['total_sites']}")
    print(f"     Regional diversity: {summary['regional_diversity']} regions")
    print(f"     Species diversity: {summary['species_diversity']} species groups")
    print(f"     Baseline extent range: {summary['baseline_extent_range'][0]:.1f} - {summary['baseline_extent_range'][1]:.1f} ha")
    
    # Display individual site comparisons
    print(f"\n   🏞️  Individual Site Analysis:")
    for site_name, stats in report["site_comparisons"].items():
        print(f"\n     {site_name}:")
        print(f"       Region: {stats['region']}")
        print(f"       Species: {', '.join(stats['species'])}")
        print(f"       Baseline extent: {stats['baseline_extent']:.1f} ha")
        print(f"       Mean historical extent: {stats['mean_extent']:.1f} ha")
        print(f"       Temporal span: {stats['temporal_span'][0]}-{stats['temporal_span'][1]}")
        print(f"       Data quality score: {stats['data_quality_score']:.3f}")
        
        # Calculate trend
        extent_range = stats['extent_range']
        if extent_range[1] > extent_range[0]:
            trend = "Increasing"
        elif extent_range[1] < extent_range[0]:
            trend = "Decreasing"
        else:
            trend = "Stable"
        print(f"       Overall trend: {trend}")
    
    # Perform individual analyses for comparison
    print(f"\n   🔍 Detailed Analysis Comparison:")
    current_year = 2024
    current_extents = {"Broughton Archipelago": 85.0, "Monterey Bay": 190.0, "Tasmania Coast": 75.0}
    
    for site_name in site_names:
        print(f"\n     {site_name}:")
        
        analysis = analyzer.perform_comprehensive_analysis(
            site_name,
            current_extent=current_extents[site_name],
            current_year=current_year
        )
        
        change_detection = analysis["change_detection"]
        print(f"       Current vs historical: {change_detection['relative_change_percent']:.1f}%")
        print(f"       Change significance: {'Yes' if change_detection['is_significant'] else 'No'}")
        
        trends = analysis["temporal_trends"]
        if "risk_assessment" in trends:
            risk_level = trends["risk_assessment"]["overall_risk_level"]
            print(f"       Risk level: {risk_level.upper()}")
    
    return analyzer, report

def demo_interactive_exploration():
    """Interactive exploration of historical analysis capabilities."""
    print("\n🎮 Interactive Historical Analysis Exploration")
    print("=" * 60)
    
    analyzer = HistoricalBaselineAnalysis()
    demo_data = create_demo_data()
    
    # Present options
    print("\nAvailable demonstration scenarios:")
    print("1. Declining kelp forest (Broughton Archipelago)")
    print("2. Stable kelp forest with cycles (Monterey Bay)")
    print("3. Recovery pattern (Tasmania)")
    print("4. Custom analysis")
    
    try:
        choice = input("\nSelect scenario (1-4): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"  # Default for non-interactive environments
    
    if choice == "1":
        site_name = "Broughton Declining Demo"
        data_key = "broughton"
        current_extent = 85.0
        description = "Shows a declining kelp forest trend due to warming waters"
        
    elif choice == "2":
        site_name = "Monterey Stable Demo"
        data_key = "monterey"
        current_extent = 190.0
        description = "Shows a relatively stable forest with El Niño cyclical effects"
        
    elif choice == "3":
        site_name = "Tasmania Recovery Demo"
        data_key = "tasmania"
        current_extent = 75.0
        description = "Shows initial decline followed by recovery pattern"
        
    else:
        site_name = "Custom Demo"
        data_key = "broughton"  # Default
        current_extent = 100.0
        description = "Custom analysis with default parameters"
    
    print(f"\n🔬 Analyzing: {site_name}")
    print(f"📝 Description: {description}")
    
    # Create and analyze site
    analyzer.create_historical_site(
        name=site_name,
        latitude=50.0,
        longitude=-125.0,
        region="Demo Region",
        historical_period=(1850, 1960),
        data_sources=["Demo Data"],
        species=["Demo Species"]
    )
    
    site_data = demo_data[data_key]
    dataset = analyzer.digitize_historical_data(site_name, site_data)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Observations: {len(dataset.temporal_data)}")
    print(f"   Time range: {min(dataset.temporal_data.keys())}-{max(dataset.temporal_data.keys())}")
    print(f"   Baseline extent: {dataset.baseline_extent:.1f} hectares")
    
    # Perform analysis
    analysis = analyzer.perform_comprehensive_analysis(
        site_name,
        current_extent=current_extent,
        current_year=2024,
        analysis_options={"include_forecast": True}
    )
    
    # Display comprehensive results
    print(f"\n📈 Analysis Results:")
    
    # Change detection
    change = analysis["change_detection"]
    print(f"   Historical mean: {change['historical_mean']:.1f} ha")
    print(f"   Current extent: {current_extent:.1f} ha")
    print(f"   Overall change: {change['relative_change_percent']:.1f}%")
    print(f"   Change significance: {'SIGNIFICANT' if change['is_significant'] else 'Not significant'}")
    
    # Pattern analysis
    patterns = analysis["change_patterns"]
    if "trend_analysis" in patterns:
        trend = patterns["trend_analysis"]
        print(f"   Trend direction: {trend['trend_direction'].upper()}")
        print(f"   Trend significance: {'SIGNIFICANT' if trend['trend_significance'] == 'significant' else 'Not significant'}")
    
    # Risk assessment
    trends = analysis["temporal_trends"]
    if "risk_assessment" in trends:
        risk = trends["risk_assessment"]
        print(f"   Risk level: {risk['overall_risk_level'].upper()}")
        
        if risk["risk_factors"]:
            print(f"   Risk factors: {', '.join(risk['risk_factors'])}")
    
    # Forecast
    if "forecast" in trends and trends["forecast"]:
        forecast = trends["forecast"]
        print(f"   Next 5-year forecast:")
        for i, (year, extent) in enumerate(zip(forecast["forecast_years"][:5], forecast["predicted_extents"][:5])):
            print(f"     {year}: {extent:.1f} ha")
    
    # Export option
    print(f"\n💾 Export Options:")
    print("   Results saved to temporary location for review")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / f"{site_name.replace(' ', '_')}_analysis"
        analyzer.export_results(site_name, output_path)
        print(f"   Exported to: {output_path}")
    
    return analyzer

def demo_advanced_features():
    """Demonstrate advanced analysis features."""
    print("\n🚀 Advanced Historical Analysis Features")
    print("=" * 60)
    
    analyzer = HistoricalBaselineAnalysis()
    
    # 1. Quality control demonstration
    print("\n1. Quality Control Procedures:")
    
    # Create test data with various quality issues
    problematic_data = {
        1850: {"extent": "150.0", "confidence": 0.9, "source": "Good"},
        1860: {"extent": "invalid_data", "confidence": 0.8, "source": "Corrupted"},  # Invalid
        1870: {"extent": "140.0", "confidence": 0.3, "source": "Poor quality"},     # Low confidence
        1880: {"extent": "2000.0", "confidence": 0.9, "source": "Outlier"},        # Extreme outlier
        1890: {"extent": "130.0", "confidence": 0.85, "source": "Good"},
        1900: {"extent": "125.0", "confidence": 0.9, "source": "Excellent"}
    }
    
    analyzer.create_historical_site(
        name="Quality Control Demo",
        latitude=50.0,
        longitude=-125.0,
        region="Demo",
        historical_period=(1850, 1950),
        data_sources=["Test"],
        species=["Test"]
    )
    
    # Apply quality control
    qc_params = {"min_confidence": 0.6, "max_extent_change": 3.0}
    processed = analyzer._apply_quality_control(problematic_data, qc_params)
    
    print(f"   Original data points: {len(problematic_data)}")
    print(f"   After quality control: {len(processed)}")
    print(f"   Filtered out: {len(problematic_data) - len(processed)} problematic records")
    
    for year in problematic_data:
        if year not in processed:
            reason = "Unknown"
            if problematic_data[year]["extent"] == "invalid_data":
                reason = "Invalid extent data"
            elif problematic_data[year]["confidence"] < 0.6:
                reason = "Low confidence"
            elif float(problematic_data[year]["extent"]) > 1000:
                reason = "Extreme outlier"
            print(f"     Excluded {year}: {reason}")
    
    # 2. Statistical method comparison
    print("\n2. Statistical Method Comparison:")
    
    # Create comparison dataset
    historical_data = {year: 150 - (year - 1850) * 0.5 for year in range(1850, 1901, 10)}
    current_data = {2024: 100.0}
    
    change_analyzer = analyzer.change_analyzer
    
    methods = ["mann_kendall", "t_test", "wilcoxon"]
    print(f"   Testing change detection methods:")
    
    for method in methods:
        result = change_analyzer.detect_significant_changes(
            historical_data, current_data, method=method
        )
        
        if "error" not in result:
            print(f"     {method.upper()}:")
            print(f"       Test statistic: {result['test_statistic']:.3f}")
            print(f"       P-value: {result['p_value']:.4f}")
            print(f"       Significant: {'Yes' if result['is_significant'] else 'No'}")
    
    # 3. Export format demonstration
    print("\n3. Export Format Options:")
    
    # Create sample dataset for export
    analyzer.digitize_historical_data("Quality Control Demo", processed)
    
    # Generate different export formats
    sites = ["Quality Control Demo"]
    
    dict_report = analyzer.generate_comparison_report(sites, output_format="dict")
    json_report = analyzer.generate_comparison_report(sites, output_format="json")
    markdown_report = analyzer.generate_comparison_report(sites, output_format="markdown")
    
    print(f"   Dictionary format: {type(dict_report).__name__} with {len(dict_report)} sections")
    print(f"   JSON format: {len(json_report)} characters")
    print(f"   Markdown format: {len(markdown_report)} characters")
    print(f"   Markdown preview:")
    print("     " + "\n     ".join(markdown_report.split("\n")[:5]))
    
    return analyzer

def main():
    """Main demonstration function."""
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    print("🌊 Historical Baseline Analysis for Kelp Detection")
    print("Kelpie Carbon v1 - Task D1 Implementation Demo")
    print("=" * 80)
    
    # Parse command line arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "basic"
    
    if mode == "basic" or mode == "all":
        demo_basic_workflow()
    
    if mode == "uvic" or mode == "all":
        demo_uvic_methodology()
    
    if mode == "comparison" or mode == "all":
        demo_multi_site_comparison()
    
    if mode == "interactive" or mode == "all":
        demo_interactive_exploration()
    
    if mode == "advanced" or mode == "all":
        demo_advanced_features()
    
    print("\n" + "=" * 80)
    print("✅ Historical Baseline Analysis Demonstration Complete!")
    print("\nKey Features Demonstrated:")
    print("  📊 Historical data digitization and quality control")
    print("  📈 Statistical change detection (Mann-Kendall, t-test, Wilcoxon)")
    print("  🔮 Temporal trend analysis and forecasting")
    print("  ⚠️  Risk assessment and management recommendations")
    print("  🌍 Multi-site comparative analysis")
    print("  📁 Multiple export formats (JSON, Markdown, visualizations)")
    print("  🎯 UVic methodology compliance for kelp research")
    
    print(f"\nThis implementation provides comprehensive historical baseline")
    print(f"analysis capabilities for kelp detection following UVic research")
    print(f"methodology, enabling comparison of current kelp extent against")
    print(f"historical baselines from 1858-1956 chart data.")

if __name__ == "__main__":
    main() 