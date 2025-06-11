#!/usr/bin/env python3
"""
Interactive Demo: Phase 3 Real Data Acquisition System.

This script demonstrates the comprehensive Phase 3 data acquisition capabilities
for Task C1.5 real-world validation of SKEMA kelp detection algorithms.

Features demonstrated:
- 6 global validation sites across 4 regions
- Realistic Sentinel-2 scene generation
- Comprehensive quality assessment
- Benchmark suite creation and management
- Quality reporting and recommendations

Usage:
    python scripts/test_phase3_data_acquisition_demo.py [mode]
    
    Modes:
    - basic: Basic site information and single dataset creation
    - comprehensive: Full benchmark suite with all sites
    - quality: Quality assessment and reporting demonstration
    - regional: Regional analysis and comparison
    - interactive: Interactive site exploration
"""

import sys
import datetime
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.kelpie_carbon_v1.validation.phase3_data_acquisition import (
    Phase3DataAcquisition,
    ValidationSite,
    ValidationDataset,
    create_phase3_data_acquisition,
    get_validation_sites,
    create_benchmark_dataset,
    create_full_benchmark_suite,
)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")


def format_site_info(site: ValidationSite) -> str:
    """Format site information for display."""
    lat, lon = site.coordinates
    season_start, season_end = site.kelp_season
    
    return f"""
Site ID: {site.site_id}
Name: {site.name}
Location: {lat:.4f}°N, {abs(lon):.4f}°{'E' if lon >= 0 else 'W'}
Species: {site.species}
Region: {site.region}
Kelp Season: {datetime.date(2024, season_start, 1).strftime('%B')} - {datetime.date(2024, season_end, 1).strftime('%B')}
Confidence: {site.validation_confidence.title()}
Data Sources: {', '.join(site.data_sources)}
Notes: {site.notes}
"""


def format_dataset_summary(dataset: ValidationDataset) -> str:
    """Format dataset summary for display."""
    metrics = dataset.quality_metrics
    
    return f"""
Dataset for {dataset.site.name}:
  • Scenes: {len(dataset.satellite_scenes)}
  • Overall Quality: {metrics['overall_quality']:.2f}/1.0
  • Average Cloud Coverage: {metrics['average_cloud_coverage']:.1f}%
  • Temporal Span: {metrics['temporal_span_days']} days
  • Excellent Scenes: {metrics['excellent_scenes_percent']:.1f}%
  • Good or Better: {metrics['good_or_better_percent']:.1f}%
  • Usable Scenes: {metrics['usable_scenes']}/{metrics['total_scenes']}
"""


def demo_basic_functionality():
    """Demonstrate basic Phase 3 data acquisition functionality."""
    print_header("Basic Phase 3 Data Acquisition Demo")
    
    # Initialize the system
    print("Initializing Phase 3 Data Acquisition System...")
    acquisition = create_phase3_data_acquisition()
    
    print(f"✓ Initialized with {len(acquisition.validation_sites)} validation sites")
    
    # Show site summary
    print_subheader("Global Validation Sites Overview")
    summary = acquisition.get_site_summary()
    
    print(f"Total Sites: {summary['total_sites']}")
    print(f"Regions Covered: {summary['coverage_stats']['regions_covered']}")
    print(f"Species Covered: {summary['coverage_stats']['species_covered']}")
    print(f"Total Data Sources: {summary['coverage_stats']['total_data_sources']}")
    
    print("\nRegional Distribution:")
    for region, count in summary['sites_by_region'].items():
        print(f"  • {region}: {count} sites")
    
    print("\nSpecies Distribution:")
    for species, count in summary['sites_by_species'].items():
        print(f"  • {species}: {count} sites")
    
    # Demonstrate single site dataset creation
    print_subheader("Single Site Dataset Creation")
    
    # Use Broughton Archipelago as primary example
    site_id = "broughton_archipelago"
    print(f"Creating validation dataset for {site_id}...")
    
    dataset = create_benchmark_dataset(site_id, num_scenes=8)
    
    print("✓ Dataset created successfully!")
    print(format_dataset_summary(dataset))
    
    # Show some scene details
    print("Sample Scenes:")
    for i, scene in enumerate(dataset.satellite_scenes[:3]):  # Show first 3 scenes
        print(f"  {i+1}. {scene.scene_id}")
        print(f"     Date: {scene.acquisition_date.strftime('%Y-%m-%d')}")
        print(f"     Cloud: {scene.cloud_coverage:.1f}% ({scene.data_quality})")
        print(f"     Season: {scene.metadata.get('season_phase', 'unknown')}")


def demo_comprehensive_benchmark():
    """Demonstrate comprehensive benchmark suite creation."""
    print_header("Comprehensive Benchmark Suite Demo")
    
    print("Creating full benchmark suite across all validation sites...")
    print("This includes 6 sites with 8 scenes each (48 total scenes)")
    
    # Create full benchmark suite
    benchmark_suite = create_full_benchmark_suite(num_scenes_per_site=8)
    
    print(f"✓ Created benchmark suite with {len(benchmark_suite)} datasets")
    
    # Show summary for each site
    print_subheader("Benchmark Suite Summary")
    
    total_scenes = 0
    total_quality = 0.0
    
    for site_id, dataset in benchmark_suite.items():
        print(format_dataset_summary(dataset))
        total_scenes += len(dataset.satellite_scenes)
        total_quality += dataset.quality_metrics['overall_quality']
    
    avg_quality = total_quality / len(benchmark_suite)
    
    print_subheader("Overall Benchmark Statistics")
    print(f"Total Datasets: {len(benchmark_suite)}")
    print(f"Total Scenes: {total_scenes}")
    print(f"Average Quality: {avg_quality:.3f}/1.0")
    
    # Find best and worst quality sites
    best_site = max(benchmark_suite.items(), key=lambda x: x[1].quality_metrics['overall_quality'])
    worst_site = min(benchmark_suite.items(), key=lambda x: x[1].quality_metrics['overall_quality'])
    
    print(f"Highest Quality: {best_site[0]} ({best_site[1].quality_metrics['overall_quality']:.3f})")
    print(f"Lowest Quality: {worst_site[0]} ({worst_site[1].quality_metrics['overall_quality']:.3f})")


def demo_quality_assessment():
    """Demonstrate quality assessment and reporting capabilities."""
    print_header("Quality Assessment & Reporting Demo")
    
    acquisition = create_phase3_data_acquisition()
    
    # Create datasets with different scene counts for quality comparison
    print("Creating datasets with varying scene counts for quality analysis...")
    
    test_datasets = {}
    scene_counts = [5, 10, 15]
    
    for num_scenes in scene_counts:
        print(f"  Creating {num_scenes}-scene dataset for Monterey Bay...")
        dataset = acquisition.create_validation_dataset("monterey_bay", num_scenes)
        test_datasets[f"monterey_bay_{num_scenes}"] = dataset
    
    # Generate comprehensive quality report
    print_subheader("Quality Report Generation")
    
    report = acquisition.generate_quality_report(test_datasets)
    
    print("Quality Report Summary:")
    print(f"  • Total Datasets: {report['report_metadata']['total_datasets']}")
    print(f"  • Total Scenes: {report['report_metadata']['total_scenes']}")
    print(f"  • Average Quality: {report['overall_quality']['average_quality']:.3f}")
    print(f"  • Average Cloud Coverage: {report['overall_quality']['average_cloud_coverage']:.1f}%")
    
    print("\nQuality Distribution:")
    for quality, count in report['quality_distribution'].items():
        percentage = count / report['report_metadata']['total_scenes'] * 100
        print(f"  • {quality.title()}: {count} scenes ({percentage:.1f}%)")
    
    print("\nSite-Specific Quality:")
    for site_id, site_quality in report['site_quality'].items():
        print(f"  • {site_id}: {site_quality['overall_quality']:.3f} quality")
        print(f"    - {site_quality['usable_scenes']}/{site_quality['scene_count']} usable scenes")
        print(f"    - Recommendation: {site_quality['recommendation']}")
    
    print("\nGeneral Recommendations:")
    for recommendation in report['recommendations']:
        print(f"  • {recommendation}")


def demo_regional_analysis():
    """Demonstrate regional analysis and comparison."""
    print_header("Regional Analysis Demo")
    
    acquisition = create_phase3_data_acquisition()
    
    # Analyze by region
    regions = ["British Columbia", "California", "Washington", "Tasmania"]
    
    regional_data = {}
    
    for region in regions:
        print_subheader(f"{region} Analysis")
        
        # Get sites in this region
        regional_sites = acquisition.get_validation_sites(region=region)
        
        if not regional_sites:
            print(f"  No sites found in {region}")
            continue
        
        print(f"  Sites in {region}: {len(regional_sites)}")
        
        # Create datasets for all sites in region
        regional_datasets = {}
        total_quality = 0.0
        total_cloud = 0.0
        
        for site in regional_sites:
            print(f"    • {site.name} ({site.species})")
            dataset = acquisition.create_validation_dataset(site.site_id, 6)
            regional_datasets[site.site_id] = dataset
            
            total_quality += dataset.quality_metrics['overall_quality']
            total_cloud += dataset.quality_metrics['average_cloud_coverage']
        
        # Regional statistics
        avg_quality = total_quality / len(regional_sites)
        avg_cloud = total_cloud / len(regional_sites)
        
        regional_data[region] = {
            'avg_quality': avg_quality,
            'avg_cloud': avg_cloud,
            'site_count': len(regional_sites),
            'datasets': regional_datasets
        }
        
        print(f"    Average Quality: {avg_quality:.3f}")
        print(f"    Average Cloud Coverage: {avg_cloud:.1f}%")
    
    # Regional comparison
    print_subheader("Regional Comparison")
    
    print("Quality Ranking by Region:")
    sorted_regions = sorted(regional_data.items(), key=lambda x: x[1]['avg_quality'], reverse=True)
    
    for i, (region, data) in enumerate(sorted_regions, 1):
        print(f"  {i}. {region}: {data['avg_quality']:.3f} quality, {data['avg_cloud']:.1f}% cloud")
    
    # Species analysis
    print_subheader("Species Distribution Analysis")
    
    all_sites = acquisition.get_validation_sites()
    species_count = {}
    
    for site in all_sites:
        species = site.species
        if species not in species_count:
            species_count[species] = 0
        species_count[species] += 1
    
    print("Species Coverage:")
    for species, count in sorted(species_count.items()):
        print(f"  • {species}: {count} sites")


def demo_interactive_exploration():
    """Interactive site exploration demo."""
    print_header("Interactive Site Exploration")
    
    acquisition = create_phase3_data_acquisition()
    all_sites = acquisition.get_validation_sites()
    
    print(f"Available validation sites ({len(all_sites)} total):")
    for i, site in enumerate(all_sites, 1):
        print(f"  {i}. {site.name} ({site.region})")
    
    while True:
        try:
            choice = input(f"\nSelect a site to explore (1-{len(all_sites)}, 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                break
            
            site_index = int(choice) - 1
            if 0 <= site_index < len(all_sites):
                selected_site = all_sites[site_index]
                
                print_subheader(f"Exploring {selected_site.name}")
                print(format_site_info(selected_site))
                
                # Ask if user wants to create a dataset
                create_dataset = input("Create validation dataset for this site? (y/n): ").strip().lower()
                
                if create_dataset == 'y':
                    num_scenes = input("Number of scenes (default 8): ").strip()
                    num_scenes = int(num_scenes) if num_scenes.isdigit() else 8
                    
                    print(f"Creating dataset with {num_scenes} scenes...")
                    dataset = acquisition.create_validation_dataset(selected_site.site_id, num_scenes)
                    
                    print(format_dataset_summary(dataset))
                    
                    # Option to save dataset
                    save_dataset = input("Save dataset to file? (y/n): ").strip().lower()
                    if save_dataset == 'y':
                        filepath = acquisition.save_validation_dataset(dataset)
                        print(f"✓ Dataset saved to: {filepath}")
            else:
                print("Invalid selection. Please try again.")
                
        except (ValueError, KeyboardInterrupt):
            print("\nExiting interactive exploration.")
            break


def main():
    """Main demonstration function."""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "basic"
    
    print(f"Phase 3 Data Acquisition Demo - Mode: {mode}")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if mode == "basic":
            demo_basic_functionality()
        elif mode == "comprehensive":
            demo_comprehensive_benchmark()
        elif mode == "quality":
            demo_quality_assessment()
        elif mode == "regional":
            demo_regional_analysis()
        elif mode == "interactive":
            demo_interactive_exploration()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: basic, comprehensive, quality, regional, interactive")
            return 1
        
        print("\n" + "=" * 80)
        print("  Demo completed successfully!")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 