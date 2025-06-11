#!/usr/bin/env python3
"""
Environmental Robustness Testing Script for SKEMA Kelp Detection.

This script runs comprehensive environmental condition testing to validate
the robustness of SKEMA kelp detection algorithms across various real-world
conditions including tidal effects, water clarity, and seasonal variations.

Usage:
    python scripts/run_environmental_testing.py --site broughton --date 2023-07-15
    python scripts/run_environmental_testing.py --mode comprehensive --output results/
    python scripts/run_environmental_testing.py --help
"""

import asyncio
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from kelpie_carbon_v1.validation.environmental_testing import (
    EnvironmentalRobustnessValidator,
    validate_tidal_effects,
)
from kelpie_carbon_v1.logging_config import setup_logging

# Define validation sites for environmental testing
VALIDATION_SITES = {
    "broughton": {
        "name": "Broughton Archipelago",
        "lat": 50.0833,
        "lng": -126.1667,
        "description": "UVic primary SKEMA site - Bull kelp with strong tidal currents",
    },
    "saanich": {
        "name": "Saanich Inlet",
        "lat": 48.5830,
        "lng": -123.5000,
        "description": "Sheltered inlet - Multi-species kelp validation",
    },
    "monterey": {
        "name": "Monterey Bay",
        "lat": 36.8000,
        "lng": -121.9000,
        "description": "Giant kelp forest - Clear water conditions",
    },
}

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run environmental robustness testing for SKEMA kelp detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all conditions at Broughton Archipelago
  python scripts/run_environmental_testing.py --site broughton --date 2023-07-15
  
  # Run comprehensive testing across all sites
  python scripts/run_environmental_testing.py --mode comprehensive
  
  # Test only tidal effects
  python scripts/run_environmental_testing.py --mode tidal --site monterey
  
  # Save results to specific directory
  python scripts/run_environmental_testing.py --site saanich --output results/environmental/
        """
    )
    
    parser.add_argument(
        "--site",
        choices=list(VALIDATION_SITES.keys()),
        help="Validation site to test (default: broughton)",
        default="broughton"
    )
    
    parser.add_argument(
        "--date",
        type=str,
        help="Base date for testing (YYYY-MM-DD, default: 2023-07-15)",
        default="2023-07-15"
    )
    
    parser.add_argument(
        "--mode",
        choices=["comprehensive", "tidal", "clarity", "seasonal"],
        help="Testing mode (default: comprehensive)",
        default="comprehensive"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for results (default: environmental_results/)",
        default="environmental_results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


async def run_comprehensive_testing(site_info, base_date, output_dir):
    """Run comprehensive environmental testing for a site."""
    logger.info(f"🌊 Starting comprehensive environmental testing for {site_info['name']}")
    
    validator = EnvironmentalRobustnessValidator()
    
    # Run comprehensive testing
    results = await validator.run_comprehensive_testing(
        lat=site_info["lat"],
        lng=site_info["lng"],
        base_date=base_date
    )
    
    # Save results
    output_file = Path(output_dir) / f"comprehensive_{site_info['name'].lower().replace(' ', '_')}_{base_date}.json"
    save_results(results, output_file)
    
    # Print summary
    print_results_summary(results, site_info["name"])
    
    return results


async def run_tidal_testing(site_info, base_date, output_dir):
    """Run tidal effects testing for a site."""
    logger.info(f"🌊 Starting tidal effects testing for {site_info['name']}")
    
    # Run tidal effects testing
    results = await validate_tidal_effects(
        lat=site_info["lat"],
        lng=site_info["lng"],
        base_date=base_date
    )
    
    # Save results
    output_file = Path(output_dir) / f"tidal_{site_info['name'].lower().replace(' ', '_')}_{base_date}.json"
    save_results(results, output_file)
    
    # Print summary
    print_results_summary(results, f"{site_info['name']} (Tidal Effects)")
    
    return results


async def run_all_sites_testing(mode, base_date, output_dir):
    """Run testing across all validation sites."""
    logger.info(f"🌍 Starting {mode} testing across all validation sites")
    
    all_results = {}
    
    for site_key, site_info in VALIDATION_SITES.items():
        logger.info(f"Testing site: {site_info['name']}")
        
        if mode == "comprehensive":
            results = await run_comprehensive_testing(site_info, base_date, output_dir)
        elif mode == "tidal":
            results = await run_tidal_testing(site_info, base_date, output_dir)
        # Add more modes as needed
        
        all_results[site_key] = results
    
    # Save combined results
    combined_file = Path(output_dir) / f"all_sites_{mode}_{base_date}.json"
    save_results(all_results, combined_file)
    
    # Print overall summary
    print_overall_summary(all_results)
    
    return all_results


def save_results(results, output_file):
    """Save results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"📁 Results saved to: {output_file}")


def print_results_summary(results, site_name):
    """Print a summary of test results."""
    print(f"\n{'='*60}")
    print(f"🧪 Environmental Testing Results: {site_name}")
    print(f"{'='*60}")
    
    if "summary" in results:
        summary = results["summary"]
        print(f"📊 Total conditions tested: {summary['total_conditions']}")
        print(f"✅ Successful tests: {summary['successful_tests']}")
        print(f"📈 Success rate: {summary['success_rate']*100:.1f}%")
        
        if "detailed_results" in results:
            print(f"\n📋 Detailed Results:")
            for result in results["detailed_results"]:
                status = "✅ PASS" if result["success"] else "❌ FAIL"
                print(f"  {status} {result['condition_name']}: {result['detection_rate']*100:.1f}% detection")
    
    print(f"{'='*60}\n")


def print_overall_summary(all_results):
    """Print overall summary across all sites."""
    print(f"\n{'='*80}")
    print(f"🌍 Overall Environmental Testing Summary")
    print(f"{'='*80}")
    
    total_tests = 0
    total_successful = 0
    
    for site_key, results in all_results.items():
        site_name = VALIDATION_SITES[site_key]["name"]
        if "summary" in results:
            summary = results["summary"]
            tests = summary["total_conditions"]
            successful = summary["successful_tests"]
            success_rate = summary["success_rate"]
            
            total_tests += tests
            total_successful += successful
            
            print(f"📍 {site_name}: {successful}/{tests} tests passed ({success_rate*100:.1f}%)")
    
    overall_success_rate = total_successful / total_tests if total_tests > 0 else 0
    print(f"\n🎯 Overall Success Rate: {total_successful}/{total_tests} ({overall_success_rate*100:.1f}%)")
    
    if overall_success_rate >= 0.8:
        print("🎉 EXCELLENT: Environmental robustness validation successful!")
    elif overall_success_rate >= 0.6:
        print("⚠️  GOOD: Some environmental conditions need attention")
    else:
        print("🚨 NEEDS WORK: Environmental robustness requires improvement")
    
    print(f"{'='*80}\n")


async def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging()
    
    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🚀 Starting Environmental Robustness Testing")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Base date: {args.date}")
    logger.info(f"Output directory: {args.output}")
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    try:
        if args.site and args.mode != "comprehensive":
            # Test specific site
            site_info = VALIDATION_SITES[args.site]
            logger.info(f"Testing specific site: {site_info['name']}")
            
            if args.mode == "tidal":
                await run_tidal_testing(site_info, args.date, args.output)
            else:
                await run_comprehensive_testing(site_info, args.date, args.output)
        
        elif args.mode == "comprehensive":
            # Test all sites comprehensively
            await run_all_sites_testing(args.mode, args.date, args.output)
        
        else:
            # Test specific site with comprehensive mode
            site_info = VALIDATION_SITES[args.site]
            await run_comprehensive_testing(site_info, args.date, args.output)
        
        logger.info("✅ Environmental testing completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Environmental testing failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 