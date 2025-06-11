#!/usr/bin/env python3
"""
Real-world validation script for Task A2.5: Primary Validation Site Testing.

This script runs comprehensive validation of SKEMA kelp detection algorithms
against actual satellite imagery from validated kelp farm locations.

Usage:
    python scripts/run_real_world_validation.py --mode primary --days 30
    python scripts/run_real_world_validation.py --mode full --days 14 --output results/
    python scripts/run_real_world_validation.py --mode controls --days 7
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kelpie_carbon_v1.validation.real_world_validation import (
    RealWorldValidator,
    validate_primary_sites,
    validate_with_controls
)
from kelpie_carbon_v1.logging_config import get_logger

logger = get_logger(__name__)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('validation.log')
        ]
    )


async def run_primary_validation(days: int, output_dir: str) -> None:
    """Run validation on primary kelp farm sites only.
    
    Args:
        days: Number of days back to search for imagery
        output_dir: Directory to save validation results
    """
    logger.info("🌊 Starting PRIMARY VALIDATION for kelp farm sites")
    logger.info(f"📅 Searching for imagery from last {days} days")
    
    try:
        results = await validate_primary_sites(date_range_days=days)
        
        # Save results
        output_path = Path(output_dir) / f"primary_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create validator instance to save report
        validator = RealWorldValidator()
        validator.validation_results = list(results.values())
        validator.save_validation_report(str(output_path))
        
        # Print summary
        successful = sum(1 for r in results.values() if r.success)
        total = len(results)
        
        logger.info("=" * 60)
        logger.info("PRIMARY VALIDATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"✅ Successful validations: {successful}/{total}")
        logger.info(f"📊 Success rate: {successful/total:.1%}")
        logger.info(f"📁 Results saved to: {output_path}")
        
        # Site-by-site results
        for site_name, result in results.items():
            status = "✅ PASS" if result.success else "❌ FAIL"
            logger.info(f"   {result.site.name}: {result.detection_rate:.1%} detection {status}")
        
    except Exception as e:
        logger.error(f"❌ Primary validation failed: {str(e)}")
        raise


async def run_full_validation(days: int, output_dir: str) -> None:
    """Run validation on all sites including controls.
    
    Args:
        days: Number of days back to search for imagery
        output_dir: Directory to save validation results
    """
    logger.info("🌊 Starting FULL VALIDATION (kelp farms + control sites)")
    logger.info(f"📅 Searching for imagery from last {days} days")
    
    try:
        results = await validate_with_controls(date_range_days=days)
        
        # Save results
        output_path = Path(output_dir) / f"full_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create validator instance to save report
        validator = RealWorldValidator()
        validator.validation_results = list(results.values())
        validator.save_validation_report(str(output_path))
        
        # Print summary
        successful = sum(1 for r in results.values() if r.success)
        total = len(results)
        
        logger.info("=" * 60)
        logger.info("FULL VALIDATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"✅ Successful validations: {successful}/{total}")
        logger.info(f"📊 Success rate: {successful/total:.1%}")
        logger.info(f"📁 Results saved to: {output_path}")
        
        # Categorize results
        kelp_sites = {k: v for k, v in results.items() if v.site.site_type == "kelp_farm"}
        control_sites = {k: v for k, v in results.items() if "control" in v.site.site_type}
        
        if kelp_sites:
            logger.info("\n🌿 KELP FARM RESULTS:")
            for site_name, result in kelp_sites.items():
                status = "✅ PASS" if result.success else "❌ FAIL"
                logger.info(f"   {result.site.name}: {result.detection_rate:.1%} detection {status}")
        
        if control_sites:
            logger.info("\n🔍 CONTROL SITE RESULTS:")
            for site_name, result in control_sites.items():
                status = "✅ PASS" if result.success else "❌ FAIL"
                logger.info(f"   {result.site.name}: {result.detection_rate:.1%} false positive {status}")
        
    except Exception as e:
        logger.error(f"❌ Full validation failed: {str(e)}")
        raise


async def run_controls_only(days: int, output_dir: str) -> None:
    """Run validation on control sites only for false positive testing.
    
    Args:
        days: Number of days back to search for imagery
        output_dir: Directory to save validation results
    """
    logger.info("🔍 Starting CONTROL VALIDATION (false positive testing)")
    logger.info(f"📅 Searching for imagery from last {days} days")
    
    try:
        # Create validator and filter to control sites only
        validator = RealWorldValidator()
        control_sites = {
            k: v for k, v in validator.sites.items() 
            if "control" in v.site_type
        }
        validator.sites = control_sites
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        results = await validator.validate_all_sites(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        # Save results
        output_path = Path(output_dir) / f"control_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        validator.save_validation_report(str(output_path))
        
        # Print summary
        successful = sum(1 for r in results.values() if r.success)
        total = len(results)
        
        logger.info("=" * 60)
        logger.info("CONTROL VALIDATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"✅ Successful validations: {successful}/{total}")
        logger.info(f"📊 Success rate: {successful/total:.1%}")
        logger.info(f"📁 Results saved to: {output_path}")
        
        # Control site results
        for site_name, result in results.items():
            status = "✅ PASS" if result.success else "❌ FAIL"
            logger.info(f"   {result.site.name}: {result.detection_rate:.1%} false positive {status}")
        
    except Exception as e:
        logger.error(f"❌ Control validation failed: {str(e)}")
        raise


def main():
    """Main entry point for the validation script."""
    parser = argparse.ArgumentParser(
        description="Real-world validation for SKEMA kelp detection algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate primary kelp farm sites with 30 days of imagery
  python scripts/run_real_world_validation.py --mode primary --days 30

  # Full validation including control sites with 14 days of imagery
  python scripts/run_real_world_validation.py --mode full --days 14 --output results/

  # Control sites only for false positive testing
  python scripts/run_real_world_validation.py --mode controls --days 7 --verbose

Validation Modes:
  primary  - Validate only the 3 primary kelp farm sites
  full     - Validate all sites (kelp farms + control sites)
  controls - Validate only control sites for false positive testing

Primary Validation Sites:
  • Broughton Archipelago (50.08°N, 126.17°W) - UVic primary SKEMA site
  • Saanich Inlet (48.58°N, 123.50°W) - Multi-species validation
  • Monterey Bay (36.80°N, 121.90°W) - Giant kelp validation

Control Sites:
  • Mojave Desert (36.00°N, 118.00°W) - Land control
  • Open Ocean (45.00°N, 135.00°W) - Deep water control
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["primary", "full", "controls"],
        default="primary",
        help="Validation mode: primary kelp sites, full validation, or controls only"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days back to search for satellite imagery (default: 30)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="validation_results",
        help="Output directory for validation results (default: validation_results)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    logger.info("🚀 Starting SKEMA Real-World Validation")
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Date range: {args.days} days")
    logger.info(f"Output directory: {args.output}")
    
    try:
        if args.mode == "primary":
            asyncio.run(run_primary_validation(args.days, args.output))
        elif args.mode == "full":
            asyncio.run(run_full_validation(args.days, args.output))
        elif args.mode == "controls":
            asyncio.run(run_controls_only(args.days, args.output))
        
        logger.info("🎉 Validation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Validation failed: {str(e)}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 