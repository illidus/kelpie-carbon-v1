#!/usr/bin/env python3
"""
Threshold Optimization Script for Task A2.7: Optimize detection pipeline.

This script analyzes validation results and generates optimized detection
thresholds for different environmental conditions and site types.

Usage:
    python scripts/run_threshold_optimization.py --validation-results results/primary_validation_20250610_092434.json
    python scripts/run_threshold_optimization.py --validation-results results/full_validation.json --output results/optimization/
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kelpie_carbon.logging_config import get_logger
from kelpie_carbon.optimization import (
    ThresholdOptimizer,
    optimize_detection_pipeline,
)

logger = get_logger(__name__)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('optimization.log')
        ]
    )


def run_optimization(validation_results_path: str, output_dir: str) -> None:
    """Run threshold optimization analysis.
    
    Args:
        validation_results_path: Path to validation results JSON file
        output_dir: Directory to save optimization results
    """
    logger.info("🔧 Starting SKEMA Detection Pipeline Optimization")
    logger.info(f"📊 Validation results: {validation_results_path}")
    logger.info(f"📁 Output directory: {output_dir}")
    
    try:
        # Run comprehensive optimization
        optimization_results = optimize_detection_pipeline(validation_results_path, output_dir)
        
        # Display results summary
        analysis = optimization_results['current_analysis']
        recommendations = optimization_results['recommendations']
        
        logger.info("=" * 60)
        logger.info("OPTIMIZATION ANALYSIS COMPLETE")
        logger.info("=" * 60)
        
        logger.info("📈 Current Performance:")
        logger.info(f"   Mean detection rate: {analysis['mean_detection_rate']:.1%}")
        logger.info(f"   Expected detection rate: {analysis['mean_expected_rate']:.1%}")
        logger.info(f"   Over-detection ratio: {analysis['over_detection_ratio']:.1f}x")
        logger.info(f"   Accuracy score: {analysis['accuracy_score']:.3f}")
        
        logger.info("\n🎯 Optimization Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"   {i}. {rec}")
        
        logger.info("\n⚙️ Optimized Scenarios Generated:")
        scenarios = optimization_results['optimized_scenarios']
        for scenario_name, config in scenarios.items():
            logger.info(f"   📋 {scenario_name}:")
            if 'ndre_threshold' in config:
                logger.info(f"      NDRE threshold: {config['ndre_threshold']:.3f}")
            if 'kelp_fai_threshold' in config:
                logger.info(f"      FAI threshold: {config['kelp_fai_threshold']:.3f}")
            if 'min_detection_threshold' in config:
                logger.info(f"      Min detection: {config['min_detection_threshold']:.3f}")
        
        # Generate example configurations for common scenarios
        logger.info("\n🌊 Example Configurations:")
        
        optimizer = ThresholdOptimizer()
        
        # Kelp farm with moderate conditions
        kelp_farm_config = optimizer.create_adaptive_config(
            'kelp_farm', 
            {'cloud_cover': 0.15, 'turbidity': 'medium'}
        )
        logger.info("   🌿 Kelp Farm (moderate conditions):")
        logger.info(f"      NDRE: {kelp_farm_config['ndre_threshold']:.3f}, "
                   f"FAI: {kelp_farm_config['kelp_fai_threshold']:.3f}")
        
        # Open ocean with clear conditions
        ocean_config = optimizer.create_adaptive_config(
            'open_ocean',
            {'cloud_cover': 0.10, 'turbidity': 'low'}
        )
        logger.info("   🌊 Open Ocean (clear conditions):")
        logger.info(f"      NDRE: {ocean_config['ndre_threshold']:.3f}, "
                   f"FAI: {ocean_config['kelp_fai_threshold']:.3f}")
        
        # Real-time optimization
        realtime_config = optimizer.optimize_for_real_time(15.0)
        logger.info("   ⚡ Real-time optimized (15s target):")
        logger.info(f"      NDRE: {realtime_config['ndre_threshold']:.3f}, "
                   f"FAI: {realtime_config['kelp_fai_threshold']:.3f}")
        
        logger.info("=" * 60)
        logger.info("🎉 Optimization completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {str(e)}")
        raise


def demonstrate_adaptive_thresholding() -> None:
    """Demonstrate adaptive thresholding capabilities."""
    logger.info("\n🔬 ADAPTIVE THRESHOLDING DEMONSTRATION")
    logger.info("=" * 50)
    
    optimizer = ThresholdOptimizer()
    
    # Test different environmental conditions
    test_scenarios = [
        ('kelp_farm', {'cloud_cover': 0.05, 'turbidity': 'low'}, "Clear day"),
        ('kelp_farm', {'cloud_cover': 0.35, 'turbidity': 'medium'}, "Cloudy day"),
        ('kelp_farm', {'cloud_cover': 0.15, 'turbidity': 'high'}, "Turbid water"),
        ('open_ocean', {'cloud_cover': 0.20, 'turbidity': 'low'}, "Open ocean"),
        ('coastal', {'cloud_cover': 0.25, 'turbidity': 'high'}, "Turbid coastal"),
    ]
    
    for site_type, conditions, description in test_scenarios:
        config = optimizer.create_adaptive_config(site_type, conditions)
        logger.info(f"📍 {description} ({site_type}):")
        logger.info(f"   NDRE threshold: {config['ndre_threshold']:.3f}")
        logger.info(f"   FAI threshold: {config['kelp_fai_threshold']:.3f}")
        logger.info(f"   Min cluster size: {config['min_kelp_cluster_size']}")
        logger.info(f"   Morphology: {'Yes' if config['apply_morphology'] else 'No'}")


def main():
    """Main entry point for the optimization script."""
    parser = argparse.ArgumentParser(
        description="Optimize SKEMA detection thresholds based on validation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize based on primary validation results
  python scripts/run_threshold_optimization.py \\
    --validation-results results/primary_validation_20250610_092434.json
  
  # Run with custom output directory
  python scripts/run_threshold_optimization.py \\
    --validation-results results/full_validation.json \\
    --output results/my_optimization/
  
  # Demonstrate adaptive thresholding only
  python scripts/run_threshold_optimization.py --demo-only
        """
    )
    
    parser.add_argument(
        '--validation-results',
        type=str,
        help='Path to validation results JSON file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/optimization/',
        help='Output directory for optimization results (default: results/optimization/)'
    )
    
    parser.add_argument(
        '--demo-only',
        action='store_true',
        help='Only demonstrate adaptive thresholding capabilities'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    try:
        if args.demo_only:
            # Just demonstrate adaptive thresholding
            demonstrate_adaptive_thresholding()
        else:
            if not args.validation_results:
                logger.error("❌ --validation-results is required unless using --demo-only")
                parser.print_help()
                sys.exit(1)
            
            # Check if validation results file exists
            if not Path(args.validation_results).exists():
                logger.error(f"❌ Validation results file not found: {args.validation_results}")
                sys.exit(1)
            
            # Run optimization
            run_optimization(args.validation_results, args.output)
            
            # Also demonstrate adaptive thresholding
            demonstrate_adaptive_thresholding()
    
    except KeyboardInterrupt:
        logger.info("⚠️ Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Optimization failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 
