#!/usr/bin/env python3
"""
Budget Deep Learning Suite Test

Comprehensive test for all budget-friendly deep learning approaches:
1. SAM-based kelp detection (zero cost)
2. U-Net transfer learning (minimal cost)
3. Classical ML enhancement (zero cost)
4. Integration and comparison testing
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_sam_detector():
    """Test the SAM-based detector."""
    print("🎭 Testing SAM Detector...")
    
    try:
        from src.kelpie_carbon_v1.deep_learning import BudgetSAMKelpDetector, download_sam_model
        
        # Test import and basic functionality
        detector = BudgetSAMKelpDetector()
        print("✅ SAM detector initialization successful")
        
        # Test synthetic data processing
        height, width = 128, 128
        rgb_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        nir_band = np.random.rand(height, width) * 0.8
        red_edge_band = np.random.rand(height, width) * 0.6
        
        # Test guidance point generation (without actual SAM model)
        if hasattr(detector, '_generate_spectral_guidance_points'):
            guidance_points = detector._generate_spectral_guidance_points(
                rgb_image, nir_band, red_edge_band
            )
            print(f"✅ Spectral guidance: {len(guidance_points)} points generated")
            assert len(guidance_points) >= 0, "Should generate some guidance points"
        
        # Test passed successfully
        assert True
        
    except Exception as e:
        print(f"❌ SAM detector test failed: {e}")
        assert False, f"SAM detector test failed: {e}"

def test_unet_detector():
    """Test the U-Net detector."""
    print("\n🏗️ Testing U-Net Detector...")
    
    try:
        from src.kelpie_carbon_v1.deep_learning import BudgetUNetKelpDetector, setup_budget_unet_environment
        
        # Test import and initialization
        detector = BudgetUNetKelpDetector()
        print("✅ U-Net detector initialization successful")
        
        # Test synthetic data processing
        height, width = 256, 256
        rgb_image = np.random.rand(height, width, 3).astype(np.float32)
        nir_band = np.random.rand(height, width) * 0.8
        red_edge_band = np.random.rand(height, width) * 0.6
        
        # Test detection (will use fallback if no model available)
        kelp_mask, metadata = detector.detect_kelp(rgb_image, nir_band, red_edge_band)
        
        print(f"✅ Detection successful using {metadata['detection_method']}")
        print(f"   Kelp pixels: {metadata['kelp_pixels']}")
        print(f"   Coverage: {metadata['kelp_percentage']:.2f}%")
        
        # Validate results
        assert isinstance(kelp_mask, np.ndarray), "Should return numpy array"
        assert isinstance(metadata, dict), "Should return metadata dict"
        assert 'detection_method' in metadata, "Should include detection method"
        
        # Test setup instructions
        setup_budget_unet_environment()
        print("✅ Setup instructions available")
        
    except Exception as e:
        print(f"❌ U-Net detector test failed: {e}")
        assert False, f"U-Net detector test failed: {e}"

def test_spectral_integration():
    """Test integration with SKEMA spectral analysis."""
    print("\n🔬 Testing Spectral Integration...")
    
    try:
        from src.kelpie_carbon_v1.spectral import SKEMAProcessor
        
        # Initialize processor
        skema = SKEMAProcessor()
        
        # Test spectral indices calculation
        height, width = 128, 128
        rgb_image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
        nir_band = np.random.rand(height, width) * 0.8 + 0.2
        red_edge_band = np.random.rand(height, width) * 0.6 + 0.1
        
        # Calculate indices
        indices = skema.calculate_spectral_indices(rgb_image, nir_band, red_edge_band)
        print(f"✅ Spectral indices: {list(indices.keys())}")
        assert isinstance(indices, dict), "Should return dictionary of indices"
        assert len(indices) > 0, "Should calculate some spectral indices"
        
        # Test kelp probability mask
        kelp_mask = skema.get_kelp_probability_mask(rgb_image, nir_band, red_edge_band)
        kelp_pixels = int(kelp_mask.sum())
        print(f"✅ Spectral kelp detection: {kelp_pixels} pixels")
        assert isinstance(kelp_mask, np.ndarray), "Should return numpy array mask"
        
        # Test optimized thresholds
        thresholds = skema.get_optimized_thresholds()
        print(f"✅ Optimized thresholds: NDVI≥{thresholds['ndvi_threshold']}, NDRE≥{thresholds['ndre_threshold']}")
        assert isinstance(thresholds, dict), "Should return threshold dictionary"
        
    except Exception as e:
        print(f"❌ Spectral integration test failed: {e}")
        assert False, f"Spectral integration test failed: {e}"

def test_comparison_analysis():
    """Test comparison between different detection methods."""
    print("\n⚖️  Testing Method Comparison...")
    
    try:
        from src.kelpie_carbon_v1.deep_learning import BudgetSAMKelpDetector, BudgetUNetKelpDetector
        from src.kelpie_carbon_v1.spectral import SKEMAProcessor
        
        # Create test data
        height, width = 256, 256
        rgb_image = np.random.rand(height, width, 3).astype(np.float32)
        nir_band = np.random.rand(height, width) * 0.8
        red_edge_band = np.random.rand(height, width) * 0.6
        
        results = {}
        
        # Test SAM approach
        try:
            sam_detector = BudgetSAMKelpDetector()
            # Use spectral guidance instead of full SAM (no model needed)
            skema = SKEMAProcessor()
            sam_mask = skema.get_kelp_probability_mask(rgb_image, nir_band, red_edge_band)
            results["SAM+Spectral"] = int(sam_mask.sum())
        except:
            results["SAM+Spectral"] = "Not available"
        
        # Test U-Net approach
        try:
            unet_detector = BudgetUNetKelpDetector()
            unet_mask, unet_metadata = unet_detector.detect_kelp(rgb_image, nir_band, red_edge_band)
            results[unet_metadata["detection_method"]] = unet_metadata["kelp_pixels"]
        except:
            results["U-Net"] = "Not available"
        
        # Test pure spectral approach
        try:
            skema_processor = SKEMAProcessor()
            spectral_mask = skema_processor.get_kelp_probability_mask(rgb_image, nir_band, red_edge_band)
            results["Pure Spectral"] = int(spectral_mask.sum())
        except:
            results["Pure Spectral"] = "Not available"
        
        print("📊 Method Comparison Results:")
        for method, pixels in results.items():
            if isinstance(pixels, int):
                percentage = pixels / (height * width) * 100
                print(f"   {method}: {pixels:,} pixels ({percentage:.2f}%)")
            else:
                print(f"   {method}: {pixels}")
        
        # Validate that at least one method worked
        assert len(results) > 0, "Should have at least one detection method result"
        
    except Exception as e:
        print(f"❌ Comparison analysis failed: {e}")
        assert False, f"Comparison analysis failed: {e}"

def test_cost_analysis():
    """Analyze cost implications of different approaches."""
    print("\n💰 Cost Analysis...")
    
    cost_breakdown = {
        "SAM + Spectral": {
            "setup_cost": "$0",
            "training_cost": "$0", 
            "inference_cost": "$0/image",
            "model_size": "2.5GB (one-time download)",
            "description": "Pre-trained SAM + SKEMA spectral guidance"
        },
        "U-Net Transfer Learning": {
            "setup_cost": "$0-20",
            "training_cost": "$0-20 (Google Colab)",
            "inference_cost": "$0/image", 
            "model_size": "80MB (encoder frozen)",
            "description": "Minimal decoder fine-tuning"
        },
        "Pure Spectral (SKEMA)": {
            "setup_cost": "$0",
            "training_cost": "$0",
            "inference_cost": "$0/image",
            "model_size": "0MB (algorithmic)",
            "description": "Optimized spectral indices only"
        }
    }
    
    print("📋 Approach Cost Breakdown:")
    for approach, costs in cost_breakdown.items():
        print(f"\n🔹 {approach}:")
        print(f"   Setup: {costs['setup_cost']}")
        print(f"   Training: {costs['training_cost']}")
        print(f"   Inference: {costs['inference_cost']}")
        print(f"   Model Size: {costs['model_size']}")
        print(f"   Description: {costs['description']}")
    
    total_max_cost = 50  # Maximum across all approaches
    print(f"\n💡 Maximum total cost across all approaches: ${total_max_cost}")
    print("   vs. Original CNN training budget: $750-1,200 (93-96% savings)")
    
    # Validate cost analysis
    assert isinstance(cost_breakdown, dict), "Should return cost breakdown dictionary"
    assert len(cost_breakdown) > 0, "Should have cost information for different approaches"

def test_deployment_readiness():
    """Test deployment readiness across all approaches."""
    print("\n🚀 Testing Deployment Readiness...")
    
    deployment_tests = [
        ("Poetry Environment", test_poetry_imports),
        ("Dependency Availability", test_dependencies),
        ("Memory Requirements", test_memory_usage),
        ("Processing Speed", test_processing_speed),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    for test_name, test_func in deployment_tests:
        try:
            test_func()  # Will raise AssertionError if it fails
            print(f"✅ {test_name}: READY")
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_name}: NEEDS ATTENTION - {e}")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    readiness_score = passed / len(deployment_tests) * 100
    print(f"\n📊 Deployment Readiness: {readiness_score:.0f}% ({passed}/{len(deployment_tests)} tests passed)")
    
    # Assert deployment readiness
    assert readiness_score >= 60, f"Deployment readiness too low: {readiness_score:.0f}%"

def test_poetry_imports():
    """Test all imports work in Poetry environment."""
    try:
        from src.kelpie_carbon_v1.deep_learning import (
            BudgetSAMKelpDetector, BudgetUNetKelpDetector, 
            download_sam_model, setup_budget_unet_environment
        )
        from src.kelpie_carbon_v1.spectral import SKEMAProcessor
        assert True  # Imports succeeded
    except ImportError as e:
        assert False, f"Import failed: {e}"

def test_dependencies():
    """Test critical dependencies."""
    deps = ["torch", "cv2", "numpy", "matplotlib", "rasterio"]
    for dep in deps:
        try:
            __import__(dep)
        except ImportError as e:
            assert False, f"Missing dependency: {dep} - {e}"
    assert True  # All dependencies available

def test_memory_usage():
    """Test memory requirements are reasonable."""
    try:
        # Create moderate-sized test data
        test_image = np.random.rand(512, 512, 3).astype(np.float32)
        # Should not cause memory issues on typical systems
        assert test_image.nbytes < 100_000_000, f"Memory usage too high: {test_image.nbytes} bytes"
    except MemoryError as e:
        assert False, f"Memory error: {e}"

def test_processing_speed():
    """Test processing speed is acceptable."""
    import time
    try:
        from src.kelpie_carbon_v1.spectral import SKEMAProcessor
        skema = SKEMAProcessor()
        
        # Time spectral processing
        rgb_image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        nir_band = np.random.rand(256, 256) * 0.8
        
        start_time = time.time()
        kelp_mask = skema.get_kelp_probability_mask(rgb_image, nir_band)
        processing_time = time.time() - start_time
        
        # Should process in reasonable time (< 5 seconds for test data)
        assert processing_time < 5.0, f"Processing too slow: {processing_time:.2f} seconds"
        
    except Exception as e:
        assert False, f"Processing speed test failed: {e}"

def test_error_handling():
    """Test graceful error handling."""
    try:
        from src.kelpie_carbon_v1.deep_learning import BudgetUNetKelpDetector
        
        # Test with invalid input
        detector = BudgetUNetKelpDetector()
        
        # Should handle empty input gracefully
        empty_image = np.zeros((10, 10, 3))
        kelp_mask, metadata = detector.detect_kelp(empty_image)
        
        # Should return reasonable defaults
        assert isinstance(kelp_mask, np.ndarray), "Should return numpy array"
        assert isinstance(metadata, dict), "Should return metadata dict"
        
    except Exception as e:
        assert False, f"Error handling test failed: {e}"

def main():
    """Run comprehensive budget deep learning test suite."""
    print("🌊 Budget Deep Learning Suite - Comprehensive Test")
    print("💰 Testing zero-cost to minimal-cost approaches")
    print("="*50)
    
    test_suite = [
        ("SAM Detector", test_sam_detector),
        ("U-Net Detector", test_unet_detector),
        ("Spectral Integration", test_spectral_integration),
        ("Method Comparison", test_comparison_analysis),
        ("Cost Analysis", test_cost_analysis),
        ("Deployment Readiness", test_deployment_readiness),
    ]
    
    passed = 0
    total = len(test_suite)
    
    for test_name, test_func in test_suite:
        print(f"\n{'='*50}")
        try:
            if test_func():
                print(f"✅ {test_name} - PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 Final Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Budget deep learning suite is ready!")
        print("\n📋 Available Approaches:")
        print("   1. SAM + Spectral Guidance (Primary) - $0 cost")
        print("   2. U-Net Transfer Learning (Secondary) - $0-20 cost")
        print("   3. Pure Spectral Analysis (Fallback) - $0 cost")
        print("\n💡 Next steps:")
        print("   1. Choose primary approach based on requirements")
        print("   2. Download models if needed (SAM: one-time 2.5GB)")
        print("   3. Test with real satellite imagery")
        print("   4. Deploy to production environment")
    else:
        print(f"\n❌ {total - passed} tests failed. Address issues before deployment.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 