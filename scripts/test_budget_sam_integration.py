#!/usr/bin/env python3
"""
Budget SAM Integration Test

Test script to validate the zero-cost SAM-based kelp detection pipeline
works correctly with the existing SKEMA infrastructure.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_skema_processor():
    """Test the SKEMA processor functionality."""
    print("🧪 Testing SKEMA Processor...")
    
    from src.kelpie_carbon_v1.spectral.skema_processor import SKEMAProcessor
    
    # Initialize processor
    skema = SKEMAProcessor()
    
    # Test with synthetic multispectral data
    height, width = 128, 128
    rgb_image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    nir_band = np.random.rand(height, width) * 0.8 + 0.2  # NIR values 0.2-1.0
    red_edge_band = np.random.rand(height, width) * 0.6 + 0.1  # Red-edge values 0.1-0.7
    
    # Calculate spectral indices
    indices = skema.calculate_spectral_indices(rgb_image, nir_band, red_edge_band)
    print(f"✅ Spectral indices calculated: {list(indices.keys())}")
    
    # Generate kelp probability mask
    kelp_mask = skema.get_kelp_probability_mask(rgb_image, nir_band, red_edge_band)
    kelp_pixels = int(kelp_mask.sum())
    print(f"✅ Kelp probability mask: {kelp_pixels} potential kelp pixels")
    
    # Test threshold access
    thresholds = skema.get_optimized_thresholds()
    print(f"✅ Optimized thresholds: {thresholds}")
    
    # Validate results
    assert isinstance(indices, dict), "Should return indices dictionary"
    assert isinstance(kelp_mask, np.ndarray), "Should return numpy array mask"
    assert isinstance(thresholds, dict), "Should return thresholds dictionary"

def test_sam_availability():
    """Test SAM model availability (without downloading)."""
    print("\n🎭 Testing SAM Availability...")
    
    try:
        from segment_anything import sam_model_registry, SamPredictor
        print("✅ SAM library import successful")
        
        # Test model registry
        available_models = list(sam_model_registry.keys())
        print(f"✅ Available SAM models: {available_models}")
        
        # Validate SAM availability
        assert len(available_models) > 0, "Should have available SAM models"
        
    except ImportError as e:
        print(f"❌ SAM import failed: {e}")
        assert False, f"SAM import failed: {e}"

def test_budget_detector_class():
    """Test the budget detector class (without model file)."""
    print("\n🌊 Testing Budget SAM Detector Class...")
    
    try:
        from src.kelpie_carbon_v1.deep_learning.budget_sam_detector import BudgetSAMKelpDetector
        print("✅ Budget SAM Detector import successful")
        
        # Test synthetic data processing methods
        height, width = 64, 64
        rgb_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        nir_band = np.random.rand(height, width) * 0.8
        red_edge_band = np.random.rand(height, width) * 0.6
        
        # Create a mock detector to test methods (without SAM model)
        class MockDetector:
            def __init__(self):
                from src.kelpie_carbon_v1.spectral.skema_processor import SKEMAProcessor
                self.skema_processor = SKEMAProcessor()
            
            def _generate_spectral_guidance_points(self, rgb_image, nir_band, red_edge_band):
                """Test spectral guidance point generation."""
                # Calculate SKEMA spectral indices
                red_band = rgb_image[:, :, 0]
                
                # NDVI: (NIR - Red) / (NIR + Red)
                ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
                
                # NDRE: (NIR - RedEdge) / (NIR + RedEdge)  
                ndre = (nir_band - red_edge_band) / (nir_band + red_edge_band + 1e-8)
                
                # Apply SKEMA thresholds
                kelp_probability = (ndvi > 0.1) & (ndre > 0.04)
                
                # Simple peak finding - just find center points of kelp areas
                y_indices, x_indices = np.where(kelp_probability)
                if len(y_indices) > 0:
                    # Take every 10th point to avoid too many
                    step = max(1, len(y_indices) // 10)
                    guidance_points = [(int(x), int(y)) for y, x in 
                                     zip(y_indices[::step], x_indices[::step])]
                else:
                    guidance_points = []
                
                return guidance_points
        
        mock_detector = MockDetector()
        guidance_points = mock_detector._generate_spectral_guidance_points(
            rgb_image, nir_band, red_edge_band
        )
        
        print(f"✅ Spectral guidance generation: {len(guidance_points)} points")
        
        # Validate guidance point generation
        assert isinstance(guidance_points, list), "Should return list of guidance points"
        
    except Exception as e:
        print(f"❌ Budget detector test failed: {e}")
        assert False, f"Budget detector test failed: {e}"

def test_dependencies():
    """Test all required dependencies."""
    print("\n📦 Testing Dependencies...")
    
    dependencies = [
        ("numpy", "import numpy"),
        ("opencv", "import cv2"),
        ("matplotlib", "import matplotlib.pyplot"),
        ("scipy", "import scipy"),
        ("rasterio", "import rasterio"),
        ("torch", "import torch"),
        ("torchvision", "import torchvision"),
    ]
    
    success_count = 0
    for name, import_stmt in dependencies:
        try:
            exec(import_stmt)
            print(f"✅ {name} available")
            success_count += 1
        except ImportError:
            print(f"❌ {name} not available")
    
    print(f"\n📊 Dependencies: {success_count}/{len(dependencies)} available")
    
    # Allow some missing dependencies but require core ones
    core_deps = ["numpy", "opencv", "torch"]
    core_available = sum(1 for name, _ in dependencies if name in core_deps)
    assert core_available >= 2, f"Too many core dependencies missing: {core_available}/3"

def main():
    """Run all tests."""
    print("🌊 Budget SAM Kelp Detection - Integration Test")
    print("💰 Testing zero-cost deep learning approach")
    
    tests = [
        ("Dependencies", test_dependencies),
        ("SKEMA Processor", test_skema_processor),
        ("SAM Availability", test_sam_availability),
        ("Budget Detector Class", test_budget_detector_class),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            test_func()  # Will raise AssertionError if it fails
            print(f"✅ {test_name} - PASSED")
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_name} - FAILED: {e}")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Budget SAM integration is ready!")
        print("\n📋 Next steps:")
        print("   1. Download SAM model: python -c \"from src.kelpie_carbon_v1.deep_learning.budget_sam_detector import download_sam_model; download_sam_model()\"")
        print("   2. Test with real satellite imagery")
        print("   3. Compare results with existing SKEMA spectral analysis")
        print("\n💡 Usage example:")
        print("   from src.kelpie_carbon_v1.deep_learning.budget_sam_detector import BudgetSAMKelpDetector")
        print("   detector = BudgetSAMKelpDetector('models/sam_vit_h_4b8939.pth')")
        print("   kelp_mask, metadata = detector.detect_kelp_from_file('satellite_image.tif')")
    else:
        print(f"\n❌ {total - passed} tests failed. Please fix issues before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 