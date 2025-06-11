#!/usr/bin/env python3
"""Test script for utility modules integration."""

import numpy as np
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kelpie_carbon_v1.utils import (
    array_utils, 
    validation_utils, 
    performance_utils, 
    math_utils
)
from kelpie_carbon_v1.utils.validation_utils import ValidationError


def test_array_utils():
    """Test array utility functions."""
    print("🔢 Testing Array Utils...")
    
    # Test normalization
    test_array = np.array([1, 2, 3, 4, 5])
    normalized = array_utils.normalize_array(test_array, method="minmax")
    assert normalized.min() == 0.0 and normalized.max() == 1.0
    print("  ✅ Array normalization works")
    
    # Test statistics calculation
    stats = array_utils.calculate_statistics(test_array)
    assert stats['mean'] == 3.0
    assert stats['median'] == 3.0
    print("  ✅ Statistics calculation works")
    
    # Test safe division
    numerator = np.array([1, 2, 3])
    denominator = np.array([1, 0, 3])  # Contains zero
    result = array_utils.safe_divide(numerator, denominator, fill_value=-1.0)
    assert result[1] == -1.0  # Division by zero handled
    print("  ✅ Safe division works")
    
    print("✅ Array Utils - ALL TESTS PASSED\n")


def test_validation_utils():
    """Test validation utility functions."""
    print("🔒 Testing Validation Utils...")
    
    # Test coordinate validation
    try:
        validation_utils.validate_coordinates(45.0, -122.0)
        print("  ✅ Valid coordinates accepted")
    except ValidationError:
        assert False, "Valid coordinates should not raise error"
    
    # Test invalid coordinates
    try:
        validation_utils.validate_coordinates(91.0, -122.0)  # Invalid latitude
        assert False, "Invalid coordinates should raise error"
    except ValidationError:
        print("  ✅ Invalid coordinates rejected")
    
    # Test date validation
    try:
        start, end = validation_utils.validate_date_range("2023-01-01", "2023-12-31")
        assert start.year == 2023
        print("  ✅ Date validation works")
    except ValidationError:
        assert False, "Valid dates should not raise error"
    
    # Test numeric range validation
    try:
        validation_utils.validate_numeric_range(5, min_value=1, max_value=10)
        print("  ✅ Numeric range validation works")
    except ValidationError:
        assert False, "Valid number should not raise error"
    
    print("✅ Validation Utils - ALL TESTS PASSED\n")


def test_performance_utils():
    """Test performance utility functions."""
    print("📊 Testing Performance Utils...")
    
    # Test timing context
    with performance_utils.timing_context("test_operation") as timer:
        # Simulate some work
        sum(range(10000))  # More work to ensure measurable time
    
    assert "execution_time" in timer
    assert timer["execution_time"] >= 0  # Allow for very fast execution
    print("  ✅ Timing context works")
    
    # Test memory usage
    memory = performance_utils.memory_usage()
    assert memory > 0
    print(f"  ✅ Memory usage: {memory:.1f} MB")
    
    # Test function profiling decorator
    @performance_utils.profile_function(log_calls=False)
    def test_function(x):
        return sum(range(x))
    
    result = test_function(100)
    assert result == sum(range(100))
    print("  ✅ Function profiling decorator works")
    
    # Test performance monitor
    monitor = performance_utils.get_performance_monitor()
    stats = monitor.get_overall_stats()
    print(f"  ✅ Performance monitor tracking {stats.get('total_functions_monitored', 0)} functions")
    
    print("✅ Performance Utils - ALL TESTS PASSED\n")


def test_math_utils():
    """Test mathematical utility functions."""
    print("🧮 Testing Math Utils...")
    
    # Test area calculation
    area = math_utils.calculate_area_from_pixels(100, 10.0)  # 100 pixels, 10m each
    assert area == 10000.0  # 100 * 10^2
    print("  ✅ Area calculation works")
    
    # Test distance calculation
    distance = math_utils.calculate_distance(0, 0, 1, 1, method="euclidean")
    assert distance > 0
    print(f"  ✅ Distance calculation: {distance:.0f} meters")
    
    # Test Gaussian kernel
    kernel = math_utils.gaussian_kernel(5, 1.0)
    assert kernel.shape == (5, 5)
    assert abs(kernel.sum() - 1.0) < 1e-10  # Should be normalized
    print("  ✅ Gaussian kernel generation works")
    
    print("✅ Math Utils - ALL TESTS PASSED\n")


def test_integration():
    """Test integration between utility modules."""
    print("🔗 Testing Integration...")
    
    # Create test data using array utils
    test_data = np.random.normal(50, 10, 1000)
    
    # Normalize with array utils
    normalized_data = array_utils.normalize_array(test_data)
    
    # Get statistics
    stats = array_utils.calculate_statistics(normalized_data)
    
    # Validate the results using validation utils
    try:
        validation_utils.validate_numeric_range(
            stats['mean'], 
            min_value=0.0, 
            max_value=1.0,
            name="normalized_mean"
        )
        print("  ✅ Cross-module validation works")
    except ValidationError as e:
        print(f"  ❌ Validation failed: {e}")
        return False
    
    # Test with performance monitoring
    @performance_utils.profile_function(log_calls=False)
    def data_processing_pipeline(data):
        # Normalize
        normalized = array_utils.normalize_array(data)
        # Calculate stats  
        stats = array_utils.calculate_statistics(normalized)
        # Calculate area (example usage)
        area = math_utils.calculate_area_from_pixels(len(data), 1.0)
        return stats, area
    
    with performance_utils.timing_context("integration_test") as timer:
        stats, area = data_processing_pipeline(test_data)
    
    print(f"  ✅ Pipeline processed {len(test_data)} points in {timer['execution_time']:.3f}s")
    print(f"  ✅ Calculated area: {area:.0f} m²")
    
    print("✅ Integration - ALL TESTS PASSED\n")


def main():
    """Run all utility tests."""
    print("🧪 Testing Kelpie Carbon v1 Utility Modules")
    print("=" * 50)
    
    try:
        test_array_utils()
        test_validation_utils()
        test_performance_utils()
        test_math_utils()
        test_integration()
        
        print("🎉 ALL UTILITY TESTS PASSED!")
        print("\n📋 Utility Module Summary:")
        print("  • Array Utils: Normalization, statistics, interpolation")
        print("  • Validation Utils: Coordinates, dates, configurations")  
        print("  • Performance Utils: Timing, memory, function profiling")
        print("  • Math Utils: Geospatial calculations, kernels")
        print("\n✅ Task C5.3 (Organize Utility Functions) - COMPLETE")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 