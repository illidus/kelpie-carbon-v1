# Red-Edge Band Processing Enhancement - Technical Specification

## 🎯 **Overview**
Technical specification for implementing red-edge band processing capabilities in Kelpie Carbon v1 to enable advanced submerged kelp detection using SKEMA methodology insights.

## 📋 **Current System Analysis**

### **Existing Capabilities**
- ✅ Sentinel-2 L2A image processing
- ✅ RGB composite generation (Bands 4, 3, 2)
- ✅ Multi-spectral band access (13 bands)
- ✅ NDVI calculation using Bands 8 and 4
- ✅ Cloud masking and atmospheric correction
- ✅ Caching and performance optimization

### **Enhancement Requirements**
Based on SKEMA research findings, we need to add:
- 🆕 Red-edge band processing (Bands 5, 6, 7)
- 🆕 NDRE (Normalized Difference Red Edge) calculation
- 🆕 Water Anomaly Filter (WAF) implementation
- 🆕 Derivative-based feature detection
- 🆕 Enhanced submerged kelp detection algorithms

## 🔬 **Sentinel-2 Red-Edge Band Specifications**

### **Red-Edge Bands Available**
| Band | Name | Central Wavelength | Bandwidth | Spatial Resolution |
|------|------|-------------------|-----------|-------------------|
| Band 5 | Red Edge 1 | 705 nm | 15 nm | 20m |
| Band 6 | Red Edge 2 | 740 nm | 15 nm | 20m |
| Band 7 | Red Edge 3 | 783 nm | 20 nm | 20m |
| Band 8 | NIR | 842 nm | 115 nm | 10m |
| Band 8A | NIR Narrow | 865 nm | 20 nm | 20m |

### **Optimal Band Selection (Based on Research)**
- **Primary Red-Edge**: Band 6 (740nm) - optimal for submerged kelp detection
- **Secondary Red-Edge**: Band 5 (705nm) - fucoxanthin absorption detection
- **Reference Red**: Band 4 (665nm) - for NDRE calculation
- **Validation NIR**: Band 8 (842nm) - comparison with existing NDVI

## ⚙️ **Implementation Plan**

### **Phase 1: Core Red-Edge Processing**

#### **1.1 Enhanced Image Loading**
Current band loading needs enhancement to include red-edge bands.

#### **1.2 NDRE Calculation Implementation**
Calculate Normalized Difference Red Edge Index based on Timmer et al. (2022) findings:
- Superior submerged kelp detection vs NDVI
- 18% more kelp extent detected
- Detection depth: 90-100cm vs 30-50cm for NDVI

#### **1.3 API Endpoint Enhancement**
Enhanced analysis with red-edge processing including comparative analysis.

### **Phase 2: Water Anomaly Filter (WAF)**

#### **2.1 WAF Algorithm Implementation**
```python
def water_anomaly_filter(image_data, kernel_size=5):
    """
    Water Anomaly Filter based on Uhl et al. (2016)
    Removes sunglint, foam, and anthropogenic objects
    
    Steps:
    1. Moving window analysis (5x5 kernel)
    2. Calculate mean and std dev excluding center pixel
    3. Outlier-corrected mean calculation
    4. Replace anomalies with corrected values
    """
    
    filtered_image = image_data.copy()
    height, width = image_data.shape
    
    # Pad image for border handling
    pad_size = kernel_size // 2
    padded_image = np.pad(image_data, pad_size, mode='reflect')
    
    for i in range(height):
        for j in range(width):
            # Extract 5x5 window
            window = padded_image[i:i+kernel_size, j:j+kernel_size]
            center_val = window[pad_size, pad_size]
            
            # Calculate stats excluding center pixel
            window_flat = window.flatten()
            window_flat = np.delete(window_flat, len(window_flat)//2)
            window_mean = np.mean(window_flat)
            window_std = np.std(window_flat)
            
            # Outlier-corrected mean
            valid_pixels = window_flat[
                (window_flat >= window_mean - window_std) & 
                (window_flat <= window_mean + window_std)
            ]
            corrected_mean = np.mean(valid_pixels)
            
            # Replace anomalies
            if (center_val < corrected_mean - window_std or 
                center_val > corrected_mean + window_std):
                filtered_image[i, j] = corrected_mean
                
    return filtered_image
```

#### **2.2 Integrated Processing Pipeline**
```python
def enhanced_image_preprocessing(image_data):
    """Complete preprocessing pipeline with WAF"""
    
    # Step 1: Atmospheric correction (existing)
    corrected_data = apply_atmospheric_correction(image_data)
    
    # Step 2: Water Anomaly Filter (NEW)
    filtered_data = {}
    for band_name, band_data in corrected_data.items():
        filtered_data[band_name] = water_anomaly_filter(band_data)
    
    # Step 3: Geometric correction (existing)
    geometrically_corrected = apply_geometric_correction(filtered_data)
    
    # Step 4: Cloud masking (existing)
    final_data = apply_cloud_masking(geometrically_corrected)
    
    return final_data
```

### **Phase 3: Feature Detection Algorithm**

#### **3.1 Derivative-Based Feature Detection**
```python
import scipy.signal

def calculate_spectral_derivatives(spectrum, band_centers):
    """
    Calculate first-order derivatives using Savitzky-Golay filter
    Based on Uhl et al. (2016) methodology
    """
    
    # Savitzky-Golay smoothing and differentiation
    # 7-point window, 2nd degree polynomial
    smoothed_spectrum = scipy.signal.savgol_filter(spectrum, 7, 2)
    first_derivative = scipy.signal.savgol_filter(spectrum, 7, 2, deriv=1)
    
    return smoothed_spectrum, first_derivative

def detect_kelp_features(spectrum, band_centers):
    """
    Detect kelp-specific spectral features
    Target wavelengths from research:
    - 528nm ± 18nm (fucoxanthin absorption)
    - 570nm ± 10nm (reflectance peak)
    """
    
    smoothed, derivative = calculate_spectral_derivatives(spectrum, band_centers)
    
    # Find zero-crossings in derivative (local maxima/minima)
    zero_crossings = []
    for i in range(len(derivative) - 1):
        if (derivative[i] > 0 and derivative[i+1] < 0) or \
           (derivative[i] < 0 and derivative[i+1] > 0):
            # Interpolate exact zero-crossing location
            zero_point = band_centers[i] + \
                        (band_centers[i+1] - band_centers[i]) * \
                        abs(derivative[i]) / (abs(derivative[i]) + abs(derivative[i+1]))
            zero_crossings.append(zero_point)
    
    # Check for kelp-specific features
    kelp_features = {
        'fucoxanthin_absorption': any(510 <= wl <= 546 for wl in zero_crossings),
        'reflectance_peak': any(560 <= wl <= 580 for wl in zero_crossings)
    }
    
    # Classify as kelp if both features present
    is_kelp = all(kelp_features.values())
    
    return is_kelp, kelp_features, zero_crossings
```

#### **3.2 Enhanced Kelp Detection**
```python
def enhanced_kelp_detection(image_data):
    """
    Enhanced kelp detection combining multiple approaches
    """
    
    # Method 1: NDRE-based detection (for submerged kelp)
    ndre_result = create_ndre_layer(image_data)
    
    # Method 2: Feature detection (for depth-invariant detection)
    feature_result = apply_feature_detection(image_data)
    
    # Method 3: Traditional NDVI (for comparison)
    ndvi_result = calculate_ndvi(image_data['B08'], image_data['B04'])
    
    # Combine results with confidence weighting
    combined_detection = combine_detection_methods(
        ndre_result, feature_result, ndvi_result
    )
    
    return {
        'kelp_probability': combined_detection,
        'method_comparison': {
            'ndre_detected': np.sum(ndre_result['kelp_mask']),
            'feature_detected': np.sum(feature_result['kelp_mask']),
            'ndvi_detected': np.sum(ndvi_result > 0.0),
            'enhancement_factor': np.sum(ndre_result['kelp_mask']) / max(1, np.sum(ndvi_result > 0.0))
        },
        'confidence_metrics': calculate_detection_confidence(combined_detection)
    }
```

## 🌊 **Environmental Correction Integration**

### **Tidal Height Correction**
```python
def apply_tidal_correction(detection_result, tidal_height):
    """
    Apply tidal height correction based on Timmer et al. (2024) findings:
    - Low current (<10 cm/s): 22.5% extent decrease per meter
    - High current (>10 cm/s): 35.5% extent decrease per meter
    """
    
    # Correction factors from research
    low_current_factor = -0.225  # -22.5% per meter
    high_current_factor = -0.355  # -35.5% per meter
    
    # Apply correction (simplified - would need current speed data)
    corrected_extent = detection_result * (1 + low_current_factor * tidal_height)
    
    return np.clip(corrected_extent, 0, 1)  # Keep within valid range
```

## 📊 **Performance Monitoring**

### **Validation Metrics**
```python
def calculate_enhanced_metrics(predicted, ground_truth):
    """Calculate performance metrics for enhanced detection"""
    
    return {
        'detection_improvement': {
            'ndre_vs_ndvi_gain': calculate_detection_gain(),
            'depth_penetration': calculate_depth_metrics(),
            'submerged_kelp_accuracy': calculate_submerged_accuracy()
        },
        'traditional_metrics': {
            'precision': calculate_precision(predicted, ground_truth),
            'recall': calculate_recall(predicted, ground_truth),
            'iou': calculate_iou(predicted, ground_truth),
            'overall_accuracy': calculate_overall_accuracy(predicted, ground_truth)
        },
        'research_benchmarks': {
            'timmer_2022_comparison': compare_with_timmer_results(),
            'uhl_2016_comparison': compare_with_uhl_results()
        }
    }
```

## 🔧 **Implementation Timeline**

### **Week 1: Core Red-Edge Processing**
- [ ] Implement enhanced band loading
- [ ] Add NDRE calculation functions
- [ ] Create red-edge visualization layers
- [ ] Update API endpoints

### **Week 2: Water Anomaly Filter**
- [ ] Implement WAF algorithm
- [ ] Integrate with preprocessing pipeline
- [ ] Add anomaly detection logging
- [ ] Performance optimization

### **Week 3: Feature Detection**
- [ ] Implement derivative-based detection
- [ ] Add spectral feature identification
- [ ] Create combined detection algorithms
- [ ] Validation framework setup

### **Week 4: Integration & Testing**
- [ ] Full system integration
- [ ] Performance benchmarking
- [ ] Documentation updates
- [ ] User interface enhancements

## 📈 **Expected Performance Improvements**

Based on research findings:
- **🎯 Detection Depth**: 90-100cm (vs 30-50cm with NDVI)
- **🎯 Extent Detection**: +18% more kelp area detected
- **🎯 Submerged Kelp**: Significant improvement in submerged canopy detection
- **🎯 Overall Accuracy**: Target 80%+ (vs 57.66% with traditional methods)
- **🎯 False Positive Reduction**: WAF reduces sunglint and surface artifacts

## 🔄 **Integration with Existing System**

### **Modified Files**
- `src/kelpie_carbon_v1/imagery.py` - Core processing enhancements
- `src/kelpie_carbon_v1/api/imagery_routes.py` - API endpoint updates
- `src/kelpie_carbon_v1/models/imagery.py` - Response model updates
- `static/js/imagery.js` - Frontend visualization updates

### **New Files**
- `src/kelpie_carbon_v1/processing/red_edge.py` - Red-edge specific processing
- `src/kelpie_carbon_v1/processing/water_anomaly_filter.py` - WAF implementation
- `src/kelpie_carbon_v1/processing/feature_detection.py` - Spectral feature detection
- `src/kelpie_carbon_v1/validation/enhanced_metrics.py` - Performance validation

### **Configuration Updates**
```python
# config.py additions
RED_EDGE_CONFIG = {
    'primary_band': 'B06',  # 740nm
    'secondary_band': 'B05',  # 705nm
    'reference_band': 'B04',  # 665nm
    'ndre_threshold': 0.0,
    'waf_kernel_size': 5,
    'enable_feature_detection': True,
    'tidal_correction': True
}
```

---

**Status**: 📋 **Specification Complete**
**Next**: Implementation Phase 1 - Core Red-Edge Processing
**Timeline**: 4 weeks for full implementation and testing 