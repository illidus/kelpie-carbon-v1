# Satellite Imagery Visualization Feature

## Overview

The Satellite Imagery Visualization Feature transforms the Kelpie Carbon v1 frontend from a text-based interface to a rich visual satellite imagery display. Users can now see the actual Sentinel-2 satellite data being analyzed, including true-color composites, spectral indices, and kelp detection overlays.

## Feature Scope

### 🎯 Primary Goals
- Display real Sentinel-2 satellite imagery on the interactive map
- Visualize spectral indices (NDVI, FAI, NDRE) as false-color composites
- Show kelp detection masks and water masks as colored overlays
- Provide before/after analysis visualization
- Enable layer switching and opacity controls

### 🚀 User Experience
1. **Before Analysis**: User sees base map with AOI selection
2. **During Analysis**: Loading indicator with progress updates
3. **After Analysis**: Rich satellite imagery layers with:
   - True-color RGB composite
   - Spectral index visualizations
   - Kelp detection overlay
   - Analysis result summary

### 📊 Visual Components

#### 1. Satellite Base Layer
- **True-Color Composite**: RGB visualization using Red, Green, Blue bands
- **False-Color Composite**: NIR-Red-Green for vegetation enhancement
- **Coverage**: Matches the analyzed AOI extent

#### 2. Spectral Index Overlays
- **NDVI Layer**: Green-to-red gradient showing vegetation health
- **FAI Layer**: Blue-to-yellow gradient for floating algae detection
- **Red Edge NDVI**: Enhanced vegetation index for kelp forests
- **NDRE Layer**: Red edge normalized difference for biomass assessment

#### 3. Analysis Result Overlays
- **Kelp Detection Mask**: Semi-transparent green overlay
- **Water Mask**: Semi-transparent blue overlay
- **Cloud Mask**: Semi-transparent gray overlay
- **Biomass Heatmap**: Color-coded biomass density visualization

#### 4. Interactive Controls
- **Layer Switcher**: Toggle between different visualizations
- **Opacity Sliders**: Adjust transparency of overlays
- **Legend**: Color scale explanations for each layer
- **Metadata Panel**: Acquisition date, resolution, cloud coverage

## Technical Architecture

### Backend Components

#### 1. Image Processing Module (`src/kelpie_carbon_v1/imagery/`)
```
imagery/
├── __init__.py
├── generators.py          # RGB composite generation
├── overlays.py           # Mask and heatmap generation  
├── tiles.py              # Map tile generation
└── utils.py              # Color mapping utilities
```

#### 2. API Endpoints
- `GET /api/imagery/{analysis_id}/rgb` - True-color composite
- `GET /api/imagery/{analysis_id}/false-color` - False-color composite
- `GET /api/imagery/{analysis_id}/ndvi` - NDVI visualization
- `GET /api/imagery/{analysis_id}/fai` - FAI visualization
- `GET /api/imagery/{analysis_id}/kelp-mask` - Kelp detection overlay
- `GET /api/imagery/{analysis_id}/metadata` - Image metadata

#### 3. Data Storage
- Temporary image cache for analysis results
- Compressed PNG/JPEG formats for web delivery
- GeoTIFF preservation for full-resolution data

### Frontend Components

#### 1. Map Layer Management (`static/layers.js`)
```javascript
class SatelliteLayerManager {
  constructor(map) { }
  addRGBLayer(analysisId) { }
  addSpectralLayer(type, analysisId) { }
  addMaskOverlay(type, analysisId) { }
  toggleLayer(layerId) { }
  setOpacity(layerId, opacity) { }
}
```

#### 2. UI Controls (`static/controls.js`)
```javascript
class ImageryControls {
  constructor(container) { }
  createLayerSwitcher() { }
  createOpacitySliders() { }
  createLegend() { }
  createMetadataPanel() { }
}
```

#### 3. Enhanced Interface
- Layer control panel in sidebar
- Opacity adjustment sliders
- Dynamic legend based on active layers
- Metadata display for current imagery

## Implementation Plan

### Phase 1: Core Image Generation (Week 1)
- [ ] RGB composite generation from Sentinel-2 bands
- [ ] Basic API endpoints for image serving
- [ ] Simple frontend image overlay

### Phase 2: Spectral Visualizations (Week 2)
- [ ] NDVI, FAI, NDRE false-color generation
- [ ] Color mapping utilities
- [ ] Layer switching frontend controls

### Phase 3: Analysis Overlays (Week 3)
- [ ] Kelp detection mask visualization
- [ ] Water and cloud mask overlays
- [ ] Biomass heatmap generation

### Phase 4: Interactive Controls (Week 4)
- [ ] Opacity adjustment sliders
- [ ] Dynamic legend system
- [ ] Metadata display panel
- [ ] Layer management interface

### Phase 5: Performance & Polish (Week 5)
- [ ] Image caching and optimization
- [ ] Progressive loading
- [ ] Error handling and fallbacks
- [ ] Comprehensive testing

## Technical Specifications

### Image Generation

#### RGB Composite
```python
def generate_rgb_composite(dataset: xr.Dataset) -> PIL.Image:
    """Generate true-color RGB composite from Sentinel-2 bands."""
    red = normalize_band(dataset['red'])
    green = normalize_band(dataset['green'])  # If available
    blue = normalize_band(dataset['blue'])    # If available
    return create_rgb_image(red, green, blue)
```

#### Spectral Index Visualization
```python
def generate_spectral_visualization(
    index_data: xr.DataArray, 
    colormap: str = 'RdYlGn'
) -> PIL.Image:
    """Generate false-color visualization of spectral index."""
    normalized = normalize_to_0_1(index_data)
    colored = apply_colormap(normalized, colormap)
    return array_to_image(colored)
```

#### Mask Overlay
```python
def generate_mask_overlay(
    mask: xr.DataArray, 
    color: tuple = (0, 255, 0),
    alpha: float = 0.6
) -> PIL.Image:
    """Generate semi-transparent colored overlay from boolean mask."""
    rgba = create_colored_mask(mask, color, alpha)
    return array_to_image(rgba)
```

### API Response Format
```json
{
  "analysis_id": "abc123",
  "imagery": {
    "rgb_composite": "/api/imagery/abc123/rgb",
    "false_color": "/api/imagery/abc123/false-color",
    "spectral_indices": {
      "ndvi": "/api/imagery/abc123/ndvi",
      "fai": "/api/imagery/abc123/fai",
      "ndre": "/api/imagery/abc123/ndre"
    },
    "overlays": {
      "kelp_mask": "/api/imagery/abc123/kelp-mask",
      "water_mask": "/api/imagery/abc123/water-mask",
      "cloud_mask": "/api/imagery/abc123/cloud-mask"
    },
    "metadata": {
      "bounds": [-122.0, 36.5, -121.5, 37.0],
      "acquisition_date": "2023-08-15",
      "resolution": 10,
      "cloud_coverage": 5.2
    }
  }
}
```

### Frontend Layer Configuration
```javascript
const layerConfigs = {
  'rgb': {
    name: 'True Color',
    type: 'base',
    opacity: 1.0,
    visible: true
  },
  'ndvi': {
    name: 'NDVI',
    type: 'overlay',
    opacity: 0.7,
    visible: false,
    legend: 'ndvi_colorbar.png'
  },
  'kelp-mask': {
    name: 'Kelp Detection',
    type: 'overlay',
    opacity: 0.6,
    visible: true,
    color: '#00ff00'
  }
};
```

## Quality Assurance

### Testing Strategy
1. **Unit Tests**: Image generation functions
2. **Integration Tests**: API endpoint responses
3. **Visual Tests**: Image output validation
4. **Frontend Tests**: Layer management functionality
5. **Performance Tests**: Image loading times

### Test Scenarios
- Various satellite data quality levels
- Different AOI sizes and locations
- Edge cases (high cloud cover, no kelp detected)
- Mobile device compatibility
- Slow network conditions

## Performance Considerations

### Image Optimization
- **Format**: PNG for masks, JPEG for composites
- **Compression**: Balanced quality vs. file size
- **Resolution**: Appropriate for web display (max 2048x2048)
- **Caching**: Browser and server-side caching

### Loading Strategy
- **Progressive**: Load base layer first, then overlays
- **Lazy**: Only generate images when requested
- **Fallback**: Graceful degradation if imagery fails

## User Interface Mockup

```
┌─────────────────────────────────────────────────────────┐
│ 🌊 Kelpie Carbon v1                                     │
│ Kelp Forest Carbon Sequestration Assessment             │
├─────────────────────────────────────────────────────────┤
│ Controls: [Date] [Date] [Run Analysis]                  │
├─────────────────────────────────────────────────────────┤
│                    │                                   │
│ Layer Controls     │         Map Display               │
│                    │                                   │
│ ☑ True Color       │    [Satellite Imagery with]      │
│ ☐ False Color      │    [Kelp Detection Overlay]      │
│ ☑ NDVI            │                                   │
│ ☑ Kelp Detection   │                                   │
│                    │                                   │
│ Opacity:           │                                   │
│ NDVI     [████▓▓▓] │                                   │
│ Kelp     [██████▓] │                                   │
│                    │                                   │
│ Legend:            │                                   │
│ [NDVI Color Bar]   │                                   │
│ [Kelp: Green]      │                                   │
│                    │                                   │
├────────────────────┼───────────────────────────────────┤
│ Metadata:          │ Analysis Results:                 │
│ Date: 2023-08-15   │ Biomass: 4,930 kg/ha            │
│ Resolution: 10m    │ Carbon: 0.17 kg C/m²            │
│ Clouds: 5.2%       │ Confidence: 0.70                 │
└────────────────────┴───────────────────────────────────┘
```

## Success Metrics

### User Experience
- **Visual Clarity**: Users can clearly identify kelp forests
- **Intuitive Controls**: Layer switching without confusion
- **Performance**: < 3 seconds for imagery loading
- **Accessibility**: Compatible with screen readers

### Technical Performance
- **Image Quality**: Scientifically accurate visualizations
- **Load Times**: < 2 seconds for initial display
- **Memory Usage**: < 50MB browser memory footprint
- **Error Rate**: < 1% imagery generation failures

## Future Enhancements

### Advanced Visualizations
- **Time Series**: Before/after comparison sliders
- **3D Visualization**: Biomass as elevation data
- **Animation**: Seasonal kelp growth patterns
- **Export**: High-resolution image downloads

### Analysis Integration
- **Interactive Selection**: Click to get pixel-level data
- **Measurement Tools**: Distance and area calculations
- **Reporting**: PDF reports with embedded imagery
- **Comparison**: Side-by-side analysis results

## Dependencies

### New Python Packages
```toml
[tool.poetry.dependencies]
pillow = "^10.0.0"           # Image processing
matplotlib = "^3.7.0"        # Colormaps
rasterio = "^1.3.0"         # Geospatial raster I/O
```

### Frontend Libraries
```html
<!-- Enhanced Leaflet plugins -->
<script src="leaflet.opacity.js"></script>
<script src="leaflet.layerswitcher.js"></script>
```

## Risk Mitigation

### Technical Risks
- **Memory Usage**: Large imagery processing
  - *Mitigation*: Chunked processing, image downsampling
- **Network Latency**: Slow image loading
  - *Mitigation*: Progressive loading, image compression
- **Browser Compatibility**: Old browser support
  - *Mitigation*: Polyfills, graceful degradation

### User Experience Risks
- **Complexity**: Too many controls
  - *Mitigation*: Default configurations, progressive disclosure
- **Performance**: Slow on mobile
  - *Mitigation*: Responsive design, mobile-optimized images

This comprehensive satellite imagery visualization feature will transform Kelpie Carbon v1 into a professional-grade satellite analysis tool with rich visual capabilities. 