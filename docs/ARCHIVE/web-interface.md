# 🌊 Web Interface

The Kelpie Carbon v1 web interface provides a sophisticated, interactive map-based tool for kelp forest carbon assessment with real-time satellite data analysis.

## ✨ Features

### 🗺️ Interactive Map Interface
- **Responsive Map**: Built with Leaflet.js, optimized for all screen sizes
- **Geographic Context**: OpenStreetMap base layer with zoom and pan controls
- **Smart Marker Placement**: Single-click AOI selection with coordinate display
- **Visual Feedback**: Real-time status updates and progress indicators

### 📊 Advanced Layer Management
- **Dynamic Layer Controls**: Toggle and adjust opacity for multiple analysis layers
- **Async Layer Loading**: Progressive loading with priority-based rendering
- **Smart Bounds Management**: Automatic geographic bounds fetching for proper layer positioning
- **Layer Name Mapping**: Intelligent mapping between API responses and display names

### 🛰️ Satellite Data Integration
- **Real-time Analysis**: Direct integration with Sentinel-2 satellite data
- **Multiple Visualizations**: RGB, spectral indices, and analysis overlays
- **Quality Filtering**: Automatic cloud and water masking
- **Performance Optimization**: Image caching and progressive loading

## 📱 User Interface Components

### **Map Controls**
- **Zoom Controls**: Standard zoom in/out buttons
- **Layer Toggle**: Dynamic controls appear after analysis completion
- **Opacity Sliders**: Fine-tune transparency for each layer
- **Legend Display**: Color-coded explanations for all visualizations

### **Analysis Panel**
- **AOI Selection**: Click-to-select with coordinate validation
- **Date Range Picker**: Intuitive date selection with format validation
- **Run Analysis Button**: One-click analysis execution
- **Results Display**: Comprehensive analysis metrics and metadata

### **Performance Dashboard** *(Optional)*
- **Keyboard Shortcut**: `Ctrl+Shift+P` to open
- **Real-time Metrics**: Layer loading times and cache efficiency
- **Memory Management**: Browser cache status and cleanup

## 🔄 Layer System

### **Available Layer Types**

#### **Base Layers**
- **RGB Composite**: True-color satellite imagery for visual context
- **False Color**: Enhanced vegetation visualization using NIR band

#### **Spectral Index Layers**
- **NDVI**: Normalized Difference Vegetation Index (red to green scale)
- **FAI**: Floating Algae Index (blue to red scale, optimized for kelp)
- **NDRE**: Normalized Difference Red Edge Index (purple to yellow scale)
- **Kelp Index**: Custom index optimized for kelp forest detection

#### **Analysis Overlays**
- **Kelp Detection**: Machine learning-based kelp forest identification
- **Water Mask**: Automated water body detection and masking
- **Cloud Coverage**: Cloud detection and quality assessment

### **Layer Management Features**
- **Asynchronous Loading**: Non-blocking layer creation with proper error handling
- **Priority Ordering**: RGB loads first, followed by masks, then spectral indices
- **Geographic Accuracy**: Proper bounds fetching ensures correct positioning
- **Name Resolution**: Automatic mapping (e.g., `kelp_mask` → `kelp` for display)

## 🔌 API Integration

### **Backend Endpoints**
```
GET  /                              # Serves web interface
POST /api/run                       # Execute analysis pipeline
GET  /api/imagery/{id}/rgb          # RGB composite image
GET  /api/imagery/{id}/mask/{type}  # Analysis masks (kelp, water, cloud)
GET  /api/imagery/{id}/index/{type} # Spectral indices (ndvi, fai, ndre)
GET  /api/imagery/{id}/metadata     # Image metadata and bounds
```

### **Request Format**
```json
{
  "aoi": {"lat": 49.2827, "lng": -123.1207},
  "start_date": "2023-08-01",
  "end_date": "2023-08-31"
}
```

### **Response Format**
```json
{
  "analysis_id": "abc123ef-4567-89gh-ijkl-mnopqrstuvwx",
  "status": "completed",
  "processing_time": "15.32s",
  "aoi": {"lat": 49.2827, "lng": -123.1207},
  "date_range": {"start": "2023-08-01", "end": "2023-08-31"},
  "biomass": "125.7 tons/ha",
  "carbon": "62.85 tons C/ha",
  "satellite_scene": "S2A_MSIL2A_20230816T191911_R099_T10UDV_20230817T042129",
  "cloud_coverage": "0.001307%",
  "available_layers": [
    "rgb", "false_color", "ndvi", "fai", "ndre",
    "kelp_mask", "water_mask", "cloud_mask"
  ]
}
```

## 🛠️ Technology Stack

### **Frontend Technologies**
- **Core**: HTML5, CSS3, JavaScript (ES6+)
- **Mapping**: Leaflet.js 1.9.4 with OpenStreetMap tiles
- **Async Operations**: Modern Promise/async-await patterns
- **Error Handling**: Exponential backoff retry mechanisms

### **Backend Integration**
- **API**: FastAPI with static file serving
- **Data Processing**: Real-time satellite data analysis
- **Caching**: Optimized image caching and delivery

### **Performance Features**
- **Progressive Loading**: Priority-based layer rendering
- **Memory Management**: Automatic cleanup of blob URLs
- **Cache Optimization**: Browser-side image caching
- **Error Recovery**: Robust retry mechanisms with fallbacks

## 📋 Usage Workflow

### **1. Access Interface**
1. Navigate to application URL (e.g., `http://localhost:8001`)
2. Interactive map loads automatically
3. Default view centers on British Columbia, Canada

### **2. Select Analysis Area**
1. Navigate to desired coastal region using map controls
2. Single-click on map to place AOI marker
3. Coordinates display in popup for verification
4. Click elsewhere to reposition if needed

### **3. Configure Analysis**
1. Set start date using date picker (format: YYYY-MM-DD)
2. Set end date (must be after start date)
3. System validates date range and format
4. Default range provides 1-month analysis window

### **4. Execute Analysis**
1. Click "Run Analysis" button to start processing
2. Status messages show real-time progress:
   - Searching satellite data (5-10s)
   - Processing imagery (20-40s)
   - Generating visualizations (10-20s)
   - Preparing results (5-10s)

### **5. Explore Results**
1. **Layer Controls** automatically appear upon completion
2. Toggle different visualization layers on/off
3. Adjust opacity for optimal viewing
4. Review analysis metrics in results panel
5. Use legend to interpret color-coded overlays

## 🔧 Browser Compatibility

### **Supported Browsers**
- **Chrome**: 90+ (Recommended)
- **Firefox**: 88+
- **Safari**: 14+
- **Edge**: 90+

### **Required Features**
- JavaScript ES6+ support
- Fetch API
- Promises/async-await
- CSS Grid and Flexbox

## 🚀 Performance Optimization

### **Loading Strategy**
- **Base layers load first** (RGB, false color)
- **Analysis overlays load second** (kelp, water masks)
- **Spectral indices load last** (NDVI, FAI, NDRE)

### **Memory Management**
- Automatic cleanup of blob URLs
- Cache size limits to prevent memory leaks
- Progressive garbage collection

### **Network Optimization**
- Image compression and format optimization
- Retry mechanisms with exponential backoff
- Efficient caching strategies

## 🎯 Best Practices

### **For Optimal Performance**
1. Use Chrome or Firefox for best experience
2. Ensure stable internet connection for satellite data
3. Close unused browser tabs to free memory
4. Clear browser cache if experiencing issues

### **For Accurate Results**
1. Select coastal areas known for kelp forests
2. Choose clear-weather periods (low cloud coverage)
3. Use growing season dates for your region
4. Allow full analysis completion before exploring layers

## 🆕 Recent Improvements

- ✅ **Fixed layer switching functionality** - All layers now display correctly
- ✅ **Enhanced async layer loading** - Non-blocking, progressive rendering
- ✅ **Improved geographic accuracy** - Proper bounds fetching for all layers
- ✅ **Better error handling** - Robust retry mechanisms and user feedback
- ✅ **Performance optimizations** - Faster loading and reduced memory usage

---

*For technical implementation details, see [Developer Guide](DEVELOPMENT_GUIDE.md) and [API Reference](API_REFERENCE.md).*
