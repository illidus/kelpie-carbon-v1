# 👥 Kelpie Carbon v1: User Guide

## **Welcome to Kelpie Carbon v1**

Kelpie Carbon v1 is an advanced satellite imagery analysis application designed to assess kelp forest carbon sequestration. This guide will help you navigate the application and perform comprehensive kelp forest analysis using real Sentinel-2 satellite data.

## **Getting Started**

### **Accessing the Application**
1. Open your web browser (Chrome, Firefox, Safari, or Edge)
2. Navigate to the application URL (e.g., `http://localhost:8000` for local deployment)
3. The application loads automatically - no login required

### **System Requirements**
- **Browser**: Modern web browser with JavaScript enabled
- **Internet**: Required for satellite data access
- **Screen**: 1024x768 minimum resolution (1920x1080 recommended)

---

## **User Interface Overview**

### **Main Components**

#### **🗺️ Interactive Map**
- **Center Panel**: Primary map interface with zoom and pan controls
- **Base Map**: OpenStreetMap tiles showing geographic context
- **Zoom Controls**: Located in the top-left corner of the map
- **Scale**: Distance reference in the bottom-left corner

#### **🎛️ Control Panel** (Top-left)
- **AOI Selection**: Click on map to select Area of Interest
- **Date Range**: Choose start and end dates for analysis
- **Run Analysis**: Execute the kelp forest assessment
- **Results Display**: View analysis outcomes and metrics

#### **📊 Layer Controls** (Appears after analysis)
- **Opacity Sliders**: Adjust transparency of each layer
- **Layer Toggle**: Show/hide individual analysis layers
- **Legend**: Color-coded explanation of map overlays
- **Metadata Panel**: Detailed information about satellite imagery

#### **⚡ Performance Dashboard** (Optional)
- **Keyboard Shortcut**: Press `Ctrl+Shift+P` to open
- **Real-time Metrics**: Monitor loading times and cache performance
- **Export Data**: Download performance metrics for analysis

---

## **Step-by-Step Analysis Guide**

### **Step 1: Select Area of Interest (AOI)**

#### **How to Select AOI**
1. **Navigate to Region**: Use map controls to zoom to your study area
2. **Click on Map**: Single-click to place a marker at your desired location
3. **Confirmation**: A red marker appears with coordinates displayed
4. **Adjust if Needed**: Click elsewhere to move the marker

#### **AOI Selection Tips**
- **Coastal Areas**: Choose locations known for kelp forests
- **Water Clarity**: Avoid areas with heavy sedimentation
- **Size Consideration**: Satellite pixels cover 10-60m areas
- **Popular Locations**:
  - California Coast: Monterey Bay, Channel Islands
  - Pacific Northwest: Puget Sound, British Columbia
  - Global: Tasmania, Chile, Norway

#### **Coordinate Reference**
- **Format**: Decimal degrees (WGS84)
- **Latitude Range**: -90° to +90° (negative = South)
- **Longitude Range**: -180° to +180° (negative = West)
- **Example**: Monterey Bay, CA = 36.8°N, -121.9°W

### **Step 2: Choose Date Range**

#### **Date Selection Guidelines**
- **Season**: Choose kelp growing season (varies by location)
  - **California**: April - October (peak: June - September)
  - **Tasmania**: October - April (Southern Hemisphere)
  - **Norway**: May - October
- **Cloud Coverage**: Avoid periods with frequent storms
- **Recent Data**: Sentinel-2 data available from 2015 onwards
- **Duration**: 1-3 months recommended for single analysis

#### **Setting Dates**
1. **Start Date**: Click start date field and select from calendar
2. **End Date**: Click end date field and select from calendar
3. **Validation**: System checks for valid date range
4. **Data Availability**: System searches for Sentinel-2 scenes

#### **Date Format**
- **Required Format**: YYYY-MM-DD
- **Example**: June 1, 2023 = `2023-06-01`
- **Time Zone**: All dates in UTC

### **Step 3: Run Analysis**

#### **Starting Analysis**
1. **Verify Settings**: Confirm AOI marker and date range
2. **Click "Run Analysis"**: Blue button in control panel
3. **Progress Tracking**: Status messages show analysis progress
4. **Wait Time**: 30-120 seconds depending on data complexity

#### **Analysis Process**
The application performs several automated steps:

1. **Satellite Data Search** (5-10 seconds)
   - Searches Microsoft Planetary Computer
   - Finds Sentinel-2 scenes for your area and dates
   - Selects best scene based on cloud coverage

2. **Data Processing** (20-40 seconds)
   - Downloads satellite imagery bands
   - Calculates spectral indices (NDVI, FAI, NDRE)
   - Runs machine learning kelp detection
   - Generates biomass estimates

3. **Visualization Generation** (10-20 seconds)
   - Creates RGB composite images
   - Generates spectral index visualizations
   - Produces analysis overlays and masks
   - Caches images for fast loading

4. **Results Preparation** (5-10 seconds)
   - Compiles analysis statistics
   - Prepares interactive controls
   - Updates user interface

#### **Progress Indicators**
- **Status Messages**: Text updates on current step
- **Loading Spinners**: Visual progress indicators
- **Console Logs**: Detailed progress (press F12 to view)

### **Step 4: Explore Results**

#### **Analysis Results Panel**
After analysis completion, the results panel displays:

- **Analysis ID**: Unique identifier for this analysis
- **AOI Coordinates**: Latitude and longitude of study area
- **Date Range**: Selected temporal period
- **Processing Time**: How long analysis took
- **Biomass Estimate**: Kelp biomass in tons per hectare
- **Carbon Sequestration**: Carbon storage in tons C per hectare
- **Satellite Scene Info**: Details about imagery used

#### **Understanding Results**
- **Biomass Values**:
  - `0-50 tons/ha`: Low kelp density
  - `50-100 tons/ha`: Moderate kelp density
  - `100-150 tons/ha`: High kelp density
  - `150+ tons/ha`: Very high kelp density

- **Carbon Storage**:
  - Typically 40-50% of biomass weight
  - Accounts for kelp carbon content
  - Represents carbon sequestered from atmosphere

---

## **Interactive Layer Controls**

### **Accessing Layer Controls**
- **Automatic Display**: Controls appear after analysis completion
- **Toggle Button**: Blue "Toggle Controls" button (top-left)
- **Panel Location**: Positioned to avoid map interference

### **Available Layers**

#### **🖼️ Base Layers**
- **RGB Composite**: True-color satellite image
  - Shows natural appearance of ocean and land
  - Useful for visual context and orientation
  - High contrast enhancement for water clarity

#### **📈 Spectral Index Layers**
- **NDVI (Normalized Difference Vegetation Index)**
  - Red to green color scale
  - Higher values = more vegetation
  - Useful for general vegetation health

- **FAI (Floating Algae Index)**
  - Blue to red color scale
  - Specifically designed for algae detection
  - Most sensitive to kelp forests

- **NDRE (Normalized Difference Red Edge)**
  - Purple to yellow color scale
  - Sensitive to vegetation stress
  - Useful for kelp health assessment

- **Kelp Index (Custom)**
  - Optimized specifically for kelp detection
  - Combines multiple spectral bands
  - Highest accuracy for kelp identification

#### **🎭 Analysis Overlays**
- **Kelp Detection Mask**
  - Green overlay showing detected kelp areas
  - Binary classification (kelp/no-kelp)
  - Based on machine learning model

- **Water Areas Mask**
  - Blue overlay showing water bodies
  - Helps distinguish water from land
  - Useful for quality control

- **Cloud Coverage Mask**
  - White/gray overlay showing clouds
  - Important for data quality assessment
  - Areas to interpret with caution

### **Layer Control Features**

#### **🎚️ Opacity Sliders**
- **Individual Control**: Each layer has its own opacity slider
- **Range**: 0% (invisible) to 100% (opaque)
- **Real-time Updates**: Changes apply immediately
- **Recommended Settings**:
  - RGB: 100% (base layer)
  - Spectral Indices: 60-80%
  - Masks: 50-70%

#### **👁️ Layer Visibility**
- **Toggle On/Off**: Click layer name to show/hide
- **Layer Order**: Higher layers overlay lower layers
- **Performance**: Hide unused layers for better performance

#### **🎨 Dynamic Legend**
- **Color Scales**: Shows color-to-value mapping
- **Units**: Displays appropriate units for each layer
- **Auto-Update**: Changes based on active layers

#### **📋 Metadata Panel**
Displays detailed information about the satellite imagery:
- **Scene ID**: Unique Sentinel-2 scene identifier
- **Acquisition Date**: When satellite captured the image
- **Cloud Coverage**: Percentage of scene covered by clouds
- **Processing Level**: Data processing level (L2A recommended)
- **Spatial Resolution**: Pixel size in meters
- **Spectral Bands**: Available electromagnetic spectrum bands

---

## **Interpreting Results**

### **Understanding Satellite Imagery**

#### **RGB Composite Interpretation**
- **Clear Water**: Deep blue to black colors
- **Shallow Water**: Lighter blue colors
- **Kelp Forests**: Dark patches in water (high absorption)
- **Land**: Brown, green, or gray depending on vegetation
- **Clouds**: Bright white areas
- **Cloud Shadows**: Dark areas near clouds

#### **Spectral Index Interpretation**

**NDVI (Normalized Difference Vegetation Index)**
- **Scale**: -1 to +1
- **Water**: Negative values (typically -0.5 to 0)
- **Kelp**: Positive values (0.1 to 0.6)
- **Land Vegetation**: High positive values (0.3 to 0.8)

**FAI (Floating Algae Index)**
- **Scale**: Varies by location and conditions
- **Clear Water**: Near zero or negative
- **Kelp Presence**: Positive values
- **Dense Kelp**: High positive values

### **Quality Assessment**

#### **Data Quality Indicators**
- **Cloud Coverage**: <10% excellent, 10-30% good, >30% poor
- **Sun Angle**: Higher angles provide better illumination
- **Sea State**: Calm conditions reduce surface reflection
- **Atmospheric Clarity**: Haze can affect spectral measurements

#### **Analysis Confidence**
- **High Confidence**: Clear water, low clouds, distinct kelp signatures
- **Medium Confidence**: Some clouds or atmospheric interference
- **Low Confidence**: High cloud cover, rough seas, or poor visibility

### **Biological Interpretation**

#### **Kelp Forest Characteristics**
- **Seasonal Variation**: Kelp biomass changes throughout year
- **Depth Dependency**: Most kelp grows in 5-40m depth
- **Species Differences**: Giant kelp vs. other kelp species
- **Environmental Factors**: Temperature, nutrients, light availability

#### **Carbon Sequestration Context**
- **Blue Carbon**: Ocean-based carbon storage
- **Kelp Contribution**: Significant but often underestimated
- **Temporal Dynamics**: Kelp forests can sequester carbon rapidly
- **Global Importance**: Kelp forests store carbon globally

---

## **Performance Features**

### **Progressive Loading**
The application uses smart loading strategies for optimal performance:

#### **Loading Priority**
1. **RGB Composite**: Loads first for immediate visual context
2. **Kelp Detection**: Primary analysis layer loads second
3. **Spectral Indices**: Additional analysis layers load progressively
4. **Masks**: Quality control layers load last

#### **Visual Feedback**
- **Loading Spinners**: Show progress for each layer
- **Progress Messages**: Describe current loading activity
- **Error Recovery**: Retry options if loading fails

### **Caching System**
The application automatically caches data for improved performance:

#### **Browser Cache**
- **Image Storage**: Processed images cached locally
- **Automatic Cleanup**: Old cache entries removed automatically
- **Size Management**: Cache size limited to prevent memory issues

#### **Cache Benefits**
- **Faster Reloading**: Subsequent views load instantly
- **Reduced Server Load**: Less data transfer required
- **Offline Viewing**: Some functionality available without internet

### **Performance Monitoring**
Access real-time performance metrics:

#### **Performance Dashboard**
- **Keyboard Shortcut**: `Ctrl+Shift+P`
- **Metrics Displayed**:
  - Page load time
  - Memory usage
  - Cache hit rate
  - API response times
- **Data Export**: Download metrics as JSON

#### **Console Logging**
For technical users, detailed logs available in browser console:
- **Press F12**: Open developer tools
- **Console Tab**: View detailed operation logs
- **Error Tracking**: See any issues with emoji indicators

---

## **Tips and Best Practices**

### **Selecting Optimal Study Areas**

#### **Ideal Characteristics**
- **Known Kelp Locations**: Areas with documented kelp forests
- **Clear Water**: Low sediment and pollution
- **Appropriate Depth**: 5-40 meters depth range
- **Minimal Development**: Away from urban runoff

#### **Coastal Regions to Try**
- **North America**: California coast, Pacific Northwest
- **Europe**: Norway, Scotland, Ireland
- **Southern Hemisphere**: Tasmania, southern Chile, South Africa
- **Asia**: Japan, South Korea (limited areas)

### **Timing Your Analysis**

#### **Seasonal Considerations**
- **Growing Season**: Choose peak biomass periods
- **Weather Patterns**: Avoid storm seasons
- **Cloud Coverage**: Check historical weather data
- **Tidal Considerations**: Low tide may expose more kelp

#### **Date Range Strategy**
- **Single Scene**: 1-2 week window for specific conditions
- **Seasonal Analysis**: 2-3 month window for seasonal patterns
- **Multi-temporal**: Compare same area across different seasons

### **Optimizing Performance**

#### **Browser Settings**
- **Modern Browser**: Use latest version of Chrome, Firefox, or Safari
- **JavaScript Enabled**: Required for application functionality
- **Pop-up Blocker**: Allow pop-ups for data export
- **Cache Settings**: Allow browser to cache images

#### **Internet Connection**
- **Stable Connection**: Required for satellite data download
- **Bandwidth**: Higher bandwidth improves loading speed
- **Connection Type**: Wired connection preferred over WiFi

### **Troubleshooting Common Issues**

#### **Analysis Fails to Start**
- **Check AOI Selection**: Ensure marker is placed
- **Verify Date Range**: Start date must be before end date
- **Internet Connection**: Verify connection to satellite data service
- **Browser Console**: Check for error messages (F12)

#### **No Satellite Data Found**
- **Expand Date Range**: Try longer time period
- **Check Location**: Some areas have limited coverage
- **Cloud Coverage**: Very cloudy periods may have no usable data
- **Historical Availability**: Sentinel-2 data starts from 2015

#### **Slow Loading Performance**
- **Clear Browser Cache**: Refresh and clear cache
- **Close Other Tabs**: Reduce memory usage
- **Check Internet Speed**: Slow connections affect performance
- **Wait for Cache**: First load is slower, subsequent loads faster

#### **Layers Not Displaying**
- **Check Opacity**: Ensure opacity is above 0%
- **Layer Order**: Lower layers may be hidden by upper layers
- **Zoom Level**: Some layers only visible at certain zoom levels
- **Wait for Loading**: New async loading system improves responsiveness
- **Refresh Page**: Try reloading the application if layers still don't appear

#### **Layer Loading Issues**
- **Improved Performance**: Layer switching now uses async loading for better responsiveness
- **Loading Indicators**: Watch for loading states during layer transitions
- **Geographic Bounds**: Layers automatically fetch proper geographic bounds before display
- **Error Recovery**: System automatically retries failed layer loads
- **Layer Names**: System properly maps internal names (kelp_mask → kelp, water_mask → water)

---

## **Scientific Background**

### **Kelp Forest Ecology**

#### **Kelp Forest Characteristics**
- **Habitat**: Temperate coastal waters worldwide
- **Depth Range**: Typically 5-40 meters depth
- **Productivity**: Among the most productive ecosystems on Earth
- **Biodiversity**: Support diverse marine communities

#### **Carbon Sequestration Process**
1. **Photosynthesis**: Kelp absorbs CO₂ from seawater
2. **Biomass Storage**: Carbon incorporated into kelp tissue
3. **Detritus Export**: Dead kelp transports carbon to deep ocean
4. **Sediment Burial**: Some carbon permanently stored in sediments

### **Remote Sensing Science**

#### **Satellite Technology**
- **Sentinel-2**: European Space Agency satellite constellation
- **Spectral Bands**: 13 bands from visible to near-infrared
- **Spatial Resolution**: 10-60 meters depending on band
- **Temporal Resolution**: 5-day revisit time

#### **Spectral Analysis Principles**
- **Spectral Signatures**: Each material has unique spectral properties
- **Kelp Detection**: Kelp absorbs strongly in red, reflects in near-infrared
- **Water Penetration**: Blue and green light penetrate deepest
- **Atmospheric Correction**: Required for accurate water analysis

### **Machine Learning Approach**

#### **Model Development**
- **Training Data**: Classified examples of kelp presence/absence
- **Feature Engineering**: Spectral indices and band ratios
- **Algorithm**: Random Forest classification
- **Validation**: Cross-validation with field data

#### **Accuracy Considerations**
- **Overall Accuracy**: Typically 80-90% for clear water conditions
- **False Positives**: May include other floating vegetation
- **False Negatives**: Deep or sparse kelp may be missed
- **Confidence Levels**: Model provides probability estimates

---

## **Data Export and Sharing**

### **Saving Results**

#### **Screenshot Capture**
- **Browser Tools**: Use browser screenshot functionality
- **Print to PDF**: Print page to save as PDF document
- **Screen Capture**: Use operating system screenshot tools

#### **Performance Data Export**
- **Dashboard Export**: Use performance dashboard export button
- **JSON Format**: Machine-readable format for further analysis
- **Metrics Included**: Loading times, cache statistics, memory usage

### **Sharing Analysis**

#### **URL Sharing**
- **Current State**: Browser URL contains current view
- **Bookmarking**: Save specific analysis for later reference
- **Social Sharing**: Share URLs with colleagues or collaborators

#### **Citation Information**
When using results in publications or reports:
- **Application**: Kelpie Carbon v1
- **Data Source**: Sentinel-2 satellite imagery via Microsoft Planetary Computer
- **Processing Date**: Include date of analysis
- **Limitations**: Note any data quality or coverage limitations

---

## **Frequently Asked Questions (FAQ)**

### **General Questions**

**Q: What areas can I analyze?**
A: Any coastal area worldwide with kelp forests. The application works best for temperate coastal regions with clear water and known kelp populations.

**Q: How recent is the satellite data?**
A: Sentinel-2 data is typically available within 1-3 days of capture. The application automatically selects the best available scene for your date range.

**Q: Can I analyze multiple locations?**
A: Currently, each analysis covers one location. Run separate analyses for different study areas.

**Q: Is there a cost to use the application?**
A: The application is free to use. Satellite data is provided through Microsoft's Planetary Computer initiative.

### **Technical Questions**

**Q: What browsers are supported?**
A: Modern versions of Chrome, Firefox, Safari, and Edge. JavaScript must be enabled.

**Q: Why is my analysis taking a long time?**
A: Processing time depends on data complexity and internet speed. Large areas or cloudy conditions increase processing time.

**Q: Can I use the application offline?**
A: No, internet connection is required for satellite data access. However, previously loaded layers are cached for faster viewing.

**Q: How accurate are the results?**
A: Accuracy is typically 80-90% for clear water conditions. Accuracy may be lower in turbid water or during adverse weather conditions.

### **Data Questions**

**Q: What's the spatial resolution of the analysis?**
A: Results are based on 10-20 meter satellite pixels. Features smaller than this may not be detected.

**Q: Can I download the raw satellite data?**
A: The application processes data automatically. For raw data access, visit Microsoft Planetary Computer directly.

**Q: How do I interpret negative biomass values?**
A: Negative values indicate no kelp detected. This is normal for areas without kelp forests.

**Q: Why do results vary between similar dates?**
A: Natural variation in kelp growth, tidal conditions, and atmospheric conditions can affect results.

---

## **Support and Resources**

### **Getting Help**
- **Documentation**: Comprehensive guides available in the `/docs` directory
- **GitHub Issues**: Report bugs or request features
- **Community Forum**: Connect with other users and researchers

### **Additional Resources**
- **Kelp Forest Research**: Links to scientific publications
- **Satellite Data Access**: Microsoft Planetary Computer documentation
- **Remote Sensing Tutorials**: Educational resources for satellite imagery analysis

### **Contact Information**
- **Technical Support**: GitHub Issues for technical problems
- **Scientific Questions**: Contact marine ecology research communities
- **Feature Requests**: Submit ideas through GitHub Discussions

---

# 📈 **Performance Optimizations**

The Kelpie Carbon v1 application includes several performance optimizations to ensure efficient operation:

## **🚀 Server Performance**

### **Selective File Watching**
- Hot reload is optimized to watch only Python files in the source directory
- Excludes documentation, test files, and temporary files
- Reduces CPU usage by ~80% during development

### **Smart Caching**
- Intelligent cache management with size limits (500MB default)
- LRU (Least Recently Used) eviction strategy
- Automatic cleanup prevents memory leaks

### **Security Headers**
- Comprehensive security headers added to all responses
- Content Security Policy for XSS protection
- HSTS headers for HTTPS deployments
- Frame options to prevent clickjacking

## **⚡ API Optimizations**

### **Image Response Caching**
- HTTP caching headers for satellite imagery
- ETags for efficient browser caching
- Optimized PNG/JPEG compression

### **Constants Management**
- Centralized constants for maintainability
- No more magic numbers scattered in code
- Easy configuration of thresholds and limits

### **Future-Proof Dependencies**
- Updated to use modern API methods
- Eliminated FutureWarnings from dependencies
- Better long-term compatibility

## **🔧 Configuration**

Most optimizations work automatically, but you can configure:

```yaml
# Custom cache limits (if needed)
cache:
  max_size_mb: 1000  # Increase for larger datasets
  max_items: 200     # Increase for more concurrent users
```

See **constants.py** for all configurable values.

## **📊 Performance Monitoring**

Monitor performance using:
- Application logs for processing times
- Cache hit/miss ratios in debug mode
- Memory usage through system monitoring

---

This user guide provides comprehensive instruction for using the Kelpie Carbon v1 application effectively. Whether you're a marine researcher, environmental manager, or curious citizen scientist, these guidelines will help you conduct meaningful kelp forest carbon assessments using cutting-edge satellite technology. 