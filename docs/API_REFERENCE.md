# 📡 Kelpie Carbon v1: API Reference

## **Overview**

The Kelpie Carbon v1 API provides RESTful endpoints for kelp forest carbon sequestration analysis using satellite imagery. The API is built with FastAPI and provides automatic OpenAPI documentation.

## **Base URL**
```
http://localhost:8000
```

## **Interactive Documentation**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## **Endpoints Overview**

### **Analysis Endpoints**
- `POST /api/run` - Run traditional carbon analysis
- `POST /api/imagery/analyze-and-cache` - Generate and cache satellite imagery

### **Imagery Endpoints**
- `GET /api/imagery/{analysis_id}/metadata` - Get imagery metadata
- `GET /api/imagery/{analysis_id}/rgb` - Get RGB composite image
- `GET /api/imagery/{analysis_id}/spectral/{index}` - Get spectral index visualization
- `GET /api/imagery/{analysis_id}/mask/{mask_type}` - Get analysis mask overlay
- `GET /api/imagery/{analysis_id}/biomass` - Get biomass heatmap

### **Static Endpoints**
- `GET /` - Main web interface
- `GET /static/{file_path}` - Static assets (CSS, JS, images)

---

## **Analysis Endpoints**

### **POST /api/run**

Run traditional kelp forest carbon sequestration analysis.

#### **Request Body**
```json
{
  "aoi": {
    "lat": 34.4140,
    "lng": -119.8489
  },
  "start_date": "2023-06-01",
  "end_date": "2023-08-31"
}
```

#### **Parameters**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `aoi.lat` | float | Yes | Latitude of area of interest (-90 to 90) |
| `aoi.lng` | float | Yes | Longitude of area of interest (-180 to 180) |
| `start_date` | string | Yes | Start date in YYYY-MM-DD format |
| `end_date` | string | Yes | End date in YYYY-MM-DD format |

#### **Response**
```json
{
  "analysis_id": "b8c9d2e1-f3a4-5b6c-7d8e-9f0a1b2c3d4e",
  "status": "completed",
  "processing_time": "45.23 seconds",
  "biomass": "156.7 tons/hectare",
  "carbon": "78.35 tons C/hectare",
  "aoi_coordinates": {
    "lat": 34.4140,
    "lng": -119.8489
  },
  "date_range": {
    "start": "2023-06-01",
    "end": "2023-08-31"
  },
  "satellite_scenes": [
    {
      "scene_id": "S2A_MSIL2A_20230816T191911_R099_T10UDV_20230817T042129",
      "date": "2023-08-16T19:19:11.024000Z",
      "cloud_coverage": 0.001307
    }
  ]
}
```

#### **Status Codes**
- `200 OK` - Analysis completed successfully
- `422 Unprocessable Entity` - Invalid input parameters
- `500 Internal Server Error` - Analysis processing failed

#### **Example Usage**
```bash
curl -X POST "http://localhost:8000/api/run" \
  -H "Content-Type: application/json" \
  -d '{
    "aoi": {"lat": 34.4140, "lng": -119.8489},
    "start_date": "2023-06-01",
    "end_date": "2023-08-31"
  }'
```

---

### **POST /api/imagery/analyze-and-cache**

Generate and cache satellite imagery visualizations for the specified area and time period.

#### **Request Body**
```json
{
  "aoi": {
    "lat": 34.4140,
    "lng": -119.8489
  },
  "start_date": "2023-06-01",
  "end_date": "2023-08-31",
  "buffer_km": 1.0,
  "max_cloud_cover": 0.3
}
```

#### **Parameters**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `aoi.lat` | float | Yes | Latitude of area of interest (-90 to 90) |
| `aoi.lng` | float | Yes | Longitude of area of interest (-180 to 180) |
| `start_date` | string | Yes | Start date in YYYY-MM-DD format |
| `end_date` | string | Yes | End date in YYYY-MM-DD format |
| `buffer_km` | float | No | Buffer distance in kilometers (default: 1.0) |
| `max_cloud_cover` | float | No | Maximum cloud coverage (0.0-1.0, default: 0.3) |

#### **Response**
```json
{
  "analysis_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "success",
  "message": "Imagery analysis completed and cached",
  "available_layers": {
    "base_layers": ["rgb"],
    "spectral_indices": ["ndvi", "fai", "ndre", "kelp_index"],
    "masks": ["kelp", "water", "cloud"],
    "biomass": true
  },
  "processing_info": {
    "satellite_scenes": 1,
    "processing_time": 28.45,
    "cache_expires": "2024-01-15T18:30:00Z"
  }
}
```

#### **Status Codes**
- `200 OK` - Imagery generated and cached successfully
- `422 Unprocessable Entity` - Invalid coordinates or date range
- `500 Internal Server Error` - Imagery processing failed

---

## **Imagery Endpoints**

### **GET /api/imagery/{analysis_id}/metadata**

Get metadata information for a cached imagery analysis.

#### **Path Parameters**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `analysis_id` | string | Yes | Unique identifier for the analysis |

#### **Response**
```json
{
  "analysis_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "coordinates": {
    "lat": 34.4140,
    "lng": -119.8489
  },
  "date_range": {
    "start": "2023-06-01",
    "end": "2023-08-31"
  },
  "satellite_info": {
    "scene_id": "S2A_MSIL2A_20230816T191911_R099_T10UDV_20230817T042129",
    "acquisition_date": "2023-08-16T19:19:11.024000Z",
    "cloud_coverage": 0.001307,
    "resolution": "10m"
  },
  "available_layers": {
    "base_layers": ["rgb"],
    "spectral_indices": ["ndvi", "fai", "ndre", "kelp_index"],
    "masks": ["kelp", "water", "cloud"],
    "biomass": true
  },
  "processing_info": {
    "created": "2024-01-15T17:30:00Z",
    "expires": "2024-01-15T18:30:00Z",
    "processing_time": 28.45
  }
}
```

#### **Status Codes**
- `200 OK` - Metadata retrieved successfully
- `404 Not Found` - Analysis ID not found or expired

---

### **GET /api/imagery/{analysis_id}/rgb**

Get the RGB composite image for the specified analysis.

#### **Path Parameters**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `analysis_id` | string | Yes | Unique identifier for the analysis |

#### **Response**
- **Content-Type**: `image/jpeg`
- **Body**: Binary image data (JPEG format, optimized for web display)
- **Headers**:
  - `Cache-Control: public, max-age=3600, stale-while-revalidate=86400`
  - `ETag: "unique-hash-value"`
  - `Content-Length: image-size-in-bytes`

#### **Status Codes**
- `200 OK` - Image generated successfully
- `404 Not Found` - Analysis ID not found
- `422 Unprocessable Entity` - Invalid data for RGB generation
- `500 Internal Server Error` - Image generation failed

#### **Example Usage**
```html
<img src="http://localhost:8000/api/imagery/a1b2c3d4-e5f6-7890-abcd-ef1234567890/rgb" 
     alt="RGB Composite" />
```

---

### **GET /api/imagery/{analysis_id}/spectral/{index}**

Get spectral index visualization for the specified analysis.

#### **Path Parameters**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `analysis_id` | string | Yes | Unique identifier for the analysis |
| `index` | string | Yes | Spectral index type |

#### **Available Spectral Indices**
| Index | Description | Use Case |
|-------|-------------|----------|
| `ndvi` | Normalized Difference Vegetation Index | General vegetation health |
| `fai` | Floating Algae Index | Algae and kelp detection |
| `ndre` | Normalized Difference Red Edge | Vegetation stress analysis |
| `kelp_index` | Custom Kelp Index | Optimized kelp detection |

#### **Response**
- **Content-Type**: `image/png`
- **Body**: Binary image data (PNG format with transparency)
- **Headers**: Standard caching headers

#### **Status Codes**
- `200 OK` - Spectral index image generated successfully
- `404 Not Found` - Analysis ID not found or index not available
- `422 Unprocessable Entity` - Invalid spectral index type
- `500 Internal Server Error` - Image generation failed

#### **Example Usage**
```javascript
const layerUrl = `http://localhost:8000/api/imagery/${analysisId}/spectral/fai`;
const spectralLayer = L.imageOverlay(layerUrl, bounds, { opacity: 0.7 });
```

---

### **GET /api/imagery/{analysis_id}/mask/{mask_type}**

Get analysis mask overlay for the specified type.

#### **Path Parameters**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `analysis_id` | string | Yes | Unique identifier for the analysis |
| `mask_type` | string | Yes | Type of mask overlay |

#### **Query Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.6 | Transparency level (0.0 to 1.0) |

#### **Available Mask Types**
| Mask Type | Description | Color Scheme |
|-----------|-------------|--------------|
| `kelp` | Kelp forest detection | Green overlay |
| `water` | Water body identification | Blue overlay |
| `cloud` | Cloud coverage | White/gray overlay |

#### **Response**
- **Content-Type**: `image/png`
- **Body**: Binary image data (PNG format with alpha channel)
- **Headers**: Standard caching headers

#### **Status Codes**
- `200 OK` - Mask overlay generated successfully
- `404 Not Found` - Analysis ID not found or mask not available
- `422 Unprocessable Entity` - Invalid mask type or alpha value
- `500 Internal Server Error` - Mask generation failed

#### **Example Usage**
```javascript
const maskUrl = `http://localhost:8000/api/imagery/${analysisId}/mask/kelp?alpha=0.8`;
const kelpMask = L.imageOverlay(maskUrl, bounds, { opacity: 1.0 });
```

---

### **GET /api/imagery/{analysis_id}/biomass**

Get biomass heatmap visualization for the specified analysis.

#### **Path Parameters**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `analysis_id` | string | Yes | Unique identifier for the analysis |

#### **Response**
- **Content-Type**: `image/png`
- **Body**: Binary image data (PNG format with color scale)
- **Headers**: Standard caching headers

#### **Color Scale**
- **Blue**: Low biomass (0-50 tons/hectare)
- **Green**: Medium biomass (50-100 tons/hectare)
- **Yellow**: High biomass (100-150 tons/hectare)
- **Red**: Very high biomass (150+ tons/hectare)

#### **Status Codes**
- `200 OK` - Biomass heatmap generated successfully
- `404 Not Found` - Analysis ID not found
- `500 Internal Server Error` - Heatmap generation failed

---

## **Static Endpoints**

### **GET /**

Serve the main web interface.

#### **Response**
- **Content-Type**: `text/html`
- **Body**: Complete HTML application with embedded CSS and JavaScript

### **GET /static/{file_path}**

Serve static assets (CSS, JavaScript, images).

#### **Path Parameters**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | string | Yes | Path to static file |

#### **Available Static Files**
- `style.css` - Application styling
- `app.js` - Main application logic
- `layers.js` - Layer management
- `controls.js` - Interactive controls
- `loading.js` - Progressive loading system
- `performance.js` - Performance monitoring

---

## **Error Handling**

### **Error Response Format**
```json
{
  "detail": "Descriptive error message",
  "error_code": "ERROR_TYPE",
  "timestamp": "2024-01-15T17:30:00Z"
}
```

### **Common Error Codes**
| Code | Status | Description |
|------|--------|-------------|
| `INVALID_COORDINATES` | 422 | Latitude/longitude out of bounds |
| `INVALID_DATE_RANGE` | 422 | Invalid or impossible date range |
| `ANALYSIS_NOT_FOUND` | 404 | Analysis ID not found or expired |
| `SATELLITE_DATA_UNAVAILABLE` | 500 | No satellite data for specified area/time |
| `PROCESSING_FAILED` | 500 | Internal processing error |
| `INVALID_SPECTRAL_INDEX` | 422 | Unsupported spectral index type |
| `INVALID_MASK_TYPE` | 422 | Unsupported mask type |

---

## **Rate Limiting**

### **Current Limits**
- **Analysis Requests**: 10 per minute per IP
- **Image Requests**: 100 per minute per IP
- **Static Files**: Unlimited

### **Rate Limit Headers**
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when rate limit resets

---

## **Caching**

### **Cache Headers**
All image responses include appropriate cache headers:
- `Cache-Control: public, max-age=3600, stale-while-revalidate=86400`
- `ETag: "unique-hash-value"`
- `Last-Modified: timestamp`

### **Cache Validation**
Clients should send `If-None-Match` headers with ETags for efficient cache validation.

### **Cache Expiration**
- **Analysis Results**: 1 hour
- **Generated Images**: 1 hour
- **Static Files**: 1 day

---

## **Performance Considerations**

### **Image Optimization**
- **JPEG**: Used for RGB composites (better compression)
- **PNG**: Used for overlays and masks (transparency support)
- **Quality**: JPEG quality set to 85 for optimal size/quality balance

### **Async Processing**
All endpoints use async processing for non-blocking operations.

### **Memory Management**
Generated images are automatically cleaned up after cache expiration.

---

## **Usage Examples**

### **Complete Workflow**
```javascript
// 1. Run analysis
const analysisResponse = await fetch('/api/run', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    aoi: { lat: 34.4140, lng: -119.8489 },
    start_date: '2023-06-01',
    end_date: '2023-08-31'
  })
});
const analysisResult = await analysisResponse.json();

// 2. Generate imagery
const imageryResponse = await fetch('/api/imagery/analyze-and-cache', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    aoi: {
      lat: 34.4140,
      lng: -119.8489
    },
    start_date: '2023-06-01',
    end_date: '2023-08-31',
    buffer_km: 1.0,
    max_cloud_cover: 0.3
  })
});
const imageryResult = await imageryResponse.json();

// 3. Load layers
const analysisId = imageryResult.analysis_id;
const rgbLayer = L.imageOverlay(
  `/api/imagery/${analysisId}/rgb`, 
  bounds
);
const kelpMask = L.imageOverlay(
  `/api/imagery/${analysisId}/mask/kelp?alpha=0.7`, 
  bounds
);
```

### **Error Handling Example**
```javascript
try {
  const response = await fetch(`/api/imagery/${analysisId}/rgb`);
  if (!response.ok) {
    const error = await response.json();
    console.error('Error:', error.detail);
    // Handle specific error types
    if (response.status === 404) {
      // Analysis not found - redirect to new analysis
    } else if (response.status === 500) {
      // Server error - show retry option
    }
  }
} catch (error) {
  console.error('Network error:', error);
}
```

---

This API reference provides comprehensive documentation for integrating with the Kelpie Carbon v1 system. For interactive testing and exploration, visit the Swagger UI at http://localhost:8000/docs when the server is running. 