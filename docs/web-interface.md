# Web Interface

The Kelpie Carbon v1 web interface provides an interactive map-based tool for kelp forest carbon assessment.

## Features

### Map-based AOI Selection
- Interactive map centered on British Columbia, Canada
- Click on the map to select Area of Interest (AOI)
- Visual marker shows selected location
- Coordinates displayed in popup

### Date Range Selection
- Start and end date inputs for analysis period
- Default range: January 1, 2023 to December 31, 2023
- Validates date range before running analysis

### Analysis Execution
- "Run Analysis" button triggers pipeline
- Real-time status updates during processing
- Results displayed in organized format

## API Integration

The web interface integrates with the FastAPI backend through:

### Endpoints
- `GET /` - Serves the web interface
- `POST /api/run` - Executes analysis pipeline

### Request Format
```json
{
  "aoi": {"lat": 49.2827, "lng": -123.1207},
  "start_date": "2023-01-01",
  "end_date": "2023-12-31"
}
```

### Response Format
```json
{
  "analysis_id": "abc123ef",
  "status": "completed",
  "processing_time": "1.23s",
  "aoi": {"lat": 49.2827, "lng": -123.1207},
  "date_range": {"start": "2023-01-01", "end": "2023-12-31"},
  "biomass": "Pending implementation",
  "carbon": "Pending implementation"
}
```

## Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Mapping**: Leaflet.js with OpenStreetMap tiles
- **Backend**: FastAPI with static file serving
- **Styling**: Custom CSS with modern gradients and animations

## Usage

1. Navigate to the application root URL
2. View the interactive map of British Columbia
3. Click on the map to select your Area of Interest
4. Adjust the date range if needed
5. Click "Run Analysis" to execute the pipeline
6. View results in the results panel

## Future Enhancements

- Polygon AOI selection (not just point)
- Real-time progress updates during processing
- Export results to CSV/PDF
- Historical analysis comparison
- Integration with actual satellite data
