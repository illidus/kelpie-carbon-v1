# 🔢 Application Constants
"""
Centralized constants for Kelpie Carbon v1 application.
Extracted from codebase to improve maintainability.
"""

# 🛰️ **Satellite Data Constants**
class SatelliteData:
    """Constants for satellite data processing."""
    
    # Cloud coverage threshold (percentage)
    MAX_CLOUD_COVER = 20
    
    # Buffer around point of interest (kilometers)
    DEFAULT_BUFFER_KM = 1.0
    
    # Rough conversion: 1 degree ≈ 111 km at equator
    KM_PER_DEGREE = 111.0
    
    # Sentinel-2 scaling factor (data range 0-10000)
    SENTINEL_SCALE_FACTOR = 10000.0
    
    # Default image resolution (meters)
    DEFAULT_RESOLUTION = 10
    
    # Scene Classification Layer (SCL) values for cloud detection
    SCL_CLOUD_MEDIUM = 8
    SCL_CLOUD_HIGH = 9
    SCL_THIN_CIRRUS = 10


# 🌊 **Kelp Analysis Constants**
class KelpAnalysis:
    """Constants for kelp forest analysis."""
    
    # Carbon content factor for kelp (35% typical)
    CARBON_CONTENT_FACTOR = 0.35
    
    # Conversion factor from hectare to square meters
    HECTARE_TO_M2 = 10000
    
    # Minimum confidence threshold for predictions
    MIN_CONFIDENCE_THRESHOLD = 0.6
    
    # Water depth threshold for kelp habitat (meters)
    MAX_KELP_DEPTH = 40
    
    # Spectral index thresholds
    NDVI_WATER_THRESHOLD = 0.1
    NDRE_KELP_THRESHOLD = 0.05
    FAI_WATER_THRESHOLD = -0.02


# 🔧 **Processing Constants**
class Processing:
    """Constants for data processing operations."""
    
    # Default chunk size for dask arrays
    DEFAULT_CHUNK_SIZE = (100, 100)
    
    # Maximum processing timeout (seconds)
    MAX_PROCESSING_TIMEOUT = 300
    
    # Default file watching poll interval (milliseconds)
    FILE_WATCH_INTERVAL = 1000
    
    # Cache size limits
    MAX_CACHE_SIZE_MB = 500
    MAX_CACHE_ITEMS = 100
    
    # Binary dilation structure size for mask operations
    DILATION_STRUCTURE_SIZE = 2


# 🌐 **Network Constants**
class Network:
    """Constants for network operations."""
    
    # Default port range for auto-discovery
    DEFAULT_PORT_RANGE_START = 8000
    MAX_PORT_ATTEMPTS = 10
    
    # Request timeout (seconds)
    DEFAULT_TIMEOUT = 30
    
    # Retry attempts for failed requests
    MAX_RETRY_ATTEMPTS = 3
    
    # HSTS max age (seconds) - 1 year
    HSTS_MAX_AGE = 31536000


# 🧪 **Test Constants**
class Testing:
    """Constants for testing."""
    
    # Mock data dimensions
    MOCK_DATA_HEIGHT = 200
    MOCK_DATA_WIDTH = 200
    
    # Test coordinates (California coast)
    TEST_LAT = 36.95
    TEST_LNG = -122.06
    
    # Test date range
    TEST_START_DATE = "2023-06-01"
    TEST_END_DATE = "2023-06-15"
    
    # Performance test thresholds
    MAX_ANALYSIS_TIME = 60  # seconds
    MAX_LAYER_LOAD_TIME = 5  # seconds


# 📊 **Model Constants**
class ModelConstants:
    """Constants for machine learning models."""
    
    # Random Forest parameters
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 20
    RF_MIN_SAMPLES_SPLIT = 5
    RF_MIN_SAMPLES_LEAF = 2
    
    # Feature selection parameters
    MAX_FEATURES = 15
    MIN_FEATURE_IMPORTANCE = 0.01
    
    # Cross-validation parameters
    CV_FOLDS = 5
    CV_RANDOM_STATE = 42


# 🗂️ **File System Constants**
class FileSystem:
    """Constants for file system operations."""
    
    # Supported file extensions
    SUPPORTED_IMAGE_FORMATS = [".tif", ".tiff", ".nc", ".zarr"]
    SUPPORTED_CONFIG_FORMATS = [".yml", ".yaml", ".json"]
    
    # Cache directory names
    CACHE_DIR_NAME = ".kelpie_cache"
    TEMP_DIR_NAME = ".kelpie_temp"
    
    # File size limits (MB)
    MAX_UPLOAD_SIZE_MB = 100
    MAX_LOG_FILE_SIZE_MB = 50 