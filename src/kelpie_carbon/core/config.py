"""Configuration management for Kelpie Carbon v1."""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


@dataclass
class ServerConfig:
    """Server configuration settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    access_log: bool = True
    log_level: str = "info"


@dataclass
class ProcessingConfig:
    """Data processing configuration."""

    max_concurrent_analyses: int = 5
    analysis_timeout: int = 300
    image_cache_size: int = 100
    image_cache_ttl: int = 3600


@dataclass
class SatelliteConfig:
    """Satellite data configuration."""

    data_source: str = "microsoft_planetary_computer"
    max_cloud_cover: float = 0.3
    preferred_resolution: int = 10
    max_scenes_per_request: int = 5
    timeout: int = 30


@dataclass
class MLModelsConfig:
    """Machine learning models configuration."""

    kelp_detection_model_type: str = "random_forest"
    kelp_detection_confidence_threshold: float = 0.7
    kelp_detection_min_area_threshold: int = 100
    biomass_estimation_model_type: str = "regression"
    biomass_estimation_units: str = "tons_per_hectare"
    biomass_estimation_carbon_conversion_factor: float = 0.45


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "detailed"
    include_timestamp: bool = True
    include_module: bool = True
    console_output: bool = True
    file_output: bool = False
    log_file: str = "logs/kelpie.log"


@dataclass
class CORSConfig:
    """CORS configuration."""

    allow_origins: list = field(default_factory=lambda: ["*"])
    allow_credentials: bool = False
    allow_methods: list = field(default_factory=lambda: ["GET", "POST", "OPTIONS"])
    allow_headers: list = field(default_factory=lambda: ["*"])


@dataclass
class SecurityConfig:
    """Security configuration."""

    trusted_hosts: list = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    rate_limiting_enabled: bool = False
    rate_limiting_requests_per_minute: int = 100


@dataclass
class PathsConfig:
    """File paths configuration."""

    static_files: str = "src/kelpie_carbon/reporting/web/static"
    templates: str = "src/kelpie_carbon/reporting/web/templates"
    logs: str = "logs"
    cache: str = "cache"
    temp: str = "tmp"


@dataclass
class Config:
    """Main configuration class."""

    app_name: str = "Kelpie Carbon v1"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True
    description: str = "Kelp Forest Carbon Sequestration Assessment"

    server: ServerConfig = field(default_factory=ServerConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    satellite: SatelliteConfig = field(default_factory=SatelliteConfig)
    ml_models: MLModelsConfig = field(default_factory=MLModelsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cors: CORSConfig = field(default_factory=CORSConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


class ConfigLoader:
    """Configuration loader with environment-based YAML loading."""

    def __init__(self, config_dir: Path | None = None):
        """Initialize the configuration loader.

        Args:
            config_dir: Directory containing configuration files.
                       Defaults to project root/config
        """
        if config_dir is None:
            # Find project root by looking for pyproject.toml
            current = Path(__file__).parent
            while current != current.parent:
                if (current / "pyproject.toml").exists():
                    config_dir = current / "config"
                    break
                current = current.parent
            else:
                config_dir = Path.cwd() / "config"

        self.config_dir = Path(config_dir)
        self._config_cache: dict[str, dict[str, Any]] = {}

    def load_config_file(self, filename: str) -> dict[str, Any]:
        """Load a YAML configuration file.

        Args:
            filename: Name of the configuration file (without extension)

        Returns:
            Dictionary containing configuration data
        """
        if filename in self._config_cache:
            return self._config_cache[filename]

        config_path = self.config_dir / f"{filename}.yml"
        if not config_path.exists():
            return {}

        try:
            with open(config_path, encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}

            # Expand environment variables
            config_data = self._expand_env_vars(config_data)

            self._config_cache[filename] = config_data
            return config_data

        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"Failed to load config file {config_path}: {e}"
            )
            return {}

    def _expand_env_vars(self, obj: Any) -> Any:
        """Recursively expand environment variables in configuration."""
        if isinstance(obj, dict):
            return {key: self._expand_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        else:
            return obj

    def create_config(self, environment: str | None = None) -> Config:
        """Create a configuration object for the specified environment.

        Args:
            environment: Environment name (development, production, etc.)
                        Defaults to KELPIE_ENV environment variable or 'development'

        Returns:
            Configured Config object
        """
        if environment is None:
            environment = os.getenv("KELPIE_ENV", "development")

        # Load base configuration
        base_config = self.load_config_file("base")

        # Load environment-specific configuration
        env_config = self.load_config_file(environment)

        # Merge configurations (environment overrides base)
        merged_config = self._deep_merge(base_config, env_config)

        # Create Config object
        return self._dict_to_config(merged_config)

    def _deep_merge(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _dict_to_config(self, config_dict: dict[str, Any]) -> Config:
        """Convert configuration dictionary to Config object."""
        config = Config()

        # App-level configuration
        app_config = config_dict.get("app", {})
        config.app_name = app_config.get("name", config.app_name)
        config.app_version = app_config.get("version", config.app_version)
        config.environment = app_config.get("environment", config.environment)
        config.debug = app_config.get("debug", config.debug)
        config.description = app_config.get("description", config.description)

        # Server configuration
        server_config = config_dict.get("server", {})
        config.server = ServerConfig(
            host=server_config.get("host", config.server.host),
            port=server_config.get("port", config.server.port),
            reload=server_config.get("reload", config.server.reload),
            workers=server_config.get("workers", config.server.workers),
            access_log=server_config.get("access_log", config.server.access_log),
            log_level=server_config.get("log_level", config.server.log_level),
        )

        # Processing configuration
        processing_config = config_dict.get("processing", {})
        config.processing = ProcessingConfig(
            max_concurrent_analyses=processing_config.get(
                "max_concurrent_analyses", config.processing.max_concurrent_analyses
            ),
            analysis_timeout=processing_config.get(
                "analysis_timeout", config.processing.analysis_timeout
            ),
            image_cache_size=processing_config.get(
                "image_cache_size", config.processing.image_cache_size
            ),
            image_cache_ttl=processing_config.get(
                "image_cache_ttl", config.processing.image_cache_ttl
            ),
        )

        # Satellite configuration
        satellite_config = config_dict.get("satellite", {})
        config.satellite = SatelliteConfig(
            data_source=satellite_config.get(
                "data_source", config.satellite.data_source
            ),
            max_cloud_cover=satellite_config.get(
                "max_cloud_cover", config.satellite.max_cloud_cover
            ),
            preferred_resolution=satellite_config.get(
                "preferred_resolution", config.satellite.preferred_resolution
            ),
            max_scenes_per_request=satellite_config.get(
                "max_scenes_per_request", config.satellite.max_scenes_per_request
            ),
            timeout=satellite_config.get("timeout", config.satellite.timeout),
        )

        # ML Models configuration
        ml_config = config_dict.get("ml_models", {})
        kelp_detection = ml_config.get("kelp_detection", {})
        biomass_estimation = ml_config.get("biomass_estimation", {})

        config.ml_models = MLModelsConfig(
            kelp_detection_model_type=kelp_detection.get(
                "model_type", config.ml_models.kelp_detection_model_type
            ),
            kelp_detection_confidence_threshold=kelp_detection.get(
                "confidence_threshold",
                config.ml_models.kelp_detection_confidence_threshold,
            ),
            kelp_detection_min_area_threshold=kelp_detection.get(
                "min_area_threshold", config.ml_models.kelp_detection_min_area_threshold
            ),
            biomass_estimation_model_type=biomass_estimation.get(
                "model_type", config.ml_models.biomass_estimation_model_type
            ),
            biomass_estimation_units=biomass_estimation.get(
                "units", config.ml_models.biomass_estimation_units
            ),
            biomass_estimation_carbon_conversion_factor=biomass_estimation.get(
                "carbon_conversion_factor",
                config.ml_models.biomass_estimation_carbon_conversion_factor,
            ),
        )

        # Logging configuration
        logging_config = config_dict.get("logging", {})
        config.logging = LoggingConfig(
            level=logging_config.get("level", config.logging.level),
            format=logging_config.get("format", config.logging.format),
            include_timestamp=logging_config.get(
                "include_timestamp", config.logging.include_timestamp
            ),
            include_module=logging_config.get(
                "include_module", config.logging.include_module
            ),
            console_output=logging_config.get(
                "console_output", config.logging.console_output
            ),
            file_output=logging_config.get("file_output", config.logging.file_output),
            log_file=logging_config.get("log_file", config.logging.log_file),
        )

        # CORS configuration
        cors_config = config_dict.get("cors", {})
        config.cors = CORSConfig(
            allow_origins=cors_config.get("allow_origins", config.cors.allow_origins),
            allow_credentials=cors_config.get(
                "allow_credentials", config.cors.allow_credentials
            ),
            allow_methods=cors_config.get("allow_methods", config.cors.allow_methods),
            allow_headers=cors_config.get("allow_headers", config.cors.allow_headers),
        )

        # Security configuration
        security_config = config_dict.get("security", {})
        rate_limiting = security_config.get("rate_limiting", {})

        config.security = SecurityConfig(
            trusted_hosts=security_config.get(
                "trusted_hosts", config.security.trusted_hosts
            ),
            rate_limiting_enabled=rate_limiting.get(
                "enabled", config.security.rate_limiting_enabled
            ),
            rate_limiting_requests_per_minute=rate_limiting.get(
                "requests_per_minute", config.security.rate_limiting_requests_per_minute
            ),
        )

        # Paths configuration
        paths_config = config_dict.get("paths", {})
        config.paths = PathsConfig(
            static_files=paths_config.get("static_files", config.paths.static_files),
            templates=paths_config.get("templates", config.paths.templates),
            logs=paths_config.get("logs", config.paths.logs),
            cache=paths_config.get("cache", config.paths.cache),
            temp=paths_config.get("temp", config.paths.temp),
        )

        return config


@lru_cache(maxsize=1)
def get_config(environment: str | None = None) -> Config:
    """Get the global configuration instance.

    Args:
        environment: Environment name. Defaults to KELPIE_ENV or 'development'

    Returns:
        Configuration instance
    """
    loader = ConfigLoader()
    return loader.create_config(environment)


# Convenience function for getting settings
def get_settings() -> Config:
    """Get the current configuration settings."""
    return get_config()


def load() -> DictConfig:
    """Load unified YAML configuration using OmegaConf.

    Returns:
        DictConfig: The loaded configuration object

    Raises:
        FileNotFoundError: If config/kelpie.yml is not found
        Exception: If there's an error parsing the YAML file
    """
    config_path = Path("config/kelpie.yml")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Make sure config/kelpie.yml exists in the project root."
        )

    try:
        config = OmegaConf.load(config_path)
        return config
    except Exception as e:
        raise Exception(f"Error loading configuration from {config_path}: {e}")


def load_yaml_config() -> DictConfig:
    """Alias for load() function for backward compatibility."""
    return load()
