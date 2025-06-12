"""Simplified configuration system for Kelpie Carbon v1."""

import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass
class SimpleConfig:
    """Simplified configuration class with sensible defaults."""

    # Application settings
    app_name: str = "Kelpie Carbon v1"
    app_version: str = "0.1.0"
    description: str = (
        "Kelp Forest Carbon Sequestration Assessment using Sentinel-2 satellite imagery"
    )
    environment: str = field(
        default_factory=lambda: os.getenv("KELPIE_ENV", "development")
    )
    debug: bool = field(
        default_factory=lambda: os.getenv("KELPIE_ENV", "development") == "development"
    )

    # Server settings
    host: str = field(default_factory=lambda: os.getenv("KELPIE_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("KELPIE_PORT", "8000")))
    reload: bool = field(
        default_factory=lambda: os.getenv("KELPIE_ENV", "development") == "development"
    )
    workers: int = field(default_factory=lambda: int(os.getenv("KELPIE_WORKERS", "1")))

    # CORS settings
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    cors_methods: list[str] = field(default_factory=lambda: ["GET", "POST", "OPTIONS"])
    cors_headers: list[str] = field(default_factory=lambda: ["*"])

    # Logging settings
    log_level: str = field(
        default_factory=lambda: os.getenv("KELPIE_LOG_LEVEL", "INFO")
    )
    log_format: str = "detailed"

    # Processing settings
    max_cloud_cover: float = 0.3
    analysis_timeout: int = 300
    image_cache_size: int = 100
    image_cache_ttl: int = 3600

    # Paths
    static_files_path: str = "src/kelpie_carbon/web/static"
    logs_path: str = "logs"
    cache_path: str = "cache"

    # ML settings
    kelp_confidence_threshold: float = 0.7
    biomass_carbon_factor: float = 0.35

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 1 <= self.port <= 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")

        if not 0.0 <= self.max_cloud_cover <= 1.0:
            raise ValueError(
                f"Max cloud cover must be between 0.0 and 1.0, got {self.max_cloud_cover}"
            )

        if self.analysis_timeout <= 0:
            raise ValueError(
                f"Analysis timeout must be positive, got {self.analysis_timeout}"
            )


@lru_cache(maxsize=1)
def get_simple_config() -> SimpleConfig:
    """Get the application configuration with caching."""
    return SimpleConfig()


def get_config_for_environment(env: str) -> SimpleConfig:
    """Get configuration for a specific environment."""
    old_env = os.environ.get("KELPIE_ENV")
    try:
        os.environ["KELPIE_ENV"] = env
        # Clear the cache to force recalculation
        get_simple_config.cache_clear()
        return get_simple_config()
    finally:
        # Restore original environment
        if old_env is not None:
            os.environ["KELPIE_ENV"] = old_env
        else:
            os.environ.pop("KELPIE_ENV", None)
        get_simple_config.cache_clear()


# Backward compatibility
def get_settings() -> SimpleConfig:
    """Backward compatibility function."""
    return get_simple_config()
