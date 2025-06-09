"""Tests for simplified configuration system."""
import os

import pytest

from kelpie_carbon_v1.config.simple import (
    SimpleConfig,
    get_config_for_environment,
    get_simple_config,
)


@pytest.mark.unit
@pytest.mark.core
class TestSimpleConfig:
    """Test simplified configuration."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = SimpleConfig()

        assert config.app_name == "Kelpie Carbon v1"
        assert config.app_version == "0.1.0"
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.max_cloud_cover == 0.3
        assert config.kelp_confidence_threshold == 0.7

    def test_environment_variables(self):
        """Test configuration from environment variables."""
        # Test environment variables
        old_env = os.environ.get("KELPIE_ENV")
        old_port = os.environ.get("KELPIE_PORT")
        old_log_level = os.environ.get("KELPIE_LOG_LEVEL")

        try:
            os.environ["KELPIE_ENV"] = "production"
            os.environ["KELPIE_PORT"] = "9000"
            os.environ["KELPIE_LOG_LEVEL"] = "ERROR"

            config = SimpleConfig()

            assert config.environment == "production"
            assert config.port == 9000
            assert config.log_level == "ERROR"
            assert config.debug is False  # Should be False for production

        finally:
            # Restore environment
            if old_env is not None:
                os.environ["KELPIE_ENV"] = old_env
            else:
                os.environ.pop("KELPIE_ENV", None)

            if old_port is not None:
                os.environ["KELPIE_PORT"] = old_port
            else:
                os.environ.pop("KELPIE_PORT", None)

            if old_log_level is not None:
                os.environ["KELPIE_LOG_LEVEL"] = old_log_level
            else:
                os.environ.pop("KELPIE_LOG_LEVEL", None)

    def test_port_validation(self):
        """Test port validation."""
        # Valid ports
        config = SimpleConfig()
        config.port = 8080
        config.__post_init__()  # Trigger validation

        # Invalid ports
        config.port = 0
        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            config.__post_init__()

        config.port = 70000
        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            config.__post_init__()

    def test_cloud_cover_validation(self):
        """Test cloud cover validation."""
        config = SimpleConfig()

        # Valid cloud cover
        config.max_cloud_cover = 0.5
        config.__post_init__()

        # Invalid cloud cover
        config.max_cloud_cover = -0.1
        with pytest.raises(
            ValueError, match="Max cloud cover must be between 0.0 and 1.0"
        ):
            config.__post_init__()

        config.max_cloud_cover = 1.5
        with pytest.raises(
            ValueError, match="Max cloud cover must be between 0.0 and 1.0"
        ):
            config.__post_init__()

    def test_timeout_validation(self):
        """Test timeout validation."""
        config = SimpleConfig()

        # Valid timeout
        config.analysis_timeout = 600
        config.__post_init__()

        # Invalid timeout
        config.analysis_timeout = -1
        with pytest.raises(ValueError, match="Analysis timeout must be positive"):
            config.__post_init__()

    def test_get_simple_config_caching(self):
        """Test that get_simple_config caches results."""
        config1 = get_simple_config()
        config2 = get_simple_config()

        # Should be the same object due to caching
        assert config1 is config2

    def test_get_config_for_environment(self):
        """Test getting configuration for specific environment."""
        # Save current environment
        old_env = os.environ.get("KELPIE_ENV")

        try:
            # Test development environment
            dev_config = get_config_for_environment("development")
            assert dev_config.environment == "development"
            assert dev_config.debug is True

            # Test production environment
            prod_config = get_config_for_environment("production")
            assert prod_config.environment == "production"
            assert prod_config.debug is False

        finally:
            # Restore original environment
            if old_env is not None:
                os.environ["KELPIE_ENV"] = old_env
            else:
                os.environ.pop("KELPIE_ENV", None)

    def test_cors_settings(self):
        """Test CORS configuration."""
        config = SimpleConfig()

        assert config.cors_origins == ["*"]
        assert config.cors_methods == ["GET", "POST", "OPTIONS"]
        assert config.cors_headers == ["*"]

    def test_path_settings(self):
        """Test path configuration."""
        config = SimpleConfig()

        assert config.static_files_path == "src/kelpie_carbon_v1/web/static"
        assert config.logs_path == "logs"
        assert config.cache_path == "cache"
