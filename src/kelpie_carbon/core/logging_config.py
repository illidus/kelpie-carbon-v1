"""Logging configuration for Kelpie Carbon v1."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

from .config import get_settings


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        if hasattr(record, "levelname"):
            color = self.COLORS.get(record.levelname, "")
            record.levelname = f"{color}{record.levelname}{self.RESET}"

        return super().format(record)


def setup_logging() -> None:
    """Setup logging configuration based on settings."""
    settings = get_settings()

    # Handle both old and new config formats
    if hasattr(settings, "logging"):
        # Old config format
        log_config = settings.logging
        log_level = log_config.level.upper()
        log_file = log_config.log_file
        console_output = log_config.console_output
        file_output = log_config.file_output
        log_format = log_config.format
        include_module = log_config.include_module
        include_timestamp = log_config.include_timestamp
    else:
        # New simplified config format
        log_level = settings.log_level.upper()
        log_file = f"{settings.logs_path}/kelpie_carbon.log"
        console_output = True
        file_output = True
        log_format = settings.log_format
        include_module = True
        include_timestamp = True

    # Create logs directory if it doesn't exist
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))

        if log_format == "detailed":
            if include_module:
                console_format = (
                    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
                    if include_timestamp
                    else "%(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
                )
            else:
                console_format = (
                    "%(asctime)s | %(levelname)-8s | %(message)s"
                    if include_timestamp
                    else "%(levelname)-8s | %(message)s"
                )
        elif log_format == "json":
            # For JSON format, we'll use a simple format for console
            console_format = "%(levelname)s: %(message)s"
        else:
            console_format = "%(message)s"

        console_formatter = ColoredFormatter(console_format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if file_output:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setLevel(getattr(logging, log_level))

        file_formatter: JsonFormatter | logging.Formatter
        if log_format == "json":
            file_formatter = JsonFormatter()
        else:
            if include_module:
                file_format = (
                    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
                    if include_timestamp
                    else "%(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
                )
            else:
                file_format = (
                    "%(asctime)s | %(levelname)-8s | %(message)s"
                    if include_timestamp
                    else "%(levelname)-8s | %(message)s"
                )
            file_formatter = logging.Formatter(file_format)

        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Set third-party library log levels
    _configure_third_party_loggers()


def _configure_third_party_loggers() -> None:
    """Configure logging levels for third-party libraries."""
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("pystac_client").setLevel(logging.WARNING)
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    logging.getLogger("fiona").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        import json

        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "filename"):
            log_data["file"] = record.filename

        if hasattr(record, "lineno"):
            log_data["line"] = record.lineno

        if hasattr(record, "funcName"):
            log_data["function"] = record.funcName

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
            }:
                log_data[key] = value

        return json.dumps(log_data)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name, typically __name__

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_performance(func_name: str, duration: float, **kwargs) -> None:
    """Log performance metrics.

    Args:
        func_name: Name of the function being measured
        duration: Duration in seconds
        **kwargs: Additional context data
    """
    logger = get_logger("performance")
    extra_data = {"duration": duration, "function": func_name, **kwargs}

    if duration > 5.0:
        logger.warning(
            f"Slow operation: {func_name} took {duration:.2f}s", extra=extra_data
        )
    else:
        logger.info(f"Performance: {func_name} took {duration:.2f}s", extra=extra_data)


def log_api_request(
    method: str, path: str, status_code: int, duration: float, **kwargs
) -> None:
    """Log API request metrics.

    Args:
        method: HTTP method
        path: Request path
        status_code: Response status code
        duration: Request duration in seconds
        **kwargs: Additional context data
    """
    logger = get_logger("api")
    extra_data = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration": duration,
        **kwargs,
    }

    if status_code >= 500:
        logger.error(
            f"{method} {path} -> {status_code} ({duration:.3f}s)", extra=extra_data
        )
    elif status_code >= 400:
        logger.warning(
            f"{method} {path} -> {status_code} ({duration:.3f}s)", extra=extra_data
        )
    else:
        logger.info(
            f"{method} {path} -> {status_code} ({duration:.3f}s)", extra=extra_data
        )


def log_satellite_data_fetch(
    location: dict[str, float],
    date_range: dict[str, str],
    success: bool,
    duration: float,
    **kwargs,
) -> None:
    """Log satellite data fetch operations.

    Args:
        location: Location dictionary with lat/lng
        date_range: Date range dictionary
        success: Whether the fetch was successful
        duration: Operation duration in seconds
        **kwargs: Additional context data
    """
    logger = get_logger("satellite")
    extra_data = {
        "location": location,
        "date_range": date_range,
        "success": success,
        "duration": duration,
        **kwargs,
    }

    if success:
        logger.info(
            f"Satellite data fetch successful: {location} ({duration:.2f}s)",
            extra=extra_data,
        )
    else:
        logger.error(
            f"Satellite data fetch failed: {location} ({duration:.2f}s)",
            extra=extra_data,
        )
