"""Lightweight logging configuration for Kelpie Carbon v1."""

import logging
import logging.config
import logging.handlers
from typing import Any


def setup_logging(level: int = logging.INFO) -> None:
    """Set up basic logging configuration.

    Args:
        level: Logging level (default: INFO)

    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
    )


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
