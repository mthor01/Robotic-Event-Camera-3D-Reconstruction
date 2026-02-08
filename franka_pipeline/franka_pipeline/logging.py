"""Logging configuration for franka_pipeline.

This module provides a centralized logging setup that follows the same pattern
as robosuite and deoxys, using Python's standard logging module with colored
console output.

Usage:
    # In main.py (call once at startup):
    from franka_pipeline.logging import setup_logging
    setup_logging(level="INFO")

    # In any module:
    from franka_pipeline.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Camera started")
    logger.warning("Connection lost")
"""

import logging

from termcolor import colored

# Package-level logger name
LOGGER_NAME = "franka_pipeline"


class FrankaPipelineFormatter(logging.Formatter):
    """Colored console formatter for franka_pipeline.

    Matches the style used by robosuite and deoxys for consistency.
    Format: [franka_pipeline LEVEL] message (filename:lineno)
    """

    FORMAT_STR = "[franka_pipeline %(levelname)s] "
    MESSAGE_STR = "%(message)s"
    MESSAGE_STR_WITH_LOCATION = "%(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: colored(FORMAT_STR, "blue", attrs=["bold"])
        + MESSAGE_STR_WITH_LOCATION,
        logging.INFO: colored(FORMAT_STR, "green", attrs=["bold"]) + MESSAGE_STR,
        logging.WARNING: colored(FORMAT_STR, "yellow", attrs=["bold"])
        + MESSAGE_STR_WITH_LOCATION,
        logging.ERROR: colored(FORMAT_STR, "red", attrs=["bold"])
        + MESSAGE_STR_WITH_LOCATION,
        logging.CRITICAL: colored(FORMAT_STR, "red", attrs=["bold", "reverse"])
        + MESSAGE_STR_WITH_LOCATION,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Apply custom formatting to the log record.

        Args:
            record: The log record to format.

        Returns:
            Formatted log string.
        """
        log_fmt = self.FORMATS.get(record.levelno, self.MESSAGE_STR)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger for the franka_pipeline package.

    Args:
        name: Logger name. If None or starts with 'franka_pipeline',
              uses that name directly. Otherwise prefixes with 'franka_pipeline.'.

    Returns:
        A configured logger instance.

    Example:
        logger = get_logger(__name__)  # e.g., "franka_pipeline.sensors.realsense"
        logger = get_logger()  # Returns the root "franka_pipeline" logger
    """
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def setup_logging(
    level: str = "INFO",
    deps_level: str = "INFO",
) -> None:
    """Configure logging for franka_pipeline and dependencies.

    This should be called once at application startup (typically in main.py).

    Args:
        level: Logging level for franka_pipeline. One of:
               "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        deps_level: Logging level for dependency packages (robosuite, deoxys).

    Example:
        # Basic usage (both at INFO)
        setup_logging()

        # Verbose mode for debugging franka_pipeline only
        setup_logging(level="DEBUG")

        # Debug both franka_pipeline and dependencies
        setup_logging(level="DEBUG", deps_level="DEBUG")
    """
    # Get or create the franka_pipeline logger
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper()))

    # Prevent duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(FrankaPipelineFormatter())
        handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(handler)

    # Prevent log messages from propagating to root logger (avoid duplicates)
    logger.propagate = False

    # Configure dependency loggers
    dep_level = getattr(logging, deps_level.upper())
    # Robosuite uses "robosuite_logs" as its logger name
    logging.getLogger("robosuite_logs").setLevel(dep_level)
    # Deoxys uses "deoxys" as its logger name
    logging.getLogger("deoxys").setLevel(dep_level)
    logging.getLogger("deoxys_examples").setLevel(dep_level)
