"""
Logging utility for DSPy HPC project.
Provides colored console output and file logging with proper formatting.
"""

import logging
import logging.handlers
import colorlog
import sys
from pathlib import Path

class ProjectLogger:
    """Centralized logging utility for the project."""

    _instance = None
    _loggers = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProjectLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)-20s - %(levelname)-8s%(reset)s - %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_formatter)

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            "logs/experiment.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf8'
        )
        file_handler.setLevel(logging.DEBUG)

        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)-20s - %(levelname)-8s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        # Store handlers
        self.console_handler = console_handler
        self.file_handler = file_handler

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the given name."""
        if name in self._loggers:
            return self._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        # Add our handlers
        logger.addHandler(self.console_handler)
        logger.addHandler(self.file_handler)

        # Prevent propagation to root logger
        logger.propagate = False

        self._loggers[name] = logger
        return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.

    Args:
        name: Name of the module/component requesting the logger

    Returns:
        Configured logger instance with colored console and file output

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting benchmark")
        >>> logger.debug("Detailed debug info")
    """
    return ProjectLogger().get_logger(name)


# Utility functions for structured logging
def log_section(logger: logging.Logger, title: str, char: str = "=", width: int = 80):
    """Log a section header."""
    logger.info(char * width)
    logger.info(f"{title:^{width}}")
    logger.info(char * width)


def log_subsection(logger: logging.Logger, title: str, char: str = "-", width: int = 80):
    """Log a subsection header."""
    logger.info(f"\n{title}")
    logger.info(char * width)


def log_dict(logger: logging.Logger, data: dict, indent: int = 0):
    """Log dictionary contents in a readable format."""
    prefix = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            logger.info(f"{prefix}{key}:")
            log_dict(logger, value, indent + 1)
        else:
            logger.info(f"{prefix}{key}: {value}")


def log_metrics(logger: logging.Logger, metrics: dict):
    """Log metrics in a formatted table-like structure."""
    logger.info("\nMetrics:")
    logger.info("-" * 60)
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            logger.info(f"  {metric_name:.<40} {metric_value:>10.4f}")
        else:
            logger.info(f"  {metric_name:.<40} {metric_value:>10}")
    logger.info("-" * 60)


if __name__ == "__main__":
    # Test the logger
    logger = get_logger("test")

    log_section(logger, "Testing Logger")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    log_subsection(logger, "Testing Dictionary Logging")
    test_data = {
        "benchmark": "matrix_multiply",
        "size": 512,
        "config": {
            "optimizer": "miprov2",
            "iterations": 10
        }
    }
    log_dict(logger, test_data)

    log_subsection(logger, "Testing Metrics Logging")
    test_metrics = {
        "execution_time_ms": 45.234,
        "speedup_vs_numpy": 0.86,
        "total_tokens": 10190
    }
    log_metrics(logger, test_metrics)

    logger.info("\nLogger test completed!")
