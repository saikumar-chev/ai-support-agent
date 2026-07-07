"""
Utility functions for the AI Support Agent.

This module contains helper functions that are used across different
parts of the application, such as logging setup. Keeping these
utilities separate helps in maintaining a clean and organized codebase.
"""

import logging
import sys
from config import LOG_FILE_PATH, LOG_LEVEL

def setup_logging():
    """
    Configures the application's logger.

    This function sets up a centralized logger that writes messages to both
    the console and a log file. The log level is configurable via the
    config.py module. This approach ensures that all parts of the
    application use a consistent logging format and destination.

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(LOG_LEVEL)

    # Prevent duplicate handlers if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a formatter for the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a handler to write to the console (stdout)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Create a handler to write to a log file, using the path from config
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger