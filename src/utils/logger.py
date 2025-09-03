import logging
import sys


def get_logger(name: str, log_file_path: str) -> logging.Logger:
    """ Returns a new logger with the specified name and log file path """
    logger = logging.getLogger(name)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Set logging level
    logger.setLevel(logging.DEBUG)

    # Create formatter with only level and message
    formatter = logging.Formatter("%(message)s")

    # Create file handler (will overwrite existing log file)
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger
