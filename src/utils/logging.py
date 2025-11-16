"""
Logging configuration for F1 prediction project.
"""
import logging
import sys
from typing import Optional

from src.utils.config import LOG_LEVEL, LOG_FORMAT


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)

    log_level = level if level else LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper()))

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))

        formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger


def configure_fastf1_logging(level: str = "INFO"):
    try:
        import fastf1
        fastf1.set_log_level(level)
    except ImportError:
        pass


def set_debug_mode(enable: bool = True):
    import os

    if enable:
        os.environ["FASTF1_DEBUG"] = "1"
        configure_fastf1_logging("DEBUG")
    else:
        if "FASTF1_DEBUG" in os.environ:
            del os.environ["FASTF1_DEBUG"]
        configure_fastf1_logging("INFO")
