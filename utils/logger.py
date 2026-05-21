"""
utils/logger.py
===============
Structured, colorized logging for the entire platform.
Uses loguru for minimal boilerplate.
"""

import sys
from loguru import logger as _loguru_logger
from config.settings import get_settings


def setup_logger(level: str = None) -> None:
    """Configure loguru for the platform."""
    cfg = get_settings()
    lvl = level or cfg.log_level

    _loguru_logger.remove()  # Remove default handler

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )

    # Console
    _loguru_logger.add(sys.stderr, level=lvl, format=fmt, colorize=True)

    # File (rotated daily, 7-day retention)
    log_file = cfg.paths["root"] / "logs" / "platform_{time:YYYY-MM-DD}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    _loguru_logger.add(
        str(log_file),
        level="DEBUG",
        format=fmt,
        rotation="1 day",
        retention="7 days",
        compression="zip",
    )


# Auto-configure on import
setup_logger()

# Re-export the configured logger
log = _loguru_logger
