"""Structured logging setup for SFU Library MCP server.

All logging goes to stderr to avoid polluting MCP stdio JSON-RPC on stdout.
Optionally also logs to a persistent file via RotatingFileHandler.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler


def setup_logging(
    level: str = "INFO",
    name: str = "sfu_library_mcp",
    log_file: str = "",
) -> logging.Logger:
    """Configure and return a logger that writes to stderr and optionally a file.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        name: Logger name.
        log_file: Path to a persistent log file. If non-empty, a
            RotatingFileHandler is added (5 MB max, 3 backups, always DEBUG).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(min(log_level, logging.DEBUG) if log_file else log_level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stderr handler — respects configured level
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(log_level)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    # Persistent file handler — always DEBUG for full diagnostics
    if log_file:
        try:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=5 * 1024 * 1024,  # 5 MB
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as e:
            logger.warning("Could not open log file %s: %s", log_file, e)

    return logger
