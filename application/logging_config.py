"""Application logging configuration utilities."""

from __future__ import annotations

import logging
import os
from logging.config import dictConfig
from config import LOG_LEVEL

DEFAULT_LOG_LEVEL = "INFO"


def _build_logging_config(log_level: str) -> dict:
    """Return the dictionary configuration used by :func:`configure_logging`."""

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "uvicorn": {
                "format": "%(asctime)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
            "uvicorn": {
                "class": "logging.StreamHandler",
                "formatter": "uvicorn",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["uvicorn"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["uvicorn"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["uvicorn"],
                "level": log_level,
                "propagate": False,
            },
            "application": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
        },
        "root": {"handlers": ["console"], "level": log_level},
    }


def configure_logging() -> None:
    """Configure logging for the FastAPI application.

    The configuration is idempotent and may be called multiple times. The
    desired log level can be customised through the ``APP_LOG_LEVEL``
    environment variable. When running under Uvicorn the configuration keeps
    existing loggers enabled so we can see access logs alongside our
    application logs.
    """

    log_level = LOG_LEVEL.upper() if LOG_LEVEL else DEFAULT_LOG_LEVEL
    dictConfig(_build_logging_config(log_level))


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger configured for the application namespace.

    Importers can safely call this helper without worrying whether the logging
    system has already been configured. The configuration call is guarded so it
    executes only once per process.
    """

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        configure_logging()
    return logging.getLogger(name or "application")


__all__ = ["configure_logging", "get_logger"]
