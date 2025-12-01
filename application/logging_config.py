"""Application logging configuration utilities."""

from __future__ import annotations

import logging
import os
from logging.config import dictConfig

DEFAULT_LOG_LEVEL = "INFO"

class SafeExtraFormatter(logging.Formatter):
    def __init__(self, *args, default_extras=None, **kwargs):
        self.default_extras = default_extras or {}
        super().__init__(*args, **kwargs)

    def format(self, record):
        standard_attrs = logging.LogRecord(
            name=None, level=None, pathname="", lineno=0, msg="", args=(), exc_info=None
        ).__dict__.keys()

        # Ensure default extras exist on the record so format strings don't fail
        for key, default in self.default_extras.items():
            if not hasattr(record, key):
                setattr(record, key, default)

        extras: dict[str, object] = {}

        # Start with configured defaults (shown only when overridden)
        for key, default in self.default_extras.items():
            value = getattr(record, key, default)
            if value not in (None, "", default):
                extras[key] = value

        # Include any other "extra" attributes provided on the LogRecord
        for key, value in record.__dict__.items():
            if key in standard_attrs or key == "extra_fields" or key in self.default_extras:
                continue
            if value not in (None, "", "-"):
                extras[key] = value

        record.extra_fields = (" - " + " - ".join(f"{k}={v}" for k, v in extras.items())) if extras else ""

        return super().format(record)
    

def _build_logging_config(log_level: str) -> dict:
    mod = __name__
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "()": f"{mod}.SafeExtraFormatter",
                "format": "%(asctime)s - %(levelname)s - %(name)s%(extra_fields)s - %(message)s",
                "default_extras": {
                    "session_id": "-",
                    "retrieved_docs": "-",
                    "original_question": "-",
                    "standalone_question": "-",
                },
            },
            "uvicorn": {"format": "%(asctime)s - %(levelname)s - %(message)s"},
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "formatter": "standard"},
            "uvicorn": {"class": "logging.StreamHandler", "formatter": "uvicorn"},
        },
        "loggers": {
            "uvicorn":        {"handlers": ["uvicorn"], "level": log_level, "propagate": False},
            "uvicorn.error":  {"handlers": ["uvicorn"], "level": log_level, "propagate": False},
            "uvicorn.access": {"handlers": ["uvicorn"], "level": log_level, "propagate": False},
            "application":    {"handlers": ["console"], "level": log_level, "propagate": False},

            # 🔇 quiet down pdfminer (and all its children)
            "pdfminer": {"level": "WARNING", "propagate": False},
        },
        "root": {"handlers": ["console"], "level": log_level},
    }





def configure_logging(log_level: str | None = None) -> None:
    """Configure logging for the FastAPI application.

    The configuration is idempotent and may be called multiple times. The
    desired log level can be customised through the ``APP_LOG_LEVEL``
    environment variable. When running under Uvicorn the configuration keeps
    existing loggers enabled so we can see access logs alongside our
    application logs.
    """
    log_level_env = os.getenv("APP_LOG_LEVEL")
    effective_log_level = (log_level or log_level_env or DEFAULT_LOG_LEVEL).upper()
    dictConfig(_build_logging_config(effective_log_level))


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
