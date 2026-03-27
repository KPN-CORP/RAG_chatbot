"""Profile-aware configuration loading helper."""

from importlib import import_module
from functools import lru_cache
from types import ModuleType

from logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=None)
def _import_config(module_name: str) -> ModuleType:
    """Import and cache a config module by name."""

    return import_module(module_name)


def load_config(profile: str = "") -> ModuleType:
    """Load the config module for a given chatbot profile.

    Profile rules:
    - ``profile == ""`` → default ``config.py`` (legacy behaviour).
    - ``profile == "IRIS"`` → ``config_IRIS.py``.
    - in general ``profile`` → ``config_{profile.upper()}.py``.

    Missing or invalid profiles fall back to the default config and log a
    warning so the caller can decide how to proceed.
    """

    if not profile:
        config = _import_config("config")
        logger.info(
            "Loaded default config module",
            extra={
                "resolved_config_module": getattr(config, "__name__", "config"),
                "resolved_config_file": getattr(config, "__file__", "unknown"),
            },
        )
        return config

    module_name = f"config_{profile.upper()}"
    logger.info(f"Attempting to load config for profile {profile}; module: {module_name}")
    try:
        config = _import_config(module_name)
        logger.info(
            "Loaded profile config module",
            extra={
                "profile": profile,
                "requested_config_module": module_name,
                "resolved_config_module": getattr(config, "__name__", module_name),
                "resolved_config_file": getattr(config, "__file__", "unknown"),
            },
        )
        return config
    except ModuleNotFoundError:
        config = _import_config("config")
        logger.warning(
            "Requested profile config not found; using default config module",
            extra={
                "profile": profile,
                "requested_config_module": module_name,
                "resolved_config_module": getattr(config, "__name__", "config"),
                "resolved_config_file": getattr(config, "__file__", "unknown"),
            },
        )
        return config


__all__ = ["load_config"]
