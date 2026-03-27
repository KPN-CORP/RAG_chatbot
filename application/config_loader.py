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
        return _import_config("config")

    module_name = f"config_{profile.upper()}"
    logger.info(f"Attempting to load config for profile {profile}; module: {module_name}")
    try:
        return _import_config(module_name)
    except ModuleNotFoundError:
        logger.warning(
            "Requested profile config not found; falling back to default",
            extra={"profile": profile, "module": module_name},
        )
        return _import_config("config")


__all__ = ["load_config"]
