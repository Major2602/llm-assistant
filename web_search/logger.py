"""
Web search logging configuration.
"""

from __future__ import annotations

import logging
import os


LOGGER_NAME = "web_search"

DEFAULT_LOG_LEVEL = logging.INFO


def _resolve_log_level(
    value: str | None,
) -> int:
    """
    Resolve logging level safely.
    """

    if not value:
        return DEFAULT_LOG_LEVEL

    level = value.upper()

    return logging.getLevelNamesMapping().get(
        level,
        DEFAULT_LOG_LEVEL,
    )


def get_logger(
    name: str = LOGGER_NAME,
) -> logging.Logger:
    """
    Get configured pipeline logger.

    Does not create handlers.
    Application owns logging infrastructure.
    """

    logger = logging.getLogger(name)

    logger.setLevel(
        _resolve_log_level(
            os.getenv(
                "WEB_SEARCH_LOG_LEVEL"
            )
        )
    )

    logger.propagate = True

    return logger
