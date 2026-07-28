"""
Web search logging configuration.
"""

from __future__ import annotations

import logging
import os


LOGGER_NAME = "web_search"


def configure_logging() -> logging.Logger:
    """
    Configure web_search logger.

    Application can attach own handlers later.
    """

    logger = logging.getLogger(
        LOGGER_NAME
    )

    level = os.getenv(
        "WEB_SEARCH_LOG_LEVEL",
        "INFO",
    )

    logger.setLevel(
        getattr(
            logging,
            level.upper(),
            logging.INFO,
        )
    )

    logger.propagate = True

    return logger
