"""
UI layer logging configuration.

Provides consistent logging setup for Chainlit UI components.

The module does not override application-wide logging configuration.
It only configures UI-specific logger hierarchy.
"""

from __future__ import annotations

import logging


# ==========================================================
# Configuration
# ==========================================================


UI_LOGGER_NAME = "ui"


DEFAULT_LEVEL = logging.INFO


# ==========================================================
# Setup
# ==========================================================


def configure_ui_logging(
    level: int = DEFAULT_LEVEL,
) -> None:
    """
    Configure UI logger hierarchy.

    Root logging configuration is owned by the application entrypoint.
    This function only controls UI namespace.
    """

    logger = logging.getLogger(
        UI_LOGGER_NAME
    )

    logger.setLevel(
        level
    )

    logger.propagate = True



# ==========================================================
# Logger factory
# ==========================================================


def get_logger(
    name: str,
) -> logging.Logger:
    """
    Return logger for UI module.

    Example:

        logger = get_logger(__name__)

    """

    configure_ui_logging()

    return logging.getLogger(
        name
    )
