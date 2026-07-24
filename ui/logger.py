import logging


logger = logging.getLogger(
    "ui"
)


def configure_ui_logging():
    """
    Configure UI layer logging.
    """

    logger.info(
        "UI logging initialized."
    )
