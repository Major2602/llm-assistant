import logging
from typing import Any


logger = logging.getLogger(__name__)


def format_sources(
    sources: list[dict[str, Any]],
) -> str:
    """
    Format sources for Chainlit UI.
    """

    if not sources:
        return ""

    result = [
        "\n\nSources:"
    ]

    for index, source in enumerate(
        sources,
        start=1,
    ):

        result.append(
            (
                f"{index}. "
                f"{source.get('title', '')}\n"
                f"{source.get('url', '')}"
            )
        )

    logger.info(
        "Formatted %d sources.",
        len(sources),
    )

    return "\n".join(result)
