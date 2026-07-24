import logging

from web_search.models import Source

logger = logging.getLogger(__name__)


def format_sources(
    sources: list[Source],
) -> str:
    """
    Format sources for Chainlit message.
    """

    if not sources:
        return ""

    lines = [
        "",
        "",
        "---",
        "",
        "### Sources",
    ]

    for index, source in enumerate(
        sources,
        start=1,
    ):
        lines.append(
            f"{index}. [{source.title}]({source.url})"
        )

    logger.info(
        "Formatted %d sources.",
        len(sources),
    )

    return "\n".join(lines)
