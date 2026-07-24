"""
Source rendering utilities.

Responsible only for converting retrieved sources
into user-facing UI representation.

This module does not depend on:
- LangChain
- Chainlit
- agent logic
- retrieval logic
"""

from __future__ import annotations

import logging

from web_search.models import Source


logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================


MAX_SOURCES_DISPLAY = 5


# ==========================================================
# Formatting
# ==========================================================


def _format_score(
    score: float | None,
) -> str:
    """
    Format similarity score for display.
    """

    if score is None:
        return ""

    return f" (relevance: {score:.2f})"


def _normalize_title(
    title: str | None,
) -> str:
    """
    Ensure source title is always present.
    """

    if not title:
        return "Untitled source"

    return title.strip()


def _normalize_url(
    url: str | None,
) -> str:
    """
    Ensure URL is safe for markdown rendering.
    """

    if not url:
        return ""

    return url.strip()


# ==========================================================
# Public API
# ==========================================================


def format_sources(
    sources: list[Source],
) -> str:
    """
    Convert sources into markdown citation block.

    Example output:

    ---
    
    ### Sources

    1. OpenAI Research
       https://openai.com

    2. LangChain Docs
       https://python.langchain.com
    """

    if not sources:

        logger.debug(
            "No sources to format."
        )

        return ""


    logger.info(
        "Formatting %d sources.",
        len(sources),
    )


    lines: list[str] = [
        "",
        "",
        "---",
        "",
        "### Sources",
        "",
    ]


    displayed = sources[
        :MAX_SOURCES_DISPLAY
    ]


    for index, source in enumerate(
        displayed,
        start=1,
    ):

        title = _normalize_title(
            source.title
        )

        url = _normalize_url(
            source.url
        )

        score = _format_score(
            source.score
        )


        if url:

            lines.append(
                f"{index}. [{title}]({url}){score}"
            )

        else:

            lines.append(
                f"{index}. {title}{score}"
            )


    logger.debug(
        "Sources markdown generated successfully."
    )


    return "\n".join(lines)
