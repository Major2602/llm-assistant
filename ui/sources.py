"""
Source rendering utilities.

Responsibilities:
- render source metadata for the UI;
- produce markdown citations;
- remain independent of retrieval implementation.

This module must not depend on:
- LangChain
- Chainlit
- Exa
- Qdrant
- retrieval pipeline
"""

from __future__ import annotations

import logging
from datetime import datetime

from web_search.models import Source

logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================

MAX_SOURCES_DISPLAY = 5


# ==========================================================
# Helpers
# ==========================================================


def _title(source: Source) -> str:
    """
    Return normalized title.
    """

    title = source.title.strip()

    return title or "Untitled source"


def _url(source: Source) -> str:
    """
    Return normalized URL.
    """

    return source.url.strip()


def _provider(source: Source) -> str:
    """
    Human-readable provider.
    """

    if not source.provider:
        return ""

    return source.provider.upper()


def _score(source: Source) -> str:
    """
    Format relevance score.
    """

    if source.score is None:
        return ""

    return f"{source.score:.2f}"


def _date(source: Source) -> str:
    """
    Format publication date.
    """

    if not source.published_date:
        return ""

    try:

        dt = datetime.fromisoformat(
            source.published_date.replace(
                "Z",
                "+00:00",
            )
        )

        return dt.strftime("%Y-%m-%d")

    except Exception:

        return source.published_date


def _metadata(source: Source) -> str:
    """
    Build metadata line.
    """

    parts: list[str] = []

    provider = _provider(source)

    if provider:
        parts.append(provider)

    if source.author:
        parts.append(source.author)

    date = _date(source)

    if date:
        parts.append(date)

    score = _score(source)

    if score:
        parts.append(f"relevance {score}")

    return " • ".join(parts)


# ==========================================================
# Public API
# ==========================================================


def format_sources(
    sources: list[Source],
) -> str:
    """
    Render markdown source block.

    Example:

    ---
    ### Sources

    1. OpenAI Research
       https://openai.com
       EXA • 2025-01-01 • relevance 0.98
    """

    if not sources:

        logger.debug(
            "No sources to render."
        )

        return ""

    displayed = sources[
        :MAX_SOURCES_DISPLAY
    ]

    logger.info(
        "Rendering %d sources.",
        len(displayed),
    )

    lines: list[str] = [

        "",

        "---",

        "",

        "### Sources",

        "",

    ]

    for index, source in enumerate(
        displayed,
        start=1,
    ):

        title = _title(source)

        url = _url(source)

        metadata = _metadata(source)

        if url:

            lines.append(
                f"{index}. [{title}]({url})"
            )

        else:

            lines.append(
                f"{index}. {title}"
            )

        if metadata:

            lines.append(
                f"   {metadata}"
            )

        lines.append("")

    logger.debug(
        "Source rendering completed."
    )

    return "\n".join(lines).rstrip()
