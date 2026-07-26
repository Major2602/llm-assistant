"""
Query preprocessing layer.

Pipeline:

User query
    |
    v
Query preprocessing
    |
    +-- normalization
    +-- whitespace cleanup
    +-- language detection
    +-- intent detection
    +-- optional query expansion (future)
    |
    v
Hybrid Retrieval


This module intentionally does NOT know about:

- Exa
- Qdrant
- Embeddings
- Reranker
- Compression
- LLM
"""

from __future__ import annotations

import logging
import re
from enum import Enum

from web_search.models import SearchQuery


logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================


MAX_QUERY_LENGTH = 512


MULTISPACE_RE = re.compile(r"\s+")

PUNCT_RE = re.compile(
    r"[^\w\s\-.:/?]",
    flags=re.UNICODE,
)


CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")


# ==========================================================
# Intent
# ==========================================================


class QueryIntent(str, Enum):
    """
    High-level search intent.
    """

    GENERAL = "general"

    FACTUAL = "factual"

    NEWS = "news"

    COMPARISON = "comparison"

    HOW_TO = "how_to"


NEWS_KEYWORDS = {
    "today",
    "latest",
    "recent",
    "news",
    "2025",
    "2026",
    "сегодня",
    "последние",
    "новости",
    "новое",
}


COMPARE_KEYWORDS = {
    "vs",
    "versus",
    "compare",
    "comparison",
    "лучше",
    "сравнение",
    "или",
}


HOW_TO_KEYWORDS = {
    "how",
    "guide",
    "tutorial",
    "install",
    "setup",
    "как",
    "инструкция",
    "настроить",
    "установить",
}


FACTUAL_PREFIXES = (
    "what",
    "who",
    "when",
    "where",
    "why",
    "сколько",
    "что",
    "кто",
    "когда",
    "где",
)


# ==========================================================
# Helpers
# ==========================================================


def _normalize(
    query: str,
) -> str:
    """
    Normalize query.
    """

    query = query.strip()

    query = MULTISPACE_RE.sub(
        " ",
        query,
    )

    query = PUNCT_RE.sub(
        " ",
        query,
    )

    query = MULTISPACE_RE.sub(
        " ",
        query,
    )

    return query[:MAX_QUERY_LENGTH]


def _detect_language(
    query: str,
) -> str:
    """
    Very lightweight language detection.
    """

    cyr = len(
        CYRILLIC_RE.findall(query)
    )

    lat = len(
        LATIN_RE.findall(query)
    )

    if cyr > lat:
        return "ru"

    return "en"


def _detect_intent(
    query: str,
) -> QueryIntent:
    """
    Infer query intent using lightweight heuristics.
    """

    q = query.lower()

    words = set(q.split())

    if words & NEWS_KEYWORDS:
        return QueryIntent.NEWS

    if words & COMPARE_KEYWORDS:
        return QueryIntent.COMPARISON

    if words & HOW_TO_KEYWORDS:
        return QueryIntent.HOW_TO

    if q.startswith(FACTUAL_PREFIXES):
        return QueryIntent.FACTUAL

    return QueryIntent.GENERAL


def _expand_query(
    query: str,
) -> list[str]:
    """
    Placeholder for future query expansion.

    Future versions may use:
    - synonym expansion
    - acronym expansion
    - LLM rewriting
    - multilingual expansion
    """

    return []


# ==========================================================
# Public API
# ==========================================================


def preprocess_query(
    query: str,
) -> SearchQuery:
    """
    Prepare query for retrieval pipeline.
    """

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    normalized = _normalize(query)

    language = _detect_language(
        normalized,
    )

    intent = _detect_intent(
        normalized,
    )

    expanded = _expand_query(
        normalized,
    )

    logger.info(
        "Query preprocessed "
        "(language=%s intent=%s)",
        language,
        intent.value,
    )

    return SearchQuery(
        original=query,
        normalized=normalized,
        language=language,
        intent=intent.value,
        expanded_queries=expanded,
    )
