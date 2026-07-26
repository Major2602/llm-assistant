"""
Query preprocessing layer.

Pipeline position:

USER QUERY
    |
    v
Query preprocessing
    |
    v
SearchQuery
    |
    v
Qdrant Hybrid Retrieval
    |
    v
Retrieval pipeline


Responsibilities:

- normalize user query;
- remove noise;
- detect language;
- detect search intent;
- prepare SearchQuery model;
- provide normalized query for hybrid retrieval.

This module does NOT:

- call Exa;
- call Qdrant;
- generate embeddings;
- perform ranking;
- modify documents;
- call LLM.
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


MULTISPACE_RE = re.compile(
    r"\s+"
)


PUNCT_RE = re.compile(
    r"[^\w\s\-.:/?]",
    flags=re.UNICODE,
)


CYRILLIC_RE = re.compile(
    r"[А-Яа-яЁё]"
)


LATIN_RE = re.compile(
    r"[A-Za-z]"
)


# ==========================================================
# Intent model
# ==========================================================


class QueryIntent(str, Enum):
    """
    Search intent classification.
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
}


COMPARE_KEYWORDS = {
    "vs",
    "versus",
    "compare",
    "comparison",
    "сравнение",
    "лучше",
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
    "что",
    "кто",
    "когда",
    "где",
    "сколько",
)


# ==========================================================
# Normalization
# ==========================================================


def _normalize_query(
    query: str,
) -> str:
    """
    Normalize user query.

    Operations:

    - trim spaces;
    - remove noise punctuation;
    - collapse whitespace;
    - limit length.
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


    return query[:MAX_QUERY_LENGTH].strip()



# ==========================================================
# Language detection
# ==========================================================


def _detect_language(
    query: str,
) -> str:
    """
    Lightweight language detection.

    Returns:

        ru
        en
    """

    cyrillic_count = len(
        CYRILLIC_RE.findall(
            query
        )
    )


    latin_count = len(
        LATIN_RE.findall(
            query
        )
    )


    if cyrillic_count > latin_count:

        return "ru"


    return "en"



# ==========================================================
# Intent detection
# ==========================================================


def _detect_intent(
    query: str,
) -> str:
    """
    Detect search intent.

    Lightweight heuristic classifier.
    """

    normalized = query.lower()

    words = set(
        normalized.split()
    )


    if words & NEWS_KEYWORDS:

        return QueryIntent.NEWS.value


    if words & COMPARE_KEYWORDS:

        return QueryIntent.COMPARISON.value


    if words & HOW_TO_KEYWORDS:

        return QueryIntent.HOW_TO.value


    if normalized.startswith(
        FACTUAL_PREFIXES
    ):

        return QueryIntent.FACTUAL.value


    return QueryIntent.GENERAL.value



# ==========================================================
# Query expansion
# ==========================================================


def _expand_query(
    query: str,
) -> list[str]:
    """
    Placeholder for future expansion.

    Reserved for:

    - synonyms;
    - acronym expansion;
    - multilingual expansion;
    - LLM rewriting.

    Current pipeline keeps empty list.
    """

    return []



# ==========================================================
# Public API
# ==========================================================


def preprocess_query(
    query: str,
) -> SearchQuery:
    """
    Convert raw user query into SearchQuery model.

    Output is consumed by:

    - qdrant_store.py
    - exa.py
    - retrieval pipeline
    """

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )


    normalized = _normalize_query(
        query
    )


    if not normalized:

        raise ValueError(
            "Query became empty after normalization."
        )


    language = _detect_language(
        normalized
    )


    intent = _detect_intent(
        normalized
    )


    expanded_queries = _expand_query(
        normalized
    )


    result = SearchQuery(

        original=query,

        normalized=normalized,

        language=language,

        intent=intent,

        expanded_queries=expanded_queries,

    )


    logger.info(

        "Query preprocessed. "
        "language=%s intent=%s",

        result.language,

        result.intent,

    )


    return result
