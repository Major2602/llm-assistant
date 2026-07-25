"""
Cheap document filtering layer.

Responsibilities:
- remove low quality Exa documents;
- score documents using cheap heuristics;
- select best documents before chunking.

This module does not know about:
- embeddings;
- reranking models;
- Qdrant;
- LLM.
"""

from __future__ import annotations

import logging
import re
from typing import Any


logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================


MIN_DOCUMENT_LENGTH = 800

MAX_DOCUMENTS = 5


MIN_SCORE = 0.25



# ==========================================================
# Text utilities
# ==========================================================


def _normalize_text(
    text: str,
) -> str:
    """
    Normalize text for comparison.
    """

    return re.sub(
        r"[^a-zA-Zа-яА-Я0-9 ]+",
        " ",
        text.lower(),
    )



def _tokenize(
    text: str,
) -> set[str]:
    """
    Extract meaningful words.
    """

    words = _normalize_text(
        text
    ).split()


    return {
        word
        for word in words
        if len(word) > 2
    }



# ==========================================================
# Scoring
# ==========================================================


def _keyword_score(
    query: str,
    document: dict[str, Any],
) -> float:
    """
    Cheap keyword overlap score.
    """

    query_words = _tokenize(
        query
    )


    if not query_words:
        return 0.0


    title_words = _tokenize(
        document.get(
            "title",
            "",
        )
    )


    text_words = _tokenize(
        document.get(
            "text",
            "",
        )
    )


    title_match = len(
        query_words & title_words
    )


    text_match = len(
        query_words & text_words
    )


    score = 0.0


    # Title has stronger signal.
    score += (
        title_match
        /
        len(query_words)
        *
        0.6
    )


    score += (
        text_match
        /
        len(query_words)
        *
        0.4
    )


    return min(
        score,
        1.0,
    )



def _quality_score(
    document: dict[str, Any],
) -> float:
    """
    Estimate document quality.
    """

    text = document.get(
        "text",
        "",
    )


    length = len(
        text
    )


    if length < MIN_DOCUMENT_LENGTH:
        return 0.0


    score = 1.0



    # Very long documents are not bad,
    # but reduce priority slightly.
    if length > 50000:
        score -= 0.15



    words = _tokenize(
        text
    )


    # Low unique word ratio.
    if len(words) < 100:
        score -= 0.25



    return max(
        score,
        0.0,
    )



def _calculate_score(
    query: str,
    document: dict[str, Any],
) -> float:
    """
    Combined cheap ranking score.
    """

    keyword = _keyword_score(
        query,
        document,
    )


    quality = _quality_score(
        document,
    )


    return (
        keyword * 0.7
        +
        quality * 0.3
    )



# ==========================================================
# Deduplication
# ==========================================================


def _remove_duplicates(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Remove documents with identical URLs.
    """

    result = []

    seen_urls: set[str] = set()


    for document in documents:

        url = document.get(
            "url",
            "",
        )


        if url and url in seen_urls:
            continue


        if url:
            seen_urls.add(
                url
            )


        result.append(
            document
        )


    return result



# ==========================================================
# Public API
# ==========================================================


def filter_documents(
    documents: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """
    Filter and rank Exa documents.

    Input:

        Exa documents

    Output:

        Top documents for chunking
    """

    logger.info(
        "Filtering %d Exa documents.",
        len(documents),
    )


    documents = _remove_duplicates(
        documents
    )


    scored_documents: list[
        tuple[float, dict[str, Any]]
    ] = []


    for document in documents:

        score = _calculate_score(
            query,
            document,
        )


        logger.debug(
            "Document score %.3f title=%s",
            score,
            document.get(
                "title"
            ),
        )


        if score < MIN_SCORE:
            continue


        document["filter_score"] = score


        scored_documents.append(
            (
                score,
                document,
            )
        )



    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True,
    )



    selected = [

        document

        for _, document

        in scored_documents[:MAX_DOCUMENTS]

    ]



    logger.info(
        "Selected %d documents after filtering.",
        len(selected),
    )


    return selected
