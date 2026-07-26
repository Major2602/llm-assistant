"""
Chunk-level filtering layer.

Responsibilities:
- remove low quality chunks;
- score chunks using cheap heuristics;
- select best chunks before reranking.

Pipeline position:

Exa
 |
 v
chunker.py
 |
 v
filter.py
 |
 v
reranker.py
 |
 v
qdrant_store.py


This module does not know about:
- Exa API;
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


MIN_CHUNK_LENGTH = 200

MAX_CHUNKS = 50

MIN_SCORE = 0.25



# ==========================================================
# Text utilities
# ==========================================================


def _normalize_text(
    text: str,
) -> str:
    """
    Normalize text for keyword comparison.
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
    chunk: dict[str, Any],
) -> float:
    """
    Calculate query/chunk keyword relevance.
    """

    query_words = _tokenize(
        query
    )


    if not query_words:
        return 0.0


    title_words = _tokenize(
        chunk.get(
            "title",
            "",
        )
    )


    text_words = _tokenize(
        chunk.get(
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


    # Title contains strong relevance signal.
    score += (
        title_match
        /
        len(query_words)
        *
        0.5
    )


    score += (
        text_match
        /
        len(query_words)
        *
        0.5
    )


    return min(
        score,
        1.0,
    )



def _quality_score(
    chunk: dict[str, Any],
) -> float:
    """
    Estimate chunk information quality.
    """

    text = chunk.get(
        "text",
        "",
    )


    length = len(
        text
    )


    if length < MIN_CHUNK_LENGTH:
        return 0.0



    score = 1.0



    words = _tokenize(
        text
    )


    # Penalize chunks with little information.
    if len(words) < 30:
        score -= 0.3



    # Penalize excessive repetition.
    unique_ratio = (
        len(words)
        /
        max(
            len(text.split()),
            1,
        )
    )


    if unique_ratio < 0.4:
        score -= 0.2



    return max(
        score,
        0.0,
    )



def _position_score(
    chunk: dict[str, Any],
) -> float:
    """
    Prefer early chunks from a document.

    First chunks usually contain:
    - introduction;
    - definitions;
    - main topic context.
    """

    index = chunk.get(
        "chunk_index",
        0,
    )


    if index == 0:
        return 1.0


    if index <= 2:
        return 0.8


    return 0.5



def _calculate_score(
    query: str,
    chunk: dict[str, Any],
) -> float:
    """
    Combined chunk relevance score.
    """

    keyword = _keyword_score(
        query,
        chunk,
    )


    quality = _quality_score(
        chunk,
    )


    position = _position_score(
        chunk,
    )


    return (
        keyword * 0.6
        +
        quality * 0.3
        +
        position * 0.1
    )



# ==========================================================
# Deduplication
# ==========================================================


def _remove_duplicates(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Remove identical chunks.
    """

    result = []

    seen_texts: set[str] = set()


    for chunk in chunks:

        text = (
            chunk.get(
                "text",
                "",
            )
            .strip()
        )


        fingerprint = text[:300]


        if fingerprint in seen_texts:
            continue


        seen_texts.add(
            fingerprint
        )


        result.append(
            chunk
        )


    return result



# ==========================================================
# Public API
# ==========================================================


def filter_chunks(
    chunks: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """
    Filter and rank chunks before reranking.

    Input:

        chunks from chunker.py


    Output:

        candidate chunks for reranker.py
    """

    logger.info(
        "Filtering %d chunks.",
        len(chunks),
    )


    chunks = _remove_duplicates(
        chunks
    )


    scored_chunks: list[
        tuple[float, dict[str, Any]]
    ] = []


    for chunk in chunks:

        score = _calculate_score(
            query,
            chunk,
        )


        logger.debug(
            "Chunk score %.3f title=%s index=%s",
            score,
            chunk.get(
                "title"
            ),
            chunk.get(
                "chunk_index"
            ),
        )


        if score < MIN_SCORE:
            continue


        chunk["filter_score"] = score


        scored_chunks.append(
            (
                score,
                chunk,
            )
        )



    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True,
    )



    selected = [

        chunk

        for _, chunk

        in scored_chunks[:MAX_CHUNKS]

    ]



    logger.info(
        "Selected %d chunks after filtering.",
        len(selected),
    )


    return selected
