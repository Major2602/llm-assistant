"""
Chunk quality filtering layer.

Pipeline position:

Exa
 |
 v
Document normalize
 |
 v
Chunker
 |
 v
filter.py
 |
 v
Embedding similarity
 |
 v
Cloudflare reranker
 |
 v
Qdrant


Responsibilities:

- remove bad chunks;
- remove duplicates;
- apply cheap relevance scoring;
- reduce candidate pool before embeddings.


This module does NOT know about:

- embeddings;
- Qdrant;
- BM25;
- reranking;
- LLM;
- compression.
"""


from __future__ import annotations


import logging
import re

from typing import Any


logger = logging.getLogger(__name__)



# ==========================================================
# Configuration
# ==========================================================


MAX_OUTPUT_CHUNKS = 10


MIN_TEXT_LENGTH = 200


MIN_WORDS = 40


MIN_SCORE = 0.30



# ==========================================================
# Text processing
# ==========================================================


def _normalize_text(
    text: str,
) -> str:
    """
    Normalize text for lexical comparison.
    """

    return re.sub(

        r"[^a-zA-Zа-яА-Я0-9 ]+",

        " ",

        text.lower(),

    )



def _tokens(
    text: str,
) -> set[str]:
    """
    Extract meaningful tokens.
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


def _query_overlap_score(
    query: str,
    chunk: dict[str, Any],
) -> float:
    """
    Cheap lexical relevance.

    Used only as pre-filter signal.
    """

    query_tokens = _tokens(
        query
    )


    if not query_tokens:
        return 0.0



    text = (

        chunk.get(
            "title",
            ""
        )

        +

        " "

        +

        chunk.get(
            "text",
            ""
        )

    )


    chunk_tokens = _tokens(
        text
    )


    overlap = (

        len(
            query_tokens
            &
            chunk_tokens
        )

        /

        len(
            query_tokens
        )

    )


    return min(
        overlap,
        1.0,
    )



def _quality_score(
    chunk: dict[str, Any],
) -> float:
    """
    Estimate information quality.
    """

    text = chunk.get(
        "text",
        "",
    )


    if len(text) < MIN_TEXT_LENGTH:

        return 0.0



    words = text.split()


    if len(words) < MIN_WORDS:

        return 0.2



    score = 1.0



    unique_ratio = (

        len(
            set(
                words
            )
        )

        /

        max(
            len(words),
            1,
        )

    )



    if unique_ratio < 0.35:

        score -= 0.3



    if len(text) > 5000:

        score -= 0.1



    return max(
        score,
        0.0,
    )



def _metadata_score(
    chunk: dict[str, Any],
) -> float:
    """
    Score metadata completeness.
    """

    score = 0.0


    if chunk.get(
        "title"
    ):

        score += 0.4


    if chunk.get(
        "url"
    ):

        score += 0.4


    if chunk.get(
        "document_id"
    ):

        score += 0.2


    return score



def _calculate_score(
    query: str,
    chunk: dict[str, Any],
) -> float:
    """
    Combined pre-filter score.
    """


    relevance = _query_overlap_score(
        query,
        chunk,
    )


    quality = _quality_score(
        chunk
    )


    metadata = _metadata_score(
        chunk
    )


    return (

        relevance * 0.5

        +

        quality * 0.35

        +

        metadata * 0.15

    )



# ==========================================================
# Deduplication
# ==========================================================


def _deduplicate(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Remove duplicate chunks.
    """

    result = []


    seen: set[str] = set()



    for chunk in chunks:


        text = chunk.get(
            "text",
            "",
        )


        fingerprint = (

            text[:500]

            .strip()

            .lower()

        )


        if fingerprint in seen:

            continue



        seen.add(
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
    Filter semantic chunks.

    Input:

        chunks from chunker.py


    Output:

        TOP 10 candidate chunks


    Next stage:

        embedding similarity
    """

    if not chunks:

        return []



    logger.info(

        "Filtering %d chunks.",

        len(chunks),

    )



    chunks = _deduplicate(
        chunks
    )



    scored: list[
        tuple[
            float,
            dict[str, Any]
        ]
    ] = []



    for chunk in chunks:


        score = _calculate_score(

            query,

            chunk,

        )



        if score < MIN_SCORE:

            continue



        chunk["filter_score"] = score



        scored.append(

            (
                score,

                chunk,

            )

        )



    scored.sort(

        key=lambda item:
            item[0],

        reverse=True,

    )



    result = [

        chunk

        for _, chunk

        in scored[:MAX_OUTPUT_CHUNKS]

    ]



    logger.info(

        "Selected %d chunks after filtering.",

        len(result),

    )


    return result
