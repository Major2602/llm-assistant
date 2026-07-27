"""
Chunk quality filtering layer.

Pipeline position:

DocumentChunk
        |
        v
Filter chunks
        |
        v
TOP 10 FilteredChunk
        |
        v
Embedding similarity


Responsibilities:

- remove invalid chunks;
- remove duplicates;
- calculate lightweight quality score;
- reduce candidate pool.


This module does NOT:

- generate embeddings;
- access Qdrant;
- rerank;
- compress;
- call LLM.
"""

from __future__ import annotations


import logging
import re


from typing import Iterable


from web_search.models import (
    DocumentChunk,
    FilteredChunk,
)


logger = logging.getLogger(__name__)



# ==========================================================
# Configuration
# ==========================================================


MAX_OUTPUT_CHUNKS = 10


MIN_TEXT_LENGTH = 200


MIN_WORDS = 40


MIN_SCORE = 0.30



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



def _tokens(
    text: str,
) -> set[str]:
    """
    Extract meaningful tokens.
    """

    return {

        word

        for word in _normalize_text(
            text
        ).split()

        if len(word) > 2

    }



# ==========================================================
# Scores
# ==========================================================


def _keyword_score(
    query: str,
    chunk: DocumentChunk,
) -> float:
    """
    Query keyword overlap score.
    """

    query_tokens = _tokens(
        query
    )


    if not query_tokens:

        return 0.0


    content = (

        chunk.title

        + " "

        + chunk.text

    )


    chunk_tokens = _tokens(
        content
    )


    return min(

        len(
            query_tokens
            &
            chunk_tokens
        )

        /

        len(query_tokens),

        1.0,

    )



def _quality_score(
    chunk: DocumentChunk,
) -> float:
    """
    Estimate information quality.
    """

    text = chunk.text.strip()


    if len(text) < MIN_TEXT_LENGTH:

        return 0.0



    words = text.split()


    if len(words) < MIN_WORDS:

        return 0.2



    score = 1.0



    unique_ratio = (

        len(
            set(words)
        )

        /

        len(words)

    )



    if unique_ratio < 0.35:

        score -= 0.3



    return max(
        score,
        0.0,
    )



def _length_score(
    chunk: DocumentChunk,
) -> float:
    """
    Prefer medium sized chunks.
    """

    length = len(
        chunk.text
    )


    if 500 <= length <= 3000:

        return 1.0


    if length < 300:

        return 0.3


    if length > 5000:

        return 0.5


    return 0.8



def _metadata_score(
    chunk: DocumentChunk,
) -> float:
    """
    Metadata completeness.
    """

    score = 0.0


    if chunk.title:

        score += 0.4


    if chunk.url:

        score += 0.4


    if chunk.id:

        score += 0.2


    return score



def _calculate_score(
    query: str,
    chunk: DocumentChunk,
) -> float:
    """
    Combined lightweight filtering score.
    """

    return (

        _keyword_score(
            query,
            chunk,
        )

        * 0.45


        +

        _quality_score(
            chunk,
        )

        * 0.30


        +

        _length_score(
            chunk,
        )

        * 0.15


        +

        _metadata_score(
            chunk,
        )

        * 0.10

    )



# ==========================================================
# Deduplication
# ==========================================================


def _fingerprint(
    text: str,
) -> str:
    """
    Create duplicate fingerprint.
    """

    return (

        _normalize_text(
            text[:500]
        )

        .strip()

    )



def _deduplicate(
    chunks: Iterable[DocumentChunk],
) -> list[DocumentChunk]:
    """
    Remove duplicate chunks.
    """

    result = []


    seen = set()


    for chunk in chunks:


        fingerprint = _fingerprint(
            chunk.text
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
    chunks: list[DocumentChunk],
    query: str,
) -> list[FilteredChunk]:
    """
    Filter chunks before embedding retrieval.

    Input:

        DocumentChunk list


    Output:

        TOP FilteredChunk list
    """

    if not chunks:

        return []


    logger.info(
        "Filtering chunks=%d",
        len(chunks),
    )


    unique_chunks = _deduplicate(
        chunks
    )


    scored: list[
        tuple[
            float,
            FilteredChunk
        ]
    ] = []


    for chunk in unique_chunks:


        score = _calculate_score(
            query,
            chunk,
        )


        if score < MIN_SCORE:

            continue



        filtered = FilteredChunk(

            **chunk.model_dump(),

            filter_score=score,

        )


        scored.append(

            (
                score,

                filtered,

            )

        )



    scored.sort(

        key=lambda item: item[0],

        reverse=True,

    )


    result = [

        item

        for _, item

        in scored[:MAX_OUTPUT_CHUNKS]

    ]


    logger.info(
        "Filtered chunks=%d",
        len(result),
    )


    return result
