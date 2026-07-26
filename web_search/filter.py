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
Compression


Responsibilities:

- remove invalid chunks;
- remove duplicates;
- apply lightweight lexical scoring;
- reduce candidate pool before embeddings.


This module does NOT know about:

- embeddings;
- Qdrant;
- reranking;
- compression;
- LLM.
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

    return {

        word

        for word in _normalize_text(
            text
        ).split()

        if len(word) > 2

    }



# ==========================================================
# Scoring
# ==========================================================


def _query_overlap_score(
    query: str,
    chunk: DocumentChunk,
) -> float:
    """
    Lightweight lexical relevance score.
    """

    query_tokens = _tokens(
        query
    )


    if not query_tokens:

        return 0.0



    chunk_text = (

        chunk.title

        +

        " "

        +

        chunk.text

    )


    chunk_tokens = _tokens(
        chunk_text
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
    Estimate chunk information quality.
    """

    text = chunk.text


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
    chunk: DocumentChunk,
) -> float:
    """
    Score metadata completeness.
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
    Combined filtering score.
    """

    return (

        _query_overlap_score(
            query,
            chunk,
        )
        *
        0.5

        +

        _quality_score(
            chunk,
        )
        *
        0.35

        +

        _metadata_score(
            chunk,
        )
        *
        0.15

    )



# ==========================================================
# Deduplication
# ==========================================================


def _fingerprint(
    text: str,
) -> str:
    """
    Create lightweight text fingerprint.
    """

    return (

        text[:500]

        .strip()

        .lower()

    )



def _deduplicate(
    chunks: Iterable[DocumentChunk],
) -> list[DocumentChunk]:
    """
    Remove duplicate chunks.
    """

    result: list[DocumentChunk] = []

    seen: set[str] = set()



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
    Filter semantic chunks.

    Input:

        DocumentChunk list


    Output:

        TOP filtered chunks with filter_score


    Next stage:

        embedding similarity
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

        chunk

        for _, chunk

        in scored[:MAX_OUTPUT_CHUNKS]

    ]



    logger.info(

        "Filtered chunks=%d",

        len(result),

    )



    return result
