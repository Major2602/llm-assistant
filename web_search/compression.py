"""
Extractive compression layer.

Module Responsibilities:

- reduce chunk size;
- keep most relevant sentences;
- preserve citations;
- preserve metadata;
- remove redundant information.
"""


from __future__ import annotations


import logging
import re


from typing import Any


import numpy as np


from web_search.cloudflare_embeddings import (
    get_embedding_model,
)


from web_search.models import (
    RankedChunk,
    CompressedChunk,
)


logger = logging.getLogger(__name__)



# ==========================================================
# Configuration
# ==========================================================


MAX_SENTENCES = 5


MAX_COMPRESSED_CHARS = 1200


MIN_SENTENCE_LENGTH = 40


SIMILARITY_THRESHOLD = 0.85



# ==========================================================
# Text utilities
# ==========================================================


def _split_sentences(
    text: str,
) -> list[str]:
    """
    Split text into sentences.
    """

    if not text:
        return []


    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )


    return [

        sentence.strip()

        for sentence in sentences

        if len(sentence.strip())
        >= MIN_SENTENCE_LENGTH

    ]



def _cosine_similarity(
    a: list[float],
    b: list[float],
) -> float:
    """
    Calculate cosine similarity.
    """

    vec_a = np.asarray(
        a,
        dtype=np.float32,
    )


    vec_b = np.asarray(
        b,
        dtype=np.float32,
    )


    denominator = (

        np.linalg.norm(vec_a)

        *

        np.linalg.norm(vec_b)

    )


    if denominator == 0:

        return 0.0


    return float(

        np.dot(
            vec_a,
            vec_b,
        )

        /

        denominator

    )



# ==========================================================
# Sentence ranking
# ==========================================================


async def _rank_sentences(
    query: str,
    sentences: list[str],
) -> list[tuple[str, float]]:
    """
    Rank sentences by query similarity.
    """

    if not sentences:

        return []


    embedder = get_embedding_model()


    query_vector = await embedder.embed_query(
        query
    )


    sentence_vectors = await embedder.embed_documents(
        sentences
    )


    ranked = []


    for sentence, vector in zip(
        sentences,
        sentence_vectors,
    ):

        score = _cosine_similarity(
            query_vector.values,
            vector.values,
        )


        ranked.append(
            (
                sentence,
                score,
            )
        )


    ranked.sort(
        key=lambda item: item[1],
        reverse=True,
    )


    return ranked



# ==========================================================
# Redundancy removal
# ==========================================================


def _remove_redundant(
    ranked_sentences: list[tuple[str, float]],
) -> list[str]:
    """
    Remove semantically similar sentences.

    Uses score order.
    """

    selected: list[str] = []


    for sentence, _ in ranked_sentences:


        if not selected:

            selected.append(
                sentence
            )

            continue



        duplicate = False


        sentence_words = set(
            sentence.lower().split()
        )


        for existing in selected:

            existing_words = set(
                existing.lower().split()
            )


            overlap = (

                len(
                    sentence_words
                    &
                    existing_words
                )

                /

                max(
                    len(sentence_words),
                    1,
                )

            )


            if overlap >= SIMILARITY_THRESHOLD:

                duplicate = True

                break



        if not duplicate:

            selected.append(
                sentence
            )


        if len(selected) >= MAX_SENTENCES:

            break



    return selected



# ==========================================================
# Compression
# ==========================================================


async def _compress_chunk(
    query: str,
    chunk: RankedChunk,
) -> CompressedChunk:
    """
    Compress single chunk.
    """

    text = chunk.text


    sentences = _split_sentences(
        text
    )


    ranked = await _rank_sentences(
        query,
        sentences,
    )


    selected = _remove_redundant(
        ranked
    )


    compressed_text = " ".join(
        selected
    )


    compressed_text = compressed_text[
        :MAX_COMPRESSED_CHARS
    ]


    ratio = (

        len(compressed_text)

        /

        max(
            len(text),
            1,
        )

    )


    return CompressedChunk(

        **chunk.model_dump(),

        compressed_text=compressed_text,

        compression_ratio=ratio,

    )



# ==========================================================
# Public API
# ==========================================================


async def compress_chunks(
    query: str,
    chunks: list[RankedChunk],
) -> list[CompressedChunk]:
    """
    Compress reranked chunks.

    Input:

        RankedChunk list


    Output:

        CompressedChunk list


    Next stage:

        context_optimizer.py
    """

    if not chunks:

        return []


    logger.info(
        "Compression started chunks=%d",
        len(chunks),
    )


    result: list[CompressedChunk] = []


    for chunk in chunks:

        compressed = await _compress_chunk(
            query,
            chunk,
        )


        result.append(
            compressed
        )



    logger.info(
        "Compression completed chunks=%d",
        len(result),
    )


    return result
