"""
Extractive compression layer.

Pipeline position:

TOP reranked chunks
        |
        v
compression.py
        |
        v
compressed context
        |
        v
context optimization
        |
        v
LLM generation


Responsibilities:

- reduce context size;
- keep only query-relevant sentences;
- remove redundancy;
- preserve source metadata.


This module does NOT know about:

- Exa;
- Qdrant;
- reranking;
- LLM generation;
- UI.
"""

from __future__ import annotations


import logging
import re

from typing import Any


logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================


# Maximum sentences kept per chunk.

MAX_SENTENCES_PER_CHUNK = 5



# Minimum sentence length.

MIN_SENTENCE_LENGTH = 40



# Maximum compressed characters.

MAX_COMPRESSED_LENGTH = 1200



# Simple token approximation.

CHARS_PER_TOKEN = 4



# ==========================================================
# Text processing
# ==========================================================


def _split_sentences(
    text: str,
) -> list[str]:
    """
    Split text into sentences.

    Lightweight implementation.
    Designed for Render free tier.
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



def _normalize(
    text: str,
) -> str:
    """
    Normalize text before comparison.
    """

    return re.sub(

        r"[^a-zA-Zа-яА-Я0-9 ]+",

        " ",

        text.lower(),

    )



def _keywords(
    text: str,
) -> set[str]:
    """
    Extract meaningful keywords.
    """

    words = _normalize(
        text
    ).split()


    return {

        word

        for word in words

        if len(word) > 3

    }



# ==========================================================
# Sentence scoring
# ==========================================================


def _sentence_similarity(
    query: str,
    sentence: str,
) -> float:
    """
    Lightweight semantic approximation.

    Future replacement:

        sentence embeddings
        +
        cosine similarity


    Current version:

        keyword overlap

    """

    query_words = _keywords(
        query
    )


    sentence_words = _keywords(
        sentence
    )


    if not query_words:

        return 0.0


    overlap = (

        len(
            query_words
            &
            sentence_words
        )

        /

        len(query_words)

    )


    return min(
        overlap,
        1.0,
    )



def _remove_duplicate_sentences(
    sentences: list[str],
) -> list[str]:
    """
    Remove repeated sentences.
    """

    result = []


    fingerprints = set()


    for sentence in sentences:


        fingerprint = (
            sentence[:120]
            .lower()
        )


        if fingerprint in fingerprints:

            continue


        fingerprints.add(
            fingerprint
        )


        result.append(
            sentence
        )


    return result



# ==========================================================
# Chunk compression
# ==========================================================


def compress_chunk(
    query: str,
    chunk: dict[str, Any],
) -> dict[str, Any]:
    """
    Compress single retrieved chunk.

    Input:

        reranked chunk


    Output:

        compressed chunk

    Example:

    {
        text: "...",

        original_text: "...",

        url: "...",

        title: "...",

        compression_ratio: 0.25
    }

    """


    original_text = chunk.get(
        "text",
        "",
    )


    if not original_text:

        return chunk



    sentences = _split_sentences(
        original_text
    )


    if not sentences:

        return chunk



    scored = []


    for sentence in sentences:


        score = _sentence_similarity(

            query,

            sentence,

        )


        scored.append(

            (
                score,

                sentence,

            )

        )



    scored.sort(

        key=lambda item:
            item[0],

        reverse=True,

    )



    selected = [

        sentence

        for score, sentence

        in scored[:MAX_SENTENCES_PER_CHUNK]

        if score > 0

    ]



    if not selected:

        selected = sentences[
            :MAX_SENTENCES_PER_CHUNK
        ]



    selected = _remove_duplicate_sentences(
        selected
    )



    compressed_text = "\n".join(
        selected
    )



    compressed_text = compressed_text[
        :MAX_COMPRESSED_LENGTH
    ]



    result = {

        **chunk,

        "original_text":
            original_text,


        "text":
            compressed_text,


        "compressed":
            True,


        "compression_ratio":

            round(

                len(compressed_text)

                /

                max(
                    len(original_text),
                    1,
                ),

                3,

            ),

    }



    return result



# ==========================================================
# Public API
# ==========================================================


def compress_chunks(
    query: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Compress final reranked chunks.


    Pipeline:

        TOP 3-5 chunks

              |

              v

        Extractive compression

              |

              v

        optimized context

    """


    if not chunks:

        return []



    logger.info(

        "Compressing %d chunks.",

        len(chunks),

    )



    compressed = []


    for chunk in chunks:


        compressed.append(

            compress_chunk(

                query=query,

                chunk=chunk,

            )

        )



    logger.info(

        "Compression completed. "
        "chunks=%d",

        len(compressed),

    )



    return compressed



# ==========================================================
# Context utilities
# ==========================================================


def estimate_tokens(
    text: str,
) -> int:
    """
    Estimate token count.

    Used by context optimizer.
    """

    if not text:

        return 0


    return max(

        1,

        len(text)

        //

        CHARS_PER_TOKEN,

    )



def build_compressed_context(
    chunks: list[dict[str, Any]],
) -> str:
    """
    Convert compressed chunks into final LLM context.
    """


    sections = []


    for index, chunk in enumerate(

        chunks,

        start=1,

    ):


        sections.append(

            f"""
SOURCE [{index}]

Title:
{chunk.get("title","")}

Content:
{chunk.get("text","")}

URL:
{chunk.get("url","")}
"""

        )



    return "\n\n".join(
        sections
    )
