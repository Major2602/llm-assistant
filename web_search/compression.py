"""
Extractive compression layer.

Pipeline:

Ranked chunks
      |
      v
Compression
      |
      v
Compressed chunks
      |
      v
Context optimization

Responsibilities:

- reduce chunk size;
- keep query-relevant sentences;
- remove duplicates;
- preserve metadata.

Does NOT:

- call LLM;
- generate embeddings;
- rerank;
- access storage.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from web_search.models import CompressedChunk


logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================

MAX_SENTENCES_PER_CHUNK = 5

MIN_SENTENCE_LENGTH = 40

MAX_COMPRESSED_LENGTH = 1200


# ==========================================================
# Text utilities
# ==========================================================


def _split_sentences(text: str) -> list[str]:
    """
    Split text into meaningful sentences.
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
        if len(sentence.strip()) >= MIN_SENTENCE_LENGTH
    ]



def _normalize(text: str) -> str:
    """
    Normalize text for lexical comparison.
    """

    return re.sub(
        r"[^a-zA-Zа-яА-Я0-9 ]+",
        " ",
        text.lower(),
    )



def _keywords(text: str) -> set[str]:
    """
    Extract keywords.
    """

    return {
        word
        for word in _normalize(text).split()
        if len(word) > 3
    }



def _sentence_score(
    query: str,
    sentence: str,
) -> float:
    """
    Lightweight relevance scoring.

    Future replacement:
    sentence embeddings + cosine similarity.
    """

    query_words = _keywords(query)

    if not query_words:
        return 0.0

    sentence_words = _keywords(sentence)

    return len(
        query_words & sentence_words
    ) / len(query_words)



def _remove_duplicates(
    sentences: list[str],
) -> list[str]:
    """
    Remove duplicate sentences.
    """

    result = []

    seen = set()

    for sentence in sentences:

        fingerprint = sentence[:150].lower()

        if fingerprint in seen:
            continue

        seen.add(fingerprint)
        result.append(sentence)

    return result



# ==========================================================
# Compression
# ==========================================================


def compress_chunk(
    query: str,
    chunk: dict[str, Any],
) -> dict[str, Any]:
    """
    Compress single ranked chunk.
    """

    text = chunk.get(
        "text",
        "",
    )

    if not text:
        return chunk


    sentences = _split_sentences(
        text
    )

    if not sentences:
        return chunk


    ranked_sentences = sorted(
        (
            (
                _sentence_score(
                    query,
                    sentence,
                ),
                sentence,
            )
            for sentence in sentences
        ),
        key=lambda item: item[0],
        reverse=True,
    )


    selected = [
        sentence
        for score, sentence in ranked_sentences[
            :MAX_SENTENCES_PER_CHUNK
        ]
        if score > 0
    ]


    if not selected:
        selected = sentences[
            :MAX_SENTENCES_PER_CHUNK
        ]


    selected = _remove_duplicates(
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
        "text": text,
        "compressed_text": compressed_text,
        "compression_ratio": round(
            len(compressed_text)
            /
            max(len(text), 1),
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
    Compress ranked chunks.
    """

    if not chunks:
        return []


    logger.info(
        "Compressing chunks=%d",
        len(chunks),
    )


    result = [
        compress_chunk(
            query=query,
            chunk=chunk,
        )
        for chunk in chunks
    ]


    logger.info(
        "Compression completed chunks=%d",
        len(result),
    )


    return result



def to_compressed_models(
    chunks: list[dict[str, Any]],
) -> list[CompressedChunk]:
    """
    Convert dictionaries into domain models.
    """

    return [
        CompressedChunk(
            **chunk,
        )
        for chunk in chunks
    ]



def estimate_tokens(
    text: str,
) -> int:
    """
    Approximate token count.
    """

    if not text:
        return 0

    return max(
        1,
        len(text) // 4,
    )
