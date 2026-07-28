# web_search/processing/filtering.py

from __future__ import annotations


from collections import defaultdict


from web_search.domain.models import (
    DocumentChunk,
)



DEFAULT_MIN_LENGTH = 200

DEFAULT_MIN_WORDS = 40



def _normalize_text(
    text: str,
) -> str:
    """
    Normalize text for comparison.
    """

    return " ".join(
        text.lower()
        .split()
    )



def _is_valid_length(
    text: str,
    min_length: int,
    min_words: int,
) -> bool:
    """
    Basic content quality validation.
    """

    if len(text) < min_length:
        return False


    if len(text.split()) < min_words:
        return False


    return True



def _remove_duplicates(
    chunks: list[DocumentChunk],
) -> list[DocumentChunk]:
    """
    Remove duplicated chunks.
    """

    unique: list[DocumentChunk] = []

    seen: set[str] = set()


    for chunk in chunks:

        fingerprint = _normalize_text(
            chunk.text
        )


        if fingerprint in seen:

            continue


        seen.add(
            fingerprint
        )

        unique.append(
            chunk
        )


    return unique



def filter_chunks(
    chunks: list[DocumentChunk],
    min_length: int = DEFAULT_MIN_LENGTH,
    min_words: int = DEFAULT_MIN_WORDS,
) -> list[DocumentChunk]:
    """
    Filter low-quality document chunks.

    Rules:
    - remove empty content;
    - remove short chunks;
    - remove duplicates.
    """

    if not chunks:

        return []


    filtered = [

        chunk

        for chunk in chunks

        if _is_valid_length(
            chunk.text,
            min_length,
            min_words,
        )

    ]


    return _remove_duplicates(
        filtered
    )
