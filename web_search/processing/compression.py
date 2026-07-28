# web_search/processing/compression.py

from __future__ import annotations


from web_search.domain.models import RankedChunk



def compress_chunks(
    query: str,
    chunks: list[RankedChunk],
    max_chunks: int = 10,
) -> list[RankedChunk]:
    """
    Reduce ranked chunks before context building.

    Current behavior:
    - keeps ranking order;
    - limits context size.

    Future:
    - can be replaced with LLM-based compression.
    """

    if not chunks:
        return []


    return chunks[:max_chunks]
