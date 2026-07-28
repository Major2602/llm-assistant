# web_search/processing/reranking.py

from __future__ import annotations


from web_search.domain.contracts import Reranker
from web_search.domain.models import (
    EmbeddedChunk,
    RankedChunk,
)



async def rerank_chunks(
    query: str,
    chunks: list[EmbeddedChunk],
    reranker: Reranker,
) -> list[RankedChunk]:
    """
    Apply semantic reranking.

    Infrastructure-independent:
    - reranker injected through contract;
    - no provider-specific logic.
    """

    if not chunks:

        return []


    return await reranker.rerank(
        query=query,
        chunks=chunks,
    )
