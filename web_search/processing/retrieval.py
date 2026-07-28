# web_search/processing/retrieval.py

from __future__ import annotations


from web_search.domain.contracts import Embedder
from web_search.domain.models import (
    DocumentChunk,
    EmbeddedChunk,
    DenseVector,
)



async def retrieve_by_embedding(
    query: str,
    chunks: list[DocumentChunk],
    embedder: Embedder,
) -> list[EmbeddedChunk]:
    """
    Generate embeddings for chunks and attach vectors.

    Infrastructure-independent:
    - embedder injected through contract;
    - no knowledge about Cloudflare/provider.
    """

    if not chunks:

        return []


    texts = [
        chunk.text
        for chunk in chunks
    ]


    vectors = await embedder.embed_documents(
        texts
    )


    if not vectors:

        return []


    result: list[EmbeddedChunk] = []


    for chunk, vector in zip(
        chunks,
        vectors,
    ):

        result.append(

            EmbeddedChunk(

                **chunk.model_dump(),

                embedding=vector,

            )

        )


    return result
