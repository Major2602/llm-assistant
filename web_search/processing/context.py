# web_search/processing/context.py

from __future__ import annotations


from web_search.domain.models import (
    AgentContext,
    RankedChunk,
)



def optimize_context(
    query: str,
    chunks: list[RankedChunk],
) -> AgentContext:
    """
    Build final agent context from ranked chunks.

    Responsibilities:
    - order preservation;
    - context assembly;
    - metadata preparation.

    Does not perform:
    - retrieval;
    - ranking;
    - compression.
    """

    if not chunks:

        return AgentContext(
            query=query,
            chunks=[],
            text="",
        )


    context_text = "\n\n".join(

        chunk.text

        for chunk in chunks

    )


    return AgentContext(
        query=query,
        chunks=chunks,
        text=context_text,
    )
