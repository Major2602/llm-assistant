"""
Final context optimization layer.

Responsible for:
- preparing compressed chunks for LLM;
- controlling context size;
- preserving citation metadata;
- ordering information by relevance.

Pipeline position:

Exa
 |
 v
chunker.py
 |
 v
filter.py
 |
 v
embedding_retrieval.py
 |
 v
reranker.py
 |
 v
compression.py
 |
 v
context_optimizer.py
 |
 v
AgentContext


This module does NOT:
- call LLM;
- call Exa;
- generate embeddings;
- rerank;
- access Qdrant.
"""

from __future__ import annotations


import logging

from typing import Any


from web_search.models import (
    AgentContext,
    Source,
)


logger = logging.getLogger(__name__)



# ==========================================================
# Configuration
# ==========================================================


# Approximate final context budget.
#
# Controlled because:
# - Groq context window;
# - latency;
# - token cost.
#

MAX_CONTEXT_CHARS = 12000


MAX_SOURCES = 5



# ==========================================================
# Text utilities
# ==========================================================


def _trim_text(
    text: str,
    limit: int,
) -> str:
    """
    Trim text safely.
    """

    if not text:

        return ""


    if len(text) <= limit:

        return text


    return (
        text[:limit]
        +
        "..."
    )



# ==========================================================
# Chunk processing
# ==========================================================


def _sort_chunks(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Sort compressed chunks by relevance.

    Priority:

    1. rerank score
    2. embedding score
    3. filter score
    """


    return sorted(

        chunks,

        key=lambda chunk:

            (

                chunk.get(
                    "rerank_score",
                    0,
                )

                or

                chunk.get(
                    "similarity_score",
                    0,
                )

                or

                chunk.get(
                    "filter_score",
                    0,
                )

                or 0

            ),

        reverse=True,

    )



def _build_context_text(
    chunks: list[dict[str, Any]],
) -> str:
    """
    Build final LLM context.

    Keeps source separation.
    """


    sections: list[str] = []


    current_size = 0



    for index, chunk in enumerate(
        chunks,
        start=1,
    ):


        text = (

            chunk.get(
                "compressed_text"
            )

            or

            chunk.get(
                "text",
                "",
            )

        )


        if not text:

            continue



        title = (

            chunk.get(
                "title"
            )

            or

            "Untitled source"

        )


        section = f"""
SOURCE [{index}]

Title:
{title}

Content:
{text.strip()}
"""


        section_size = len(
            section
        )



        if (

            current_size
            +
            section_size
            >
            MAX_CONTEXT_CHARS

        ):

            break



        sections.append(
            section
        )


        current_size += section_size



    return "\n\n".join(
        sections
    ).strip()



# ==========================================================
# Sources
# ==========================================================


def _extract_sources(
    chunks: list[dict[str, Any]],
) -> list[Source]:
    """
    Create citation metadata.

    Keeps unique URLs.
    """


    sources: list[Source] = []


    seen_urls: set[str] = set()



    for chunk in chunks:


        url = chunk.get(
            "url",
            "",
        )


        if not url:

            continue



        if url in seen_urls:

            continue



        seen_urls.add(
            url
        )



        sources.append(

            Source(

                title=(

                    chunk.get(
                        "title"
                    )

                    or

                    "Untitled source"

                ),


                url=url,


                provider=chunk.get(
                    "provider"
                ),


                score=(

                    chunk.get(
                        "rerank_score"
                    )

                    or

                    chunk.get(
                        "similarity_score"
                    )

                    or

                    chunk.get(
                        "filter_score"
                    )

                ),


                published_date=chunk.get(
                    "published_date"
                ),


                author=chunk.get(
                    "author"
                ),

            )

        )



        if len(sources) >= MAX_SOURCES:

            break



    return sources



# ==========================================================
# Public API
# ==========================================================


def optimize_context(
    chunks: list[dict[str, Any]],
) -> AgentContext:
    """
    Prepare final context for agent.

    Input:

        Compressed reranked chunks


    Output:

        AgentContext


    Example:

        [
            {
                "title": "...",
                "url": "...",
                "compressed_text": "...",
                "rerank_score": 0.91
            }
        ]

    Returns:

        AgentContext(
            text="...",
            sources=[...]
        )
    """


    if not chunks:

        logger.warning(
            "No chunks provided for context optimization."
        )


        return AgentContext()



    logger.info(
        "Optimizing context from %d chunks.",
        len(chunks),
    )



    ranked_chunks = _sort_chunks(
        chunks
    )



    context_text = _build_context_text(
        ranked_chunks
    )



    sources = _extract_sources(
        ranked_chunks
    )



    logger.info(

        "Context optimized. chars=%d sources=%d",

        len(context_text),

        len(sources),

    )



    return AgentContext(

        text=context_text,

        sources=sources,

    )
