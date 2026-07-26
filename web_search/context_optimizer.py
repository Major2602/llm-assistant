"""
Final context optimization layer.

Pipeline position:

Ranked chunks
        |
        v
Compression
        |
        v
Context optimization
        |
        v
AgentContext
        |
        +----------------+
        |                |
        v                v
    Groq LLM        Citations


Responsibilities:

- prepare LLM context;
- control context size;
- preserve citations;
- create final source mapping.


Does NOT:

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
    ContextDocument,
    OptimizedContext,
    Source,
)


logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================


MAX_CONTEXT_CHARS = 12000

MAX_SOURCES = 5

CHARS_PER_TOKEN = 4



# ==========================================================
# Utilities
# ==========================================================


def _estimate_tokens(
    text: str,
) -> int:
    """
    Approximate token count.
    """

    if not text:
        return 0

    return max(
        1,
        len(text) // CHARS_PER_TOKEN,
    )



def _sort_chunks(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Sort by final relevance.

    Priority:

    1. rerank_score
    2. similarity_score
    3. filter_score
    """

    return sorted(
        chunks,
        key=lambda chunk: (

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
                "embedding_score",
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



def _build_source(
    chunk: dict[str, Any],
) -> Source:
    """
    Create citation source model.
    """

    return Source(

        title=(
            chunk.get(
                "title"
            )
            or
            "Untitled source"
        ),

        url=(
            chunk.get(
                "url"
            )
            or
            ""
        ),

        provider=chunk.get(
            "provider"
        ),

        author=chunk.get(
            "author"
        ),

        published_date=chunk.get(
            "published_date"
        ),

    )



def _extract_sources(
    chunks: list[dict[str, Any]],
) -> list[Source]:
    """
    Extract unique citation sources.
    """

    sources: list[Source] = []

    seen_urls: set[str] = set()


    for chunk in chunks:

        source = _build_source(
            chunk
        )


        if not source.url:
            continue


        if source.url in seen_urls:
            continue


        seen_urls.add(
            source.url
        )


        sources.append(
            source
        )


        if len(sources) >= MAX_SOURCES:
            break


    return sources



def _build_context_documents(
    chunks: list[dict[str, Any]],
) -> list[ContextDocument]:
    """
    Convert chunks into final context documents.
    """

    documents: list[ContextDocument] = []


    for chunk in chunks:

        text = (

            chunk.get(
                "compressed_text"
            )

            or

            chunk.get(
                "text",
                ""
            )

        )


        if not text:
            continue


        documents.append(

            ContextDocument(

                text=text,

                source=_build_source(
                    chunk
                ),

                relevance_score=(

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

                    or 0.0

                ),

            )

        )


    return documents



def _build_llm_text(
    documents: list[ContextDocument],
) -> str:
    """
    Build final text context.
    """

    sections: list[str] = []

    current_length = 0


    for index, document in enumerate(
        documents,
        start=1,
    ):

        section = f"""
SOURCE [{index}]

Title:
{document.source.title}

Content:
{document.text}

URL:
{document.source.url}
"""


        size = len(section)


        if (
            current_length + size
            >
            MAX_CONTEXT_CHARS
        ):
            break


        sections.append(
            section.strip()
        )


        current_length += size


    return "\n\n".join(
        sections
    )



def _build_citation_map(
    sources: list[Source],
) -> dict[str, Source]:
    """
    Create source lookup map.
    """

    return {
        str(index + 1): source
        for index, source in enumerate(
            sources
        )
    }



# ==========================================================
# Public API
# ==========================================================


def optimize_context(
    query: str,
    chunks: list[dict[str, Any]],
) -> AgentContext:
    """
    Prepare final agent context.

    Input:

        compressed ranked chunks


    Output:

        AgentContext
    """

    if not chunks:

        logger.warning(
            "No chunks provided."
        )

        return AgentContext()



    logger.info(
        "Optimizing context chunks=%d",
        len(chunks),
    )



    ranked_chunks = _sort_chunks(
        chunks
    )


    documents = _build_context_documents(
        ranked_chunks
    )


    sources = _extract_sources(
        ranked_chunks
    )


    context_text = _build_llm_text(
        documents
    )


    optimized = OptimizedContext(

        query=query,

        documents=documents,

        total_tokens=_estimate_tokens(
            context_text
        ),

        citation_map=_build_citation_map(
            sources
        ),

    )


    logger.info(
        "Context optimized chars=%d sources=%d tokens=%d",
        len(context_text),
        len(sources),
        optimized.total_tokens,
    )



    return AgentContext(

        text=context_text,

        sources=sources,

        optimized_context=optimized,

    )
