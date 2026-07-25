"""
Exa search integration layer.

Responsibilities:
- execute Exa web search;
- normalize Exa documents;
- return clean documents for downstream processing.

This module does not know about:
- chunking;
- filtering;
- embeddings;
- reranking;
- Qdrant;
- LLM context formatting.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_exa import ExaSearchRetriever


logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================


EXA_TOKEN = os.getenv(
    "EXA_TOKEN"
)


EXA_RESULTS = int(
    os.getenv(
        "EXA_RESULTS",
        "5",
    )
)


# ==========================================================
# Exa client
# ==========================================================


_retriever: ExaSearchRetriever | None = None



def get_exa_retriever() -> ExaSearchRetriever:
    """
    Lazily initialize Exa retriever.
    """

    global _retriever


    if _retriever is not None:
        return _retriever


    if not EXA_TOKEN:

        logger.error(
            "Environment variable EXA_TOKEN is not configured."
        )

        raise RuntimeError(
            "Environment variable EXA_TOKEN is not configured."
        )


    logger.info(
        "Initializing ExaSearchRetriever."
    )


    _retriever = ExaSearchRetriever(
        exa_api_key=EXA_TOKEN,
        k=EXA_RESULTS,
        text_contents=True,
    )


    logger.info(
        "ExaSearchRetriever initialized successfully."
    )


    return _retriever



# ==========================================================
# Search
# ==========================================================


async def _search(
    query: str,
) -> list[Any]:
    """
    Execute Exa search.
    """

    logger.info(
        "Searching Exa for query='%s'.",
        query,
    )


    retriever = get_exa_retriever()


    documents = await retriever.ainvoke(
        query
    )


    logger.info(
        "Exa returned %d documents.",
        len(documents),
    )


    return documents



# ==========================================================
# Normalization
# ==========================================================


def _normalize_document(
    document: Any,
    query: str,
) -> dict[str, Any] | None:
    """
    Convert Exa document into internal format.

    Keeps full document text.
    Chunking is intentionally handled elsewhere.
    """


    metadata = (

        document.metadata

        if hasattr(
            document,
            "metadata",
        )

        else {}
    )


    text = (

        getattr(
            document,
            "page_content",
            "",
        )

        or ""

    ).strip()



    if not text:

        logger.debug(
            "Skipping empty Exa document."
        )

        return None



    return {

        "query": query,


        "title": (
            metadata.get("title")
            or "Untitled"
        ),


        "url": (

            metadata.get("url")

            or metadata.get("source")

            or metadata.get("link")

            or ""

        ),


        # Full Exa content.
        # No chunking here.
        "text": text,


        "provider": "exa",


        "published_date": (
            metadata.get(
                "published_date"
            )
        ),


        "author": (
            metadata.get(
                "author"
            )
        ),

    }



# ==========================================================
# Public API
# ==========================================================


async def search_exa(
    query: str,
) -> list[dict[str, Any]]:
    """
    Search web using Exa.

    Pipeline stage:

        Exa
          |
          v
        documents
          |
          v
        filter.py
          |
          v
        chunker.py
          |
          v
        embeddings

    Returns normalized full documents.
    """


    logger.info(
        "Starting Exa web search for '%s'.",
        query,
    )



    documents = await _search(
        query
    )



    if not documents:

        logger.warning(
            "Exa returned no results for '%s'.",
            query,
        )

        raise ValueError(
            f"No Exa search results for '{query}'."
        )



    normalized_documents: list[
        dict[str, Any]
    ] = []



    for document in documents:

        normalized = _normalize_document(
            document,
            query,
        )


        if normalized is not None:

            normalized_documents.append(
                normalized
            )



    logger.info(
        "Prepared %d normalized Exa documents.",
        len(normalized_documents),
    )



    if not normalized_documents:

        raise ValueError(
            "Exa returned no usable documents."
        )



    return normalized_documents
