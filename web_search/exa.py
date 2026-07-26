"""
Exa web search retrieval layer.

Pipeline position:

USER QUERY
    |
    v
Query preprocessing
    |
    v
Exa Search
    |
    v
WebDocument
    |
    v
chunker.py


Responsibilities:

- execute Exa search;
- retrieve documents;
- normalize metadata;
- return WebDocument models.


This module does NOT know about:

- chunking;
- filtering;
- embeddings;
- reranking;
- Qdrant;
- LLM.
"""

from __future__ import annotations


import logging
import os

from datetime import datetime, timezone

from typing import Any


from exa_py import AsyncExa


from web_search.models import (
    WebDocument,
)


logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================


EXA_API_KEY = os.getenv(
    "EXA_TOKEN"
)


EXA_RESULTS = int(
    os.getenv(
        "EXA_RESULTS",
        "100",
    )
)


if not EXA_API_KEY:
    raise RuntimeError(
        "Environment variable EXA_TOKEN is not configured."
    )



# ==========================================================
# Client
# ==========================================================


_client: AsyncExa | None = None



def get_exa_client() -> AsyncExa:
    """
    Return singleton Exa client.
    """

    global _client


    if _client is None:

        logger.info(
            "Initializing Exa client."
        )

        _client = AsyncExa(
            api_key=EXA_API_KEY
        )


    return _client



# ==========================================================
# Normalization
# ==========================================================


def _current_timestamp() -> int:
    """
    Return UTC unix timestamp.
    """

    return int(
        datetime.now(
            timezone.utc
        ).timestamp()
    )



def _normalize_document(
    document: Any,
    query: str,
) -> WebDocument | None:
    """
    Convert Exa response into internal model.

    Exa output:

        external object

            |

            v

        WebDocument
    """

    text = (
        getattr(
            document,
            "text",
            None,
        )
        or ""
    ).strip()


    if not text:

        logger.debug(
            "Skipping empty Exa document."
        )

        return None



    return WebDocument(

        query=query,


        title=(

            getattr(
                document,
                "title",
                None,
            )

            or

            "Untitled"

        ),


        url=(

            getattr(
                document,
                "url",
                None,
            )

            or

            ""

        ),


        text=text,


        provider="exa",


        author=getattr(
            document,
            "author",
            None,
        ),


        published_date=getattr(
            document,
            "published_date",
            None,
        ),


        created_at=_current_timestamp(),


        last_access=_current_timestamp(),

    )



# ==========================================================
# Search
# ==========================================================


async def _search(
    query: str,
) -> list[Any]:
    """
    Execute Exa search request.
    """

    client = get_exa_client()


    logger.info(
        "Searching Exa. query=%s",
        query,
    )



    response = await client.search_and_contents(

        query=query,


        num_results=EXA_RESULTS,


        text=True,


        type="auto",

    )


    return response.results



# ==========================================================
# Public API
# ==========================================================


async def search_exa(
    query: str,
) -> list[WebDocument]:
    """
    Search Exa and return normalized documents.

    Output:

        list[WebDocument]

    Next stage:

        chunker.py
    """

    query = query.strip()


    if not query:

        raise ValueError(
            "Query cannot be empty."
        )



    documents = await _search(
        query
    )


    if not documents:

        raise ValueError(
            "Exa returned no documents."
        )



    normalized: list[WebDocument] = []



    for document in documents:

        item = _normalize_document(

            document,

            query,

        )


        if item:

            normalized.append(
                item
            )



    logger.info(

        "Exa normalized documents=%d",

        len(normalized),

    )



    if not normalized:

        raise ValueError(
            "No usable Exa documents."
        )



    return normalized
