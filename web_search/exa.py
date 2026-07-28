"""
Exa web retrieval layer.

Module Responsibilities:

- execute Exa search;
- normalize external documents;
- create WebDocument contracts;
- preserve metadata.
"""


from __future__ import annotations


import logging
import os

from uuid import uuid4
from datetime import datetime, timezone
from typing import Any


from exa_py import AsyncExa


from web_search.models import (
    WebDocument,
    Source
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
        "EXA_TOKEN environment variable is missing."
    )



# ==========================================================
# Client
# ==========================================================


_exa_client: AsyncExa | None = None



def get_exa_client() -> AsyncExa:
    """
    Singleton Exa client.
    """

    global _exa_client


    if _exa_client is None:

        logger.info(
            "Initializing Exa client."
        )


        _exa_client = AsyncExa(
            api_key=EXA_API_KEY
        )


    return _exa_client



# ==========================================================
# Timestamp
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



# ==========================================================
# Normalization
# ==========================================================


def _extract_value(
    document: Any,
    field: str,
    default: Any = None,
) -> Any:
    """
    Safe attribute extraction.
    """

    return getattr(
        document,
        field,
        default,
    )



def _normalize_document(
    document: Any,
) -> WebDocument | None:
    """
    Convert Exa object into internal contract.
    """

    text = (
        _extract_value(
            document,
            "text",
            "",
        )
        or ""
    ).strip()



    if not text:

        logger.debug(
            "Skipping empty Exa document."
        )

        return None



    return WebDocument(

        id=str(uuid4()),

        text=text,

        source=Source(
        
            title=(
                _extract_value(
                document,
                "title",
                None,
                )
                or "Untitled"
            ),

            url=(
                _extract_value(
                    document,
                    "url",
                    None,
                )
                or ""
            ),

            provider="exa",

            author=_extract_value(
                document,
                "author",
            ),

            published_date=_extract_value(
                document,
                "published_date",
            ),
        ),

        created_at=_current_timestamp(),

        last_access=_current_timestamp(),

    )



# ==========================================================
# Search
# ==========================================================


async def _search_exa(
    query: str,
) -> list[Any]:
    """
    Execute Exa search.
    """

    client = get_exa_client()


    logger.info(
        "Exa search started query=%s",
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
    Retrieve and normalize web documents.

    Input:

        normalized query


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



    documents = await _search_exa(
        query
    )



    if not documents:

        logger.warning(
            "Exa returned no documents."
        )

        return []



    normalized: list[WebDocument] = []



    for document in documents:

        item = _normalize_document(
            document
        )


        if item:

            normalized.append(
                item
            )



    logger.info(

        "Exa normalized documents=%d",

        len(normalized),

    )



    return normalized
