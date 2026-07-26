"""
Semantic document chunking layer.

Pipeline position:

Exa
 |
 v
Document normalize
 |
 v
chunker.py
 |
 v
filter.py
 |
 v
embedding similarity
 |
 v
reranker.py
 |
 v
qdrant_store.py


Responsibilities:

- convert normalized documents into semantic chunks;
- preserve metadata;
- prepare chunks for filtering and retrieval;
- create Qdrant-ready payload structure.


This module does NOT know about:

- Exa API;
- embeddings;
- BM25;
- Qdrant;
- reranking;
- LLM;
- compression.
"""


from __future__ import annotations


import logging
import uuid

from datetime import datetime, timezone
from typing import Any


from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================


CHUNK_SIZE = 900


CHUNK_OVERLAP = 150


MAX_CHUNKS_PER_DOCUMENT = 30


MIN_CHUNK_LENGTH = 120


MAX_DOCUMENT_LENGTH = 150_000



# ==========================================================
# Splitter
# ==========================================================


_splitter = RecursiveCharacterTextSplitter(

    chunk_size=CHUNK_SIZE,

    chunk_overlap=CHUNK_OVERLAP,

    separators=[

        "\n\n",

        "\n",

        ". ",

        " ",

        "",

    ],

)



# ==========================================================
# Helpers
# ==========================================================


def _timestamp() -> int:
    """
    Current UTC timestamp.
    """

    return int(
        datetime.now(
            timezone.utc
        ).timestamp()
    )



def _clean_text(
    text: str,
) -> str:
    """
    Normalize document text.
    """

    if not text:
        return ""


    return (
        text
        .replace(
            "\x00",
            "",
        )
        .strip()
    )



def _create_document_id(
    document: dict[str, Any],
) -> str:
    """
    Create stable document identifier.
    """

    url = document.get(
        "url",
        "",
    )


    if url:

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                url,
            )
        )


    return str(
        uuid.uuid4()
    )



def _create_chunk(
    *,
    document: dict[str, Any],
    document_id: str,
    text: str,
    index: int,
    total_chunks: int,
    created_at: int,
) -> dict[str, Any]:
    """
    Create normalized chunk payload.
    """

    return {

        "id": str(
            uuid.uuid4()
        ),


        "document_id": document_id,


        "title": (
            document.get(
                "title"
            )
            or "Untitled"
        ),


        "url": (
            document.get(
                "url"
            )
            or ""
        ),


        "text": text,


        "provider": (
            document.get(
                "provider"
            )
            or "exa"
        ),


        "query": document.get(
            "query"
        ),


        "chunk_index": index,


        "total_chunks": total_chunks,


        "char_count": len(
            text
        ),


        "word_count": len(
            text.split()
        ),


        "published_date": (
            document.get(
                "published_date"
            )
        ),


        "author": (
            document.get(
                "author"
            )
        ),


        "created_at": created_at,


        "last_access": created_at,

    }



# ==========================================================
# Public API
# ==========================================================


def chunk_documents(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert normalized documents into semantic chunks.

    Input:

    [
        {
            title,
            url,
            text,
            metadata
        }
    ]


    Output:

    [
        {
            id,
            document_id,
            text,
            metadata
        }
    ]


    These chunks are consumed by:

        filter_chunks()
    """


    if not documents:

        return []



    created_at = _timestamp()


    chunks: list[dict[str, Any]] = []



    for document in documents:


        raw_text = document.get(
            "text",
            "",
        )


        text = _clean_text(
            raw_text
        )


        if not text:

            logger.debug(
                "Skipping empty document."
            )

            continue



        if len(text) > MAX_DOCUMENT_LENGTH:

            logger.warning(

                "Document truncated. title=%s size=%d",

                document.get(
                    "title"
                ),

                len(text),

            )


            text = text[
                :MAX_DOCUMENT_LENGTH
            ]



        document_id = _create_document_id(
            document
        )



        document_chunks = (
            _splitter.split_text(
                text
            )
        )



        document_chunks = [

            chunk.strip()

            for chunk in document_chunks

            if len(chunk.strip())
            >=
            MIN_CHUNK_LENGTH

        ]



        if not document_chunks:

            continue



        total_chunks = min(

            len(document_chunks),

            MAX_CHUNKS_PER_DOCUMENT,

        )



        for index, chunk_text in enumerate(

            document_chunks[:MAX_CHUNKS_PER_DOCUMENT]

        ):


            chunks.append(

                _create_chunk(

                    document=document,

                    document_id=document_id,

                    text=chunk_text,

                    index=index,

                    total_chunks=total_chunks,

                    created_at=created_at,

                )

            )



    logger.info(

        "Created %d chunks from %d documents.",

        len(chunks),

        len(documents),

    )


    return chunks
