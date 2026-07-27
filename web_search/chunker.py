"""
Document chunking layer.

Pipeline position:

EXA SEARCH
    |
    v
Document normalize
    |
    v
chunker.py
    |
    v
filter.py


Responsibilities:

- split documents into chunks;
- preserve metadata;
- create DocumentChunk models;
- generate chunk identifiers.


This module does NOT:

- filter chunks;
- rank chunks;
- generate embeddings;
- access Qdrant;
- call LLM;
"""

from __future__ import annotations


import logging
import hashlib


from datetime import datetime, timezone


from langchain_text_splitters import RecursiveCharacterTextSplitter


from web_search.models import (
    WebDocument,
    DocumentChunk,
)


logger = logging.getLogger(__name__)



# ==========================================================
# Configuration
# ==========================================================


CHUNK_SIZE = 1200

CHUNK_OVERLAP = 200



# ==========================================================
# Helpers
# ==========================================================


def _timestamp() -> int:
    """
    Current UTC unix timestamp.
    """

    return int(
        datetime.now(
            timezone.utc
        ).timestamp()
    )



def _create_chunk_id(
    url: str,
    index: int,
    text: str,
) -> str:
    """
    Stable chunk identifier.
    """

    raw = (
        f"{url}:{index}:{text[:100]}"
    )


    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()



# ==========================================================
# Splitter
# ==========================================================


def _get_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create recursive text splitter.
    """

    return RecursiveCharacterTextSplitter(

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
# Public API
# ==========================================================


def chunk_documents(
    documents: list[WebDocument],
) -> list[DocumentChunk]:
    """
    Convert documents into retrieval chunks.

    Input:

        list[WebDocument]


    Output:

        list[DocumentChunk]
    """

    if not documents:

        return []


    logger.info(
        "Chunking documents=%d",
        len(documents),
    )


    splitter = _get_splitter()


    timestamp = _timestamp()


    chunks: list[DocumentChunk] = []



    for document in documents:


        parts = splitter.split_text(
            document.text
        )


        for index, text in enumerate(
            parts
        ):


            if not text.strip():

                continue



            chunks.append(

                DocumentChunk(

                    id=_create_chunk_id(

                        document.url,

                        index,

                        text,

                    ),


                    query=document.query,


                    title=document.title,


                    url=document.url,


                    text=text,


                    provider=document.provider,


                    chunk_index=index,


                    author=document.author,


                    published_date=document.published_date,


                    created_at=timestamp,


                    last_access=timestamp,

                )

            )



    logger.info(
        "Created chunks=%d",
        len(chunks),
    )


    return chunks
