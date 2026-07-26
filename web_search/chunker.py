"""
Chunking layer for web search pipeline.

Responsible for:
- splitting raw Exa documents into semantic chunks;
- preparing chunks for filtering and reranking;
- preserving source metadata.

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
reranker.py
 |
 v
qdrant_store.py


This module does not know about:
- Exa API;
- filtering logic;
- embeddings;
- reranking;
- Qdrant;
- LLM.
"""

from __future__ import annotations

import logging
import uuid

from datetime import datetime, timezone
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================


CHUNK_SIZE = 1000

CHUNK_OVERLAP = 150


# Maximum amount of chunks generated
# from a single Exa document.
MAX_CHUNKS_PER_DOCUMENT = 20


# Ignore extremely small fragments.
MIN_CHUNK_LENGTH = 100


# Prevent processing huge unexpected payloads.
MAX_DOCUMENT_LENGTH = 100_000



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


def _current_timestamp() -> int:
    """
    Return current UTC unix timestamp.
    """

    return int(
        datetime.now(
            timezone.utc
        ).timestamp()
    )



def _prepare_text(
    text: str,
) -> str:
    """
    Normalize document text before chunking.
    """

    if not text:
        return ""

    return text.strip()



def _create_chunk(
    document: dict[str, Any],
    text: str,
    index: int,
    timestamp: int,
) -> dict[str, Any]:
    """
    Create internal chunk representation.
    """

    return {

        "id": str(
            uuid.uuid4()
        ),


        "query": (
            document.get(
                "query"
            )
        ),


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


        "chunk_index": index,


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


        "created_at": timestamp,


        "last_access": timestamp,

    }



# ==========================================================
# Public API
# ==========================================================


def chunk_documents(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert raw Exa documents into semantic chunks.

    Input:

        Normalized Exa documents.

    Output:

        Candidate chunks for filtering.

    Pipeline:

        Exa documents
              |
              v
        semantic chunks
              |
              v
        filter.py
    """

    timestamp = _current_timestamp()


    chunks: list[dict[str, Any]] = []


    for document in documents:

        raw_text = (
            document.get(
                "text"
            )
            or ""
        )


        text = _prepare_text(
            raw_text
        )


        if not text:

            logger.debug(
                "Skipping empty document."
            )

            continue



        if len(text) > MAX_DOCUMENT_LENGTH:

            logger.warning(
                "Document too large. Truncating. "
                "title=%s length=%d",
                document.get(
                    "title"
                ),
                len(text),
            )

            text = text[
                :MAX_DOCUMENT_LENGTH
            ]



        document_chunks = _splitter.split_text(
            text
        )



        created = 0


        for index, chunk_text in enumerate(
            document_chunks
        ):

            chunk_text = chunk_text.strip()


            if len(chunk_text) < MIN_CHUNK_LENGTH:

                continue



            chunks.append(
                _create_chunk(
                    document=document,
                    text=chunk_text,
                    index=index,
                    timestamp=timestamp,
                )
            )


            created += 1



            if created >= MAX_CHUNKS_PER_DOCUMENT:

                logger.debug(
                    "Chunk limit reached for document. "
                    "title=%s",
                    document.get(
                        "title"
                    ),
                )

                break



    logger.info(
        "Created %d chunks from %d Exa documents.",
        len(chunks),
        len(documents),
    )


    return chunks
