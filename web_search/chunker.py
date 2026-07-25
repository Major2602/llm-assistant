"""
Chunking layer for web search pipeline.

Responsible for:
- splitting filtered documents into semantic chunks;
- preparing chunks for embedding.

This module does not know about:
- Exa;
- embeddings;
- reranking;
- Qdrant.
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
# Chunking
# ==========================================================


def chunk_documents(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert filtered documents into embedding chunks.

    Input:
        Filtered Exa documents.

    Output:
        20-30 candidate chunks for embedding.
    """

    now = int(
        datetime.now(
            timezone.utc
        ).timestamp()
    )


    chunks: list[dict[str, Any]] = []


    for document in documents:

        text = (
            document.get("text")
            or ""
        ).strip()


        if not text:
            continue


        text_chunks = _splitter.split_text(
            text
        )


        for index, chunk in enumerate(text_chunks):

            if not chunk.strip():
                continue


            chunks.append(
                {
                    "id": str(
                        uuid.uuid4()
                    ),

                    "query": (
                        document.get("query")
                    ),

                    "title": (
                        document.get("title")
                        or "Untitled"
                    ),

                    "url": (
                        document.get("url")
                        or ""
                    ),

                    "text": chunk,

                    "provider": (
                        document.get("provider")
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

                    "created_at": now,

                    "last_access": now,
                }
            )


    logger.info(
        "Created %d chunks from %d filtered documents.",
        len(chunks),
        len(documents),
    )


    return chunks
