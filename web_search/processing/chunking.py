"""
Document chunking processing stage.

Responsibilities:

- split documents into chunks;
- preserve source metadata;
- create DocumentChunk objects.
"""

from __future__ import annotations


import logging


from web_search.domain.models import (
    WebDocument,
    DocumentChunk,
)


logger = logging.getLogger(__name__)



# ==========================================================
# Configuration
# ==========================================================


DEFAULT_CHUNK_SIZE = 1000

DEFAULT_OVERLAP = 200



# ==========================================================
# Chunking
# ==========================================================


def _split_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Split text into overlapping chunks.
    """


    if not text:

        return []



    chunks: list[str] = []


    start = 0


    length = len(text)



    while start < length:


        end = start + chunk_size


        chunk = text[start:end].strip()



        if chunk:

            chunks.append(
                chunk
            )



        start = end - overlap



        if start < 0:

            start = 0



    return chunks



# ==========================================================
# Public API
# ==========================================================


def chunk_documents(
    documents: list[WebDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[DocumentChunk]:
    """
    Convert documents into chunks.
    """


    if not documents:

        return []



    result: list[DocumentChunk] = []



    for document in documents:


        chunks = _split_text(

            document.text,

            chunk_size,

            overlap,

        )



        for index, text in enumerate(chunks):


            result.append(

                DocumentChunk(

                    id=f"{document.id}_{index}",

                    text=text,

                    source=document.source,

                    document_id=document.id,

                    chunk_index=index,

                )

            )



    logger.info(

        "Documents chunked documents=%d chunks=%d",

        len(documents),

        len(result),

    )



    return result
