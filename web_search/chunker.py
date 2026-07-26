"""
Document chunking layer.

Pipeline:

Exa
 |
 v
WebDocument
 |
 v
chunker.py
 |
 v
DocumentChunk
 |
 v
filter.py


Responsibilities:

- split documents into semantic chunks;
- preserve document metadata;
- generate stable chunk identifiers;
- prepare chunks for filtering.

This module does NOT know about:

- Exa API;
- embeddings;
- Qdrant;
- filtering;
- reranking;
- compression;
- LLM.
"""


from __future__ import annotations


import hashlib
import logging
import time

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
# ID generation
# ==========================================================


def _generate_chunk_id(
    document: WebDocument,
    chunk_index: int,
) -> str:
    """
    Generate deterministic chunk id.
    """

    raw = (

        f"{document.url}:"
        f"{chunk_index}:"
        f"{document.title}"

    )


    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()



# ==========================================================
# Text splitting
# ==========================================================


def _split_text(
    text: str,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Lightweight character based splitter.
    Designed for Render free tier.
    """

    if not text:

        return []


    text = text.strip()


    chunks: list[str] = []


    start = 0


    text_length = len(text)



    while start < text_length:


        end = min(

            start + CHUNK_SIZE,

            text_length,

        )


        chunk = text[start:end]


        chunk = chunk.strip()



        if chunk:

            chunks.append(
                chunk
            )



        if end >= text_length:

            break



        start = end - CHUNK_OVERLAP



        if start < 0:

            start = 0



    return chunks



# ==========================================================
# Public API
# ==========================================================


def chunk_document(
    document: WebDocument,
) -> list[DocumentChunk]:
    """
    Convert one document into semantic chunks.


    Input:

        WebDocument


    Output:

        list[DocumentChunk]


    Preserves:

    - title
    - url
    - provider
    - author
    - published_date
    """

    chunks = _split_text(
        document.text
    )


    if not chunks:

        return []



    created_at = int(
        time.time()
    )


    result: list[DocumentChunk] = []



    for index, text in enumerate(
        chunks
    ):


        result.append(

            DocumentChunk(

                id=_generate_chunk_id(

                    document,

                    index,

                ),


                query=document.query,


                title=document.title,


                url=document.url,


                text=text,


                provider=document.provider,


                chunk_index=index,


                author=document.author,


                published_date=document.published_date,


                created_at=created_at,


                last_access=created_at,

            )

        )



    return result



def chunk_documents(
    documents: list[WebDocument],
) -> list[DocumentChunk]:
    """
    Chunk multiple documents.


    Pipeline:

        Exa

          |

        WebDocument

          |

        DocumentChunk

    """

    if not documents:

        return []



    logger.info(

        "Chunking documents. count=%d",

        len(documents),

    )



    chunks: list[DocumentChunk] = []



    for document in documents:


        chunks.extend(

            chunk_document(
                document
            )

        )



    logger.info(

        "Created chunks. count=%d",

        len(chunks),

    )



    return chunks
