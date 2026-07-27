"""
Document chunking layer.

Module Responsibilities:

- split documents into chunks;
- preserve metadata;
- create DocumentChunk models;
- generate chunk identifiers.

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
    document_id: str,
    index: int,
    text: str,
) -> str:
    """
    Stable chunk identifier.
    """

    raw = (
        f"{document_id}:{index}:{text[:100]}"
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
                        
                        document.id,
                        
                        index,
                        
                        text,
                        
                    ),


                    document_id=document.id,

                    text=text,

                    source=document.source,

                    chunk_index=index,

                    created_at=timestamp,

                    last_acces=timestamp
                    
                )
                
            )



    logger.info(
        "Created chunks=%d",
        len(chunks),
    )


    return chunks
