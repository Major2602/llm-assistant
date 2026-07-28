"""
Cloudflare Workers AI Embedding Service.


Used by:

- dense similarity retrieval;
- Qdrant dense vectors;
- extractive compression.


Module Responsibilities:

- generate query embeddings;
- generate document embeddings;
- batch requests;
- normalize Cloudflare responses;
- validate embedding dimensions.
"""


from __future__ import annotations


import logging
import os


from typing import Any


import httpx


from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


from web_search.models import DenseVector


logger = logging.getLogger(__name__)



# ==========================================================
# Configuration
# ==========================================================


REQUEST_TIMEOUT = float(
    os.getenv(
        "CF_EMBEDDING_TIMEOUT",
        "60",
    )
)


EMBEDDING_MODEL = os.getenv(
    "CF_EMBEDDING_MODEL",
    "@cf/qwen/qwen3-embedding-0.6b",
)


MAX_BATCH_SIZE = int(
    os.getenv(
        "CF_EMBEDDING_BATCH_SIZE",
        "32",
    )
)


EMBEDDING_DIMENSION = int(
    os.getenv(
        "CF_EMBEDDING_DIMENSION",
        "1024",
    )
)



CF_ACCOUNT_ID = os.getenv(
    "CF_ACCOUNT_ID"
)


CF_API_TOKEN = os.getenv(
    "CF_API_TOKEN"
)



if not CF_ACCOUNT_ID:

    raise RuntimeError(
        "CF_ACCOUNT_ID is not configured."
    )



if not CF_API_TOKEN:

    raise RuntimeError(
        "CF_API_TOKEN is not configured."
    )



API_URL = (

    "https://api.cloudflare.com/client/v4/accounts/"

    f"{CF_ACCOUNT_ID}"

    "/ai/run/"

    f"{EMBEDDING_MODEL}"

)



# ==========================================================
# Exceptions
# ==========================================================


class CloudflareEmbeddingError(Exception):
    """
    Embedding service error.
    """



# ==========================================================
# HTTP Client
# ==========================================================


_client: httpx.AsyncClient | None = None



def get_http_client() -> httpx.AsyncClient:
    """
    Shared async HTTP client.
    """

    global _client


    if _client is None:

        logger.info(
            "Initializing Cloudflare embedding client."
        )


        _client = httpx.AsyncClient(

            timeout=httpx.Timeout(
                REQUEST_TIMEOUT
            ),

            headers={

                "Authorization":
                    f"Bearer {CF_API_TOKEN}",

                "Content-Type":
                    "application/json",

            },

            follow_redirects=True,

        )


    return _client



# ==========================================================
# Helpers
# ==========================================================


def _clean_texts(
    texts: list[str],
) -> list[str]:
    """
    Remove empty values.
    """

    cleaned=[]

    for text in texts:

         if text is None:

              cleaned.append("")

         else:

              cleaned.append(
                   text.strip()
              )
 
    return cleaned



def _split_batches(
    texts: list[str],
) -> list[list[str]]:
    """
    Split texts into API batches.
    """

    return [

        texts[index:index + MAX_BATCH_SIZE]

        for index in range(

            0,

            len(texts),

            MAX_BATCH_SIZE,

        )

    ]



def _validate_dimension(
    vector: list[float],
) -> None:
    """
    Validate embedding dimension.
    """

    if len(vector) != EMBEDDING_DIMENSION:

        raise CloudflareEmbeddingError(

            (

                "Embedding dimension mismatch. "

                f"Expected={EMBEDDING_DIMENSION} "

                f"Received={len(vector)}"

            )

        )



# ==========================================================
# Response parsing
# ==========================================================


def _parse_embeddings(
    payload: dict[str, Any],
) -> list[list[float]]:
    """
    Parse Cloudflare response.

    Expected:

    {
        success: true,
        result:
        {
            data:
            [
                [...]
            ]
        }
    }
    """

    if not payload.get(
        "success",
        False,
    ):

        raise CloudflareEmbeddingError(

            f"Cloudflare error: {payload.get('errors')}"

        )



    result = payload.get(
        "result"
    )


    if not isinstance(
        result,
        dict,
    ):

        raise CloudflareEmbeddingError(
            "Invalid embedding result."
        )



    data = result.get(
        "data"
    )


    if not isinstance(
        data,
        list,
    ):

        raise CloudflareEmbeddingError(
            "Embedding data missing."
        )



    embeddings: list[list[float]] = []



    for item in data:


        vector = None


        if isinstance(
            item,
            list,
        ):

            vector = item


        elif isinstance(
            item,
            dict,
        ):

            vector = item.get(
                "embedding"
            )



        if isinstance(
            vector,
            list,
        ):

            normalized = [

                float(value)

                for value in vector

            ]


            _validate_dimension(
                normalized
            )


            embeddings.append(
                normalized
            )



    if not embeddings:

        raise CloudflareEmbeddingError(
            "No embeddings returned."
        )



    return embeddings



# ==========================================================
# Service
# ==========================================================


class CloudflareEmbeddings:
    """
    Cloudflare embedding wrapper.
    """



    @retry(

        stop=stop_after_attempt(3),

        wait=wait_exponential(

            multiplier=1,

            min=1,

            max=8,

        ),

        retry=retry_if_exception_type(
            httpx.HTTPError
        ),

        reraise=True,

    )
    async def _request(
        self,
        texts: list[str],
    ) -> list[DenseVector]:
        """
        Execute embedding request.
        """

        client = get_http_client()

        if not texts:

            return []

        if any(
         
            not text
            for text in texts
        ):

            raise CloudflareEmbeddingError(
                "Empty text passed to embedding API"
            )


        response = await client.post(

            API_URL,

            json={

                "text": texts,

            },

        )


        response.raise_for_status()


        payload = response.json()


        embeddings = _parse_embeddings(
            payload
        )


        if len(embeddings) != len(texts):

            raise CloudflareEmbeddingError(

                (

                    "Embedding count mismatch. "

                    f"Expected={len(texts)} "

                    f"Received={len(embeddings)}"

                )

            )


        return [
            
            DenseVector(
                
                values=vector
                
            )

            for vector in embeddings
            
        ]



    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[DenseVector]:
        """
        Generate embeddings for multiple texts.
        """

        clean_texts = _clean_texts(
            texts
        )


        if not any(clean_texts):

            return []



        logger.info(

            "Generating document embeddings count=%d",

            len(clean_texts),

        )



        result: list[DenseVector] = []



        for batch in _split_batches(
            clean_texts
        ):

            embeddings = await self._request(
                batch
            )


            result.extend(
                embeddings
            )



        if len(result) != len(clean_texts):

            raise CloudflareEmbeddingError(
                "Final embedding count mismatch."
            )


        return result



    async def embed_query(
        self,
        query: str,
    ) -> DenseVector:
        """
        Generate embedding for query.
        """

        query = query.strip()


        if not query:

            raise CloudflareEmbeddingError(
                "Query cannot be empty."
            )



        result = await self._request(
            [
                query
            ]
        )


        if len(result) != 1:

            raise CloudflareEmbeddingError(
                "Invalid query embedding response."
            )


        return result[0]



# ==========================================================
# Singleton
# ==========================================================


_embedding_service: CloudflareEmbeddings | None = None



def get_embedding_model() -> CloudflareEmbeddings:
    """
    Return singleton embedding model.
    """

    global _embedding_service


    if _embedding_service is None:

        logger.info(
            "Creating Cloudflare embedding singleton."
        )


        _embedding_service = CloudflareEmbeddings()



    return _embedding_service
