import logging
import os

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ==========================================================
# Cloudflare Workers AI Embeddings
# ==========================================================

REQUEST_TIMEOUT = 60.0

MODEL_EMBEDDINGS = "@cf/qwen/qwen3-embedding-0.6b"

EMBEDDING_INSTRUCTION = (
    "Given a user question, retrieve relevant Wikipedia passages."
)

CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID")
CF_API_TOKEN = os.getenv("CF_API_TOKEN")

if not CF_ACCOUNT_ID:
    raise RuntimeError("Environment variable CF_ACCOUNT_ID is not configured.")

if not CF_API_TOKEN:
    raise RuntimeError("Environment variable CF_API_TOKEN is not configured.")

API_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/"
    f"{CF_ACCOUNT_ID}/ai/run/{MODEL_EMBEDDINGS}"
)


class CloudflareEmbeddingError(Exception):
    """Cloudflare Workers AI embedding error."""


_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """
    Lazily create and reuse AsyncClient.
    """

    global _client

    if _client is None:
        logger.info("Initializing Cloudflare AsyncClient.")

        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT),
            headers={
                "Authorization": f"Bearer {CF_API_TOKEN}",
                "Content-Type": "application/json",
            },
            follow_redirects=True,
        )

    return _client


class CloudflareEmbeddings:
    """
    Cloudflare Workers AI Embeddings.
    """

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(httpx.HTTPError),
        reraise=True,
    )
    async def _request(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Request embeddings from Cloudflare.
        """

        logger.debug(
            "Requesting embeddings for %d texts.",
            len(texts),
        )

        try:
            client = get_http_client()

            response = await client.post(
                API_URL,
                json={
                    "text": texts,
                    "instruction": EMBEDDING_INSTRUCTION,
                },
            )

            response.raise_for_status()

            payload = response.json()

            if not payload.get("success", False):
                logger.error(
                    "Cloudflare returned an unsuccessful response: %s",
                    payload.get("errors"),
                )
                raise CloudflareEmbeddingError(
                    str(payload.get("errors"))
                )

            data = (
                payload.get("result", {})
                .get("data", [])
            )

            embeddings = [
                item.get("embedding")
                for item in data
                if item.get("embedding") is not None
            ]

            if len(embeddings) != len(texts):
                logger.error(
                    "Embedding count mismatch. Sent=%d Received=%d",
                    len(texts),
                    len(embeddings),
                )

                raise CloudflareEmbeddingError(
                    "Cloudflare returned an unexpected number of embeddings."
                )

            logger.debug(
                "Successfully received %d embeddings.",
                len(embeddings),
            )

            return embeddings

        except httpx.HTTPError:
            logger.exception("Cloudflare HTTP request failed.")
            raise

        except Exception:
            logger.exception("Unexpected Cloudflare embedding error.")
            raise

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.
        """

        if not texts:
            return []

        return await self._request(texts)

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate embedding for a search query.
        """

        embeddings = await self._request([text])

        return embeddings[0]


# ==========================================================
# Singleton
# ==========================================================

_embeddings: CloudflareEmbeddings | None = None


def get_embedding_model() -> CloudflareEmbeddings:
    """
    Return singleton Cloudflare embedding model.
    """

    global _embeddings

    if _embeddings is None:
        logger.info("Initializing CloudflareEmbeddings.")

        _embeddings = CloudflareEmbeddings()

    return _embeddings
