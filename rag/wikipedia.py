import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from urllib.parse import quote

import httpx
from langdetect import LangDetectException, detect
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

SUPPORTED_LANGS = {"en", "ru", "de", "fr", "es", "it", "pt", "ja", "zh", "ar"}

USER_AGENT = "Chainlit-RAG/2.0"

REQUEST_TIMEOUT = 20.0

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

_client = httpx.AsyncClient(
    timeout=httpx.Timeout(REQUEST_TIMEOUT),
    headers={"User-Agent": USER_AGENT},
    follow_redirects=True,
)

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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(httpx.HTTPError),
    reraise=True,
)
async def _search(api: str, entity: str) -> list[dict[str, Any]]:
    """Search Wikipedia for an article."""

    try:
        logger.debug("Wikipedia search: '%s' (%s)", entity, api)

        response = await _client.get(
            api,
            params={
                "action": "query",
                "list": "search",
                "srsearch": entity,
                "utf8": 1,
                "format": "json",
            },
        )

        # response.raise_for_status()
        if response.status_code != 200:
            logger.error("Wikipedia status=%s", response.status_code)
            logger.error("Headers: %s", response.headers)
            logger.error("Body: %s", response.text)
            response.raise_for_status()

        data = response.json()

        return (
            data.get("query", {})
            .get("search", [])
        )

    except httpx.HTTPError:
        logger.exception("Wikipedia search failed for '%s'.", entity)
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(httpx.HTTPError),
    reraise=True,
)
async def _page(api: str, title: str) -> str:
    """Load Wikipedia article text."""

    try:
        response = await _client.get(
            api,
            params={
                "action": "query",
                "prop": "extracts",
                "titles": title,
                "explaintext": 1,
                "format": "json",
            },
        )

        response.raise_for_status()

        data = response.json()

        pages = (
            data.get("query", {})
            .get("pages", {})
        )

        if not pages:
            return ""

        page = next(iter(pages.values()))

        return page.get("extract", "")

    except httpx.HTTPError:
        logger.exception("Failed loading Wikipedia page '%s'.", title)
        raise


async def _load(entity: str, lang: str) -> dict[str, str] | None:
    """Load article from a specific language Wikipedia."""

    api = f"https://{lang}.wikipedia.org/w/api.php"

    results = await _search(api, entity)

    if not results:
        logger.info(
            "Wikipedia article '%s' not found in language '%s'.",
            entity,
            lang,
        )
        return None

    title = results[0].get("title")

    if not title:
        return None

    text = await _page(api, title)

    if not text.strip():
        logger.info("Wikipedia page '%s' is empty.", title)
        return None

    logger.info(
        "Loaded Wikipedia article '%s' (%s).",
        title,
        lang,
    )

    return {
        "title": title,
        "language": lang,
        "source": f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
        "text": text,
    }


def _detect_language(entity: str) -> str:
    """Detect entity language."""

    if len(entity.strip()) < 3:
        return "en"

    try:
        lang = detect(entity)
    except LangDetectException:
        logger.debug("Language detection failed for '%s'. Using English.", entity)
        return "en"

    if lang not in SUPPORTED_LANGS:
        logger.debug("Unsupported language '%s'. Using English.", lang)
        return "en"

    return lang


def _chunk_document(document: dict[str, str]) -> list[dict[str, str | int]]:
    """Split article into chunks."""

    now = datetime.now(timezone.utc).isoformat()

    chunks = _splitter.split_text(document["text"])

    result: list[dict[str, str | int]] = []

    for index, chunk in enumerate(chunks):
        result.append(
            {
                "id": str(uuid.uuid4()),
                "entity": document["title"],
                "chunk_index": index,
                "text": chunk,
                "title": document["title"],
                "language": document["language"],
                "source": document["source"],
                "created_at": now,
                "last_access": now,
            }
        )

    logger.info(
        "Wikipedia article '%s' split into %d chunks.",
        document["title"],
        len(result),
    )

    return result


async def load_wikipedia(entity: str) -> list[dict[str, str | int]]:
    """
    Load a Wikipedia article and split it into chunks.
    """

    logger.info("Loading Wikipedia article: %s", entity)

    lang = _detect_language(entity)

    logger.info("Detected language: %s", lang)

    document = await _load(entity, lang)

    if document is None and lang != "en":
        logger.warning(
            "Falling back to English Wikipedia for '%s'.",
            entity,
        )

        document = await _load(entity, "en")

    if document is None:
        logger.warning(
            "Wikipedia article '%s' not found.",
            entity,
        )

        raise ValueError(
            f"Wikipedia article '{entity}' not found."
        )

    return _chunk_document(document)
