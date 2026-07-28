from __future__ import annotations


import logging
import time
from uuid import uuid4
from datetime import datetime, timezone


from web_search.domain.contracts import SearchProvider
from web_search.domain.models import (
    NormalizedQuery,
    WebDocument,
    Source,
)


logger = logging.getLogger(__name__)



class ExaSearchClient(SearchProvider):
    """
    Exa search provider implementation.
    """

    def __init__(
        self,
        http_client,
        api_url: str,
        api_key: str,
        results_limit: int = 10,
    ):
        self._http_client = http_client
        self._api_url = api_url
        self._api_key = api_key
        self._results_limit = results_limit



    def _timestamp(self) -> int:
        return int(
            datetime.now(
                timezone.utc
            ).timestamp()
        )



    def _normalize_document(
        self,
        document: dict,
    ) -> WebDocument | None:
        """
        Convert Exa response into domain model.
        """

        text = (
            document.get("text")
            or ""
        ).strip()


        if not text:
            return None


        return WebDocument(
            id=str(uuid4()),

            text=text,

            source=Source(
                title=(
                    document.get("title")
                    or "Untitled"
                ),

                url=(
                    document.get("url")
                    or ""
                ),

                provider="exa",

                author=document.get(
                    "author"
                ),

                published_date=document.get(
                    "published_date"
                ),
            ),

            created_at=self._timestamp(),

            last_access=self._timestamp(),
        )



    async def search(
        self,
        query: NormalizedQuery,
    ) -> list[WebDocument]:
        """
        Execute Exa search.
        """

        response = await self._http_client.request(
            method="POST",
            url=self._api_url,
            headers={
                "x-api-key": self._api_key,
                "Content-Type": "application/json",
            },
            json={
                "query": query.normalized,
                "num_results": self._results_limit,
                "contents": {
                    "text": True,
                },
            },
        )


        results = (
            response.get("results")
            or []
        )


        documents: list[WebDocument] = []


        for item in results:

            document = self._normalize_document(
                item
            )

            if document:
                documents.append(
                    document
                )


        logger.info(
            "Exa documents received=%d",
            len(documents),
        )


        return documents
