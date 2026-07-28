# web_search/application/pipeline.py

from __future__ import annotations


import logging


from web_search.application.state import PipelineState
from web_search.application.services import PipelineServices
from web_search.application.policies import PipelinePolicy


from web_search.domain.models import (
    AgentContext,
)


from web_search.processing.chunking import chunk_documents
from web_search.processing.filtering import filter_chunks
from web_search.processing.retrieval import retrieve_by_embedding
from web_search.processing.reranking import rerank_chunks
from web_search.processing.compression import compress_chunks
from web_search.processing.context import optimize_context



logger = logging.getLogger(__name__)



class WebSearchPipeline:
    """
    Main web search application pipeline.

    Responsibilities:
    - stage ordering;
    - dependency coordination;
    - pipeline state management.

    Does not know infrastructure details.
    """

    def __init__(
        self,
        services: PipelineServices,
        policy: PipelinePolicy,
    ):
        self._services = services
        self._policy = policy



    async def execute(
        self,
        state: PipelineState,
    ) -> AgentContext:
        """
        Execute complete search pipeline.
        """

        await self._memory_lookup(
            state
        )


        if not state.cache_hit:

            await self._web_retrieval(
                state
            )


            await self._store_memory(
                state
            )


        if not state.ranked_chunks:

            logger.info(
                "No ranked chunks."
            )

            return AgentContext(
                metadata=state.metadata
            )


        await self._compression(
            state
        )


        if not state.ranked_chunks:

            return AgentContext(
                metadata=state.metadata
            )


        await self._context_build(
            state
        )


        return state.context



    async def _memory_lookup(
        self,
        state: PipelineState,
    ) -> None:
        """
        Retrieve from vector memory.
        """

        results = await self._services.vector_store.search(
            state.query
        )


        state.cache_hit = bool(
            results
        )


        if not results:

            return


        state.ranked_chunks = (
            await self._services.reranker.rerank(
                state.query.normalized,
                [
                    item.chunk
                    for item in results
                ],
            )
        )



    async def _web_retrieval(
        self,
        state: PipelineState,
    ) -> None:
        """
        Execute fresh web retrieval.
        """

        state.documents = (
            await self._services.search_provider.search(
                state.query
            )
        )


        if not state.documents:

            return


        state.chunks = chunk_documents(
            state.documents
        )


        state.filtered_chunks = filter_chunks(
            chunks=state.chunks,
            query=state.query.normalized,
        )


        if not state.filtered_chunks:

            return


        state.embedded_chunks = (
            await retrieve_by_embedding(
                query=state.query.normalized,
                chunks=state.filtered_chunks,
                embedder=self._services.embedder,
            )
        )


        if not state.embedded_chunks:

            return


        state.ranked_chunks = (
            await rerank_chunks(
                query=state.query.normalized,
                chunks=state.embedded_chunks,
                reranker=self._services.reranker,
            )
        )



    async def _store_memory(
        self,
        state: PipelineState,
    ) -> None:
        """
        Store retrieved chunks.
        """

        if not state.chunks:

            return


        await self._services.vector_store.store(
            state.chunks
        )



    async def _compression(
        self,
        state: PipelineState,
    ) -> None:
        """
        Compress ranked chunks.
        """

        state.ranked_chunks = (
            await compress_chunks(
                query=state.query.normalized,
                chunks=state.ranked_chunks,
            )
        )



    async def _context_build(
        self,
        state: PipelineState,
    ) -> None:
        """
        Build final agent context.
        """

        state.context = optimize_context(
            query=state.query.normalized,
            chunks=state.ranked_chunks,
        )
