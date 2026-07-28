"""
Main web search pipeline orchestration.
"""

from __future__ import annotations


import logging
import uuid


from datetime import datetime, UTC


from web_search.application.state import PipelineState
from web_search.application.services import (
    SearchService,
    MemoryService,
    Embedder,
    Reranker,
)


from web_search.domain.models import (
    AgentContext,
    PipelineMetadata,
)


from web_search.processing.chunking import chunk_documents
from web_search.processing.filtering import filter_chunks
from web_search.processing.retrieval import retrieve_by_embedding_similarity
from web_search.processing.compression import compress_chunks
from web_search.processing.context import optimize_context


from web_search.query_normalizer import preprocess_query


logger = logging.getLogger(__name__)



class WebSearchPipeline:
    """
    Application pipeline.

    Contains orchestration only.
    Business logic stays in processing layer.
    """



    def __init__(
        self,
        search_service: SearchService,
        memory_service: MemoryService,
        embedder: Embedder,
        reranker: Reranker,
    ):
        self.search_service = search_service
        self.memory_service = memory_service
        self.embedder = embedder
        self.reranker = reranker



    async def execute(
        self,
        query: str,
    ) -> AgentContext:
        """
        Execute complete pipeline.
        """


        normalized = preprocess_query(
            query
        )


        state = PipelineState(

            query=normalized,

            metadata=PipelineMetadata(

                request_id=str(
                    uuid.uuid4()
                ),

                query=normalized.normalized,

                created_at=int(
                    datetime.now(
                        UTC
                    ).timestamp()
                ),

            ),

        )


        await self._memory_stage(
            state
        )


        if not state.cache_hit:

            await self._retrieval_stage(
                state
            )


        if not state.ranked_chunks:

            return AgentContext(
                metadata=state.metadata
            )


        state.compressed_chunks = await compress_chunks(
            query=state.query.normalized,
            chunks=state.ranked_chunks,
        )


        state.mark_completed(
            "compression"
        )


        if not state.compressed_chunks:

            return AgentContext(
                metadata=state.metadata
            )


        state.context = optimize_context(
            query=state.query.normalized,
            chunks=state.compressed_chunks,
        )


        state.mark_completed(
            "context"
        )


        return AgentContext(

            **state.context.model_dump(),

            metadata=state.metadata,

        )



    async def _memory_stage(
        self,
        state: PipelineState,
    ) -> None:
        """
        Memory lookup stage.
        """


        cached = await self.memory_service.lookup(
            state.query
        )


        if not cached:

            return


        state.cache_hit = True

        state.mark_completed(
            "memory_lookup"
        )


        state.ranked_chunks = [
            item.chunk
            for item in cached
        ]



    async def _retrieval_stage(
        self,
        state: PipelineState,
    ) -> None:
        """
        Fresh web retrieval pipeline.
        """


        state.documents = await self.search_service.execute(
            state.query
        )


        state.mark_completed(
            "exa_retrieval"
        )


        state.chunks = chunk_documents(
            state.documents
        )


        state.mark_completed(
            "chunking"
        )


        filtered = filter_chunks(
            chunks=state.chunks,
            query=state.query.normalized,
        )


        state.mark_completed(
            "filtering"
        )


        embedded = await retrieve_by_embedding_similarity(
            query=state.query.normalized,
            chunks=filtered,
            embedder=self.embedder,
        )


        state.mark_completed(
            "embedding_retrieval"
        )


        state.ranked_chunks = await self.reranker.rerank(
            query=state.query.normalized,
            chunks=embedded,
        )


        state.mark_completed(
            "reranking"
        )


        if state.chunks:

            await self.memory_service.save(
                state.chunks
            )
