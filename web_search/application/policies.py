"""
Pipeline decision policies.

Contains business rules that decide:
- whether to use memory;
- whether to fallback to web;
- whether pipeline should continue;
- quality thresholds.
"""

from __future__ import annotations


from web_search.application.state import PipelineState



class MemoryPolicy:
    """
    Controls cache/memory usage decisions.
    """

    def should_use_memory(
        self,
        state: PipelineState,
    ) -> bool:
        """
        Decide whether cached results are acceptable.
        """

        return bool(
            state.cache_hit
            and state.ranked_chunks
        )



    def should_store_memory(
        self,
        state: PipelineState,
    ) -> bool:
        """
        Decide whether fresh results should be stored.
        """

        return bool(
            state.chunks
            and not state.cache_hit
        )



class RetrievalPolicy:
    """
    Controls retrieval continuation rules.
    """

    def should_continue_after_documents(
        self,
        state: PipelineState,
    ) -> bool:
        """
        Continue only when documents exist.
        """

        return bool(
            state.documents
        )



    def should_continue_after_chunks(
        self,
        state: PipelineState,
    ) -> bool:
        """
        Continue only when chunks exist.
        """

        return bool(
            state.chunks
        )



    def should_continue_after_ranking(
        self,
        state: PipelineState,
    ) -> bool:
        """
        Continue only with ranked results.
        """

        return bool(
            state.ranked_chunks
        )



class ContextPolicy:
    """
    Controls final context generation.
    """

    def should_build_context(
        self,
        state: PipelineState,
    ) -> bool:
        """
        Context requires compressed chunks.
        """

        return bool(
            state.compressed_chunks
        )



class PipelinePolicy:
    """
    Aggregated pipeline rules.
    """

    def __init__(
        self,
        memory: MemoryPolicy | None = None,
        retrieval: RetrievalPolicy | None = None,
        context: ContextPolicy | None = None,
    ):
        self.memory = (
            memory
            or MemoryPolicy()
        )

        self.retrieval = (
            retrieval
            or RetrievalPolicy()
        )

        self.context = (
            context
            or ContextPolicy()
        )
