import logging
import time

from core.config import Settings
from feedback.chat_models import ChatMessage
from retrieval.search_models import RetrievalQueryRequest
from core.schemas.rag import RAGQueryRequest, RAGQueryResponse
from core.schemas.rag import RetrievedContext
from core.schemas.common import Modality
from generation.llm_service import LLMService
from retrieval.retrieval_service import RetrievalService, RetrievalServiceError

logger = logging.getLogger(__name__)


class RAGService:
    """High-level RAG workflow service.

    This service preserves the public /api/v1/query response contract while
    delegating retrieval to the production FAISS-backed retrieval engine.
    """

    def __init__(
        self,
        settings: Settings,
        retrieval_service: RetrievalService,
        llm_service: LLMService,
    ) -> None:
        self.settings = settings
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service

    async def answer(
        self,
        request: RAGQueryRequest,
        *,
        conversation_history: list[ChatMessage] | None = None,
    ) -> RAGQueryResponse:
        """Retrieve relevant chunks, generate an answer, and return sources."""
        top_k = request.top_k or self.settings.max_retrieval_results
        logger.info(
            "RAG query received on legacy response endpoint: top_k=%s filters=%s model=%s",
            top_k,
            request.filters,
            self.settings.embedding_model,
        )

        retrieval_started_at = time.perf_counter()
        try:
            retrieval_response = await self.retrieval_service.query(
                RetrievalQueryRequest(
                    query=request.query,
                    top_k=top_k,
                    filters=request.filters,
                )
            )
        except RetrievalServiceError:
            logger.exception("RAG retrieval failed")
            raise
        retrieval_time_ms = round((time.perf_counter() - retrieval_started_at) * 1000)

        contexts = [
            RetrievedContext(
                id=match.chunk_id,
                content=match.content,
                score=match.score,
                modality=Modality.TEXT,
                metadata=match.metadata,
            )
            for match in retrieval_response.matches
        ]
        sources = self._build_sources(contexts)

        logger.info(
            "RAG retrieval completed: contexts=%s scores=%s retrieval_time_ms=%s",
            len(contexts),
            [round(context.score, 6) for context in contexts],
            retrieval_time_ms,
        )

        generated = self.llm_service.generate(
            request.query,
            contexts,
            conversation_history=conversation_history,
        )
        logger.info(
            "RAG answer generated: model=%s generation_time_ms=%s prompt_token_estimate=%s sources=%s",
            generated.model,
            generated.generation_time_ms,
            generated.prompt_token_estimate,
            sources,
        )

        return RAGQueryResponse(
            answer=generated.answer,
            contexts=contexts,
            sources=sources,
            model=generated.model,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generated.generation_time_ms,
        )

    def _build_sources(self, contexts: list[RetrievedContext]) -> list[str]:
        sources: list[str] = []
        seen: set[str] = set()

        for context in contexts:
            source_file = context.metadata.extra.get("source_file") or context.metadata.source
            page = context.metadata.page
            if not source_file:
                continue

            label = f"{source_file} Page {page}" if page else str(source_file)
            if label in seen:
                continue

            seen.add(label)
            sources.append(label)

        return sources
