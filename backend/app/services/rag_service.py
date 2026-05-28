import logging

from app.core.config import Settings
from app.retrieval.search_models import RetrievalQueryRequest
from app.schemas.rag import RAGQueryRequest, RAGQueryResponse
from app.schemas.rag import RetrievedContext
from app.schemas.common import Modality
from app.services.retrieval_service import RetrievalService, RetrievalServiceError

logger = logging.getLogger(__name__)


class RAGService:
    """High-level RAG workflow service.

    This service preserves the public /api/v1/query response contract while
    delegating retrieval to the production FAISS-backed retrieval engine.
    """

    def __init__(self, settings: Settings, retrieval_service: RetrievalService) -> None:
        self.settings = settings
        self.retrieval_service = retrieval_service

    async def answer(self, request: RAGQueryRequest) -> RAGQueryResponse:
        logger.info(
            "RAG query received on legacy response endpoint: top_k=%s filters=%s model=%s",
            request.top_k,
            request.filters,
            self.settings.embedding_model,
        )

        try:
            retrieval_response = await self.retrieval_service.query(
                RetrievalQueryRequest(
                    query=request.query,
                    top_k=request.top_k,
                    filters=request.filters,
                )
            )
        except RetrievalServiceError:
            logger.exception("RAG retrieval failed")
            raise

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

        logger.info(
            "RAG query completed: contexts=%s scores=%s",
            len(contexts),
            [round(context.score, 6) for context in contexts],
        )

        if contexts:
            answer = (
                f"Retrieved {len(contexts)} relevant placement context chunk(s). "
                "Use these contexts to generate a grounded final answer."
            )
        else:
            answer = "No relevant context found in the FAISS placement index."

        return RAGQueryResponse(
            answer=answer,
            contexts=contexts,
            model=self.settings.embedding_model,
        )
