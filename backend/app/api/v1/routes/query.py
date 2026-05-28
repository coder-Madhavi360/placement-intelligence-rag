import logging

from fastapi import APIRouter, Depends

from app.api.deps import get_rag_service
from app.core.exceptions import ApplicationError
from app.schemas.rag import RAGQueryRequest, RAGQueryResponse
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalServiceError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("", response_model=RAGQueryResponse)
async def query_rag(
    payload: RAGQueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> RAGQueryResponse:
    """Run a multimodal RAG query against the configured retriever."""
    try:
        return await rag_service.answer(payload)
    except RetrievalServiceError as exc:
        logger.exception("RAG query endpoint failed")
        raise ApplicationError(str(exc)) from exc
