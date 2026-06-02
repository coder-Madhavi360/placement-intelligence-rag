import logging

from fastapi import APIRouter, Depends

from core.api.deps import get_rag_service
from core.exceptions import ApplicationError
from core.schemas.rag import RAGQueryRequest, RAGQueryResponse
from generation.rag_service import RAGService
from retrieval.service import RetrievalServiceError

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


