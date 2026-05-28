import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_rag_service
from app.schemas.rag import RAGQueryRequest, RAGQueryResponse
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalServiceError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(
    payload: RAGQueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> RAGQueryResponse:
    """Retrieve context and generate a grounded placement intelligence answer."""
    try:
        return await rag_service.answer(payload)
    except RetrievalServiceError as exc:
        logger.exception("RAG retrieval endpoint failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
