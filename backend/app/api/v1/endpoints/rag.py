import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_retrieval_service
from app.retrieval.search_models import RetrievalQueryRequest, RetrievalResponse
from app.services.retrieval_service import RetrievalService, RetrievalServiceError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=RetrievalResponse)
async def query_rag(
    payload: RetrievalQueryRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> RetrievalResponse:
    """Retrieve top-k semantically relevant chunks from the FAISS RAG index."""
    try:
        return await retrieval_service.query(payload)
    except RetrievalServiceError as exc:
        logger.exception("RAG retrieval endpoint failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
