from fastapi import APIRouter, Depends

from app.api.deps import get_rag_service
from app.schemas.rag import RAGQueryRequest, RAGQueryResponse
from app.services.rag_service import RAGService

router = APIRouter()


@router.post("", response_model=RAGQueryResponse)
async def query_rag(
    payload: RAGQueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> RAGQueryResponse:
    """Run a multimodal RAG query against the configured retriever."""
    return await rag_service.answer(payload)
