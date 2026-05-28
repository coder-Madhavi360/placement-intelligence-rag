import logging

from fastapi import APIRouter, Depends, status

from app.api.deps import get_document_service
from app.schemas.documents import DocumentIngestRequest, DocumentIngestResponse
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("", response_model=DocumentIngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_documents(
    payload: DocumentIngestRequest,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentIngestResponse:
    """Ingest text/image references into the retrieval index."""
    logger.info("Ingestion API request accepted: documents=%s", len(payload.documents))
    result = await document_service.ingest(payload.documents)
    return DocumentIngestResponse(indexed_count=result.indexed_count)
