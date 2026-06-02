from pydantic import BaseModel, Field

from core.schemas.common import Metadata, Modality


class DocumentChunk(BaseModel):
    """Validated chunk accepted by the lightweight ingestion endpoint."""

    id: str
    content: str = Field(min_length=1)
    modality: Modality = Modality.TEXT
    metadata: Metadata = Field(default_factory=Metadata)


class DocumentIngestRequest(BaseModel):
    """Batch ingestion request for pre-chunked document content."""

    documents: list[DocumentChunk] = Field(min_length=1)


class DocumentIngestResponse(BaseModel):
    """Summary of records indexed by the ingestion endpoint."""

    indexed_count: int


class IngestResult(BaseModel):
    """Internal ingestion service result."""

    indexed_count: int
