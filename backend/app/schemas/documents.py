from pydantic import BaseModel, Field

from app.schemas.common import Metadata, Modality


class DocumentChunk(BaseModel):
    id: str
    content: str = Field(min_length=1)
    modality: Modality = Modality.TEXT
    metadata: Metadata = Field(default_factory=Metadata)


class DocumentIngestRequest(BaseModel):
    documents: list[DocumentChunk] = Field(min_length=1)


class DocumentIngestResponse(BaseModel):
    indexed_count: int


class IngestResult(BaseModel):
    indexed_count: int
