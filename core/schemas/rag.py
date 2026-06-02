from pydantic import BaseModel, Field

from core.schemas.common import Metadata, Modality


class RAGQueryRequest(BaseModel):
    """Client request for grounded placement-intelligence answers."""

    query: str = Field(min_length=1, max_length=4000)
    modality: Modality = Modality.TEXT
    top_k: int | None = Field(default=None, ge=1, le=25)
    filters: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RetrievedContext(BaseModel):
    """Context chunk used to ground the generated answer."""

    id: str
    content: str
    score: float
    modality: Modality
    metadata: Metadata = Field(default_factory=Metadata)


class RAGQueryResponse(BaseModel):
    """Answer payload returned by the RAG query endpoint."""

    answer: str
    contexts: list[RetrievedContext] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    model: str
    retrieval_time_ms: int = 0
    generation_time_ms: int = 0
