from pydantic import BaseModel, Field

from app.schemas.common import Metadata, Modality


class RAGQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    modality: Modality = Modality.TEXT
    top_k: int = Field(default=5, ge=1, le=25)
    filters: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RetrievedContext(BaseModel):
    id: str
    content: str
    score: float
    modality: Modality
    metadata: Metadata = Field(default_factory=Metadata)


class RAGQueryResponse(BaseModel):
    answer: str
    contexts: list[RetrievedContext] = Field(default_factory=list)
    model: str
