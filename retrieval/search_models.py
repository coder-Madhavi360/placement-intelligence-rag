from typing import Any

from pydantic import BaseModel, Field

from core.schemas.common import Metadata


class RetrievalQueryRequest(BaseModel):
    """Request contract for semantic RAG retrieval."""

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=25)
    filters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)


class RetrievalScoreMetadata(BaseModel):
    raw_score: float
    normalized_score: float
    rank: int
    score_type: str = "cosine_similarity"


class RetrievedChunk(BaseModel):
    """Ranked chunk returned by the retrieval engine."""

    id: str
    chunk_id: str
    source_object_id: str
    content: str
    rank: int
    score: float
    semantic_type: str
    metadata: Metadata
    score_metadata: RetrievalScoreMetadata
    extra: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    query: str
    top_k: int
    match_count: int
    matches: list[RetrievedChunk]
    filters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    index: dict[str, Any] = Field(default_factory=dict)
