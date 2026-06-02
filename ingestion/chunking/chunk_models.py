from typing import Any

from pydantic import BaseModel, Field, model_validator

from core.schemas.common import Metadata, Modality


class ChunkingConfig(BaseModel):
    """Runtime knobs for semantic chunk generation."""

    max_tokens: int = Field(default=280, ge=80, le=2000)
    overlap_tokens: int = Field(default=45, ge=0, le=500)
    min_chunk_tokens: int = Field(default=40, ge=1, le=500)
    preserve_tables: bool = True
    deduplicate: bool = True
    duplicate_threshold: float = Field(default=0.92, ge=0.5, le=1.0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        if self.min_chunk_tokens > self.max_tokens:
            raise ValueError("min_chunk_tokens must be less than or equal to max_tokens")
        return self


class SemanticChunk(BaseModel):
    """Reusable chunk object ready for embedding and vector indexing."""

    id: str
    source_object_id: str
    content: str
    modality: Modality = Modality.TEXT
    chunk_index: int
    token_count: int
    char_count: int
    metadata: Metadata
    semantic_type: str = "text"
    fingerprint: str


class DeduplicationStats(BaseModel):
    input_count: int = 0
    unique_count: int = 0
    duplicate_count: int = 0
    duplicate_ids: list[str] = Field(default_factory=list)


class ChunkStatistics(BaseModel):
    source_objects: int = 0
    raw_chunks: int = 0
    final_chunks: int = 0
    duplicate_chunks_removed: int = 0
    total_tokens: int = 0
    average_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    by_semantic_type: dict[str, int] = Field(default_factory=dict)


class ChunkingResult(BaseModel):
    chunks: list[SemanticChunk]
    statistics: ChunkStatistics
    deduplication: DeduplicationStats
    config: ChunkingConfig
    extra: dict[str, Any] = Field(default_factory=dict)

