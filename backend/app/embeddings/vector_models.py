from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.chunking.chunk_models import SemanticChunk
from app.schemas.common import Metadata, Modality


class EmbeddingConfig(BaseModel):
    """Configuration for sentence-transformer embedding generation."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int = 384
    batch_size: int = Field(default=32, ge=1, le=512)
    normalize_embeddings: bool = True
    cache_enabled: bool = True
    cache_path: Path | None = None
    device: str | None = None

    @model_validator(mode="after")
    def validate_cache_path(self) -> "EmbeddingConfig":
        if self.cache_path is not None:
            self.cache_path = self.cache_path.expanduser().resolve()
        return self


class EmbeddedChunk(BaseModel):
    """A semantic chunk plus its vector representation."""

    id: str
    chunk: SemanticChunk
    embedding: list[float]
    model_name: str
    dimensions: int
    metadata: Metadata

    @property
    def content(self) -> str:
        return self.chunk.content

    @property
    def modality(self) -> Modality:
        return self.chunk.modality


class EmbeddingBatchStats(BaseModel):
    input_chunks: int = 0
    embedded_chunks: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    model_name: str
    dimensions: int


class VectorIndexStats(BaseModel):
    index_name: str = "placement_faiss_index"
    vector_count: int = 0
    dimensions: int = 384
    embedding_model: str | None = None
    index_type: str = "IndexFlatIP"
    metric: str = "cosine_similarity"
    metadata_count: int = 0
    saved_index_path: str | None = None
    saved_metadata_path: str | None = None


class VectorSearchResult(BaseModel):
    id: str
    content: str
    score: float
    rank: int
    metadata: Metadata
    semantic_type: str
    chunk_id: str
    source_object_id: str
    extra: dict[str, Any] = Field(default_factory=dict)
