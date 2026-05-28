from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from app.schemas.common import Metadata, Modality


class VectorRecord(BaseModel):
    id: str
    vector: list[float]
    content: str
    modality: Modality = Modality.TEXT
    metadata: Metadata = Field(default_factory=Metadata)


class VectorSearchQuery(BaseModel):
    vector: list[float]
    top_k: int = 5
    filters: dict[str, str | int | float | bool] = Field(default_factory=dict)


class VectorMatch(BaseModel):
    id: str
    content: str
    score: float
    modality: Modality = Modality.TEXT
    metadata: Metadata = Field(default_factory=Metadata)


class VectorStore(ABC):
    """Interface implemented by every vector database adapter."""

    @abstractmethod
    async def upsert(self, records: list[VectorRecord]) -> int:
        raise NotImplementedError

    @abstractmethod
    async def search(self, query: VectorSearchQuery) -> list[VectorMatch]:
        raise NotImplementedError
