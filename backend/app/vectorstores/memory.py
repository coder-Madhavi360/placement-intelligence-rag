import math

from app.vectorstores.base import VectorMatch, VectorRecord, VectorSearchQuery, VectorStore


class InMemoryVectorStore(VectorStore):
    """Process-local vector store for development and tests."""

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    async def upsert(self, records: list[VectorRecord]) -> int:
        for record in records:
            self._records[record.id] = record
        return len(records)

    async def search(self, query: VectorSearchQuery) -> list[VectorMatch]:
        matches = [
            VectorMatch(
                id=record.id,
                content=record.content,
                score=_cosine_similarity(query.vector, record.vector),
                modality=record.modality,
                metadata=record.metadata,
            )
            for record in self._records.values()
            if _matches_filters(record, query.filters)
        ]
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[: query.top_k]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return dot / (left_norm * right_norm)


def _matches_filters(record: VectorRecord, filters: dict[str, str | int | float | bool]) -> bool:
    if not filters:
        return True

    metadata = record.metadata.model_dump()
    metadata.update(record.metadata.extra)
    return all(metadata.get(key) == value for key, value in filters.items())
