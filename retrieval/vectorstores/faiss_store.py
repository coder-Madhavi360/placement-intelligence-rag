import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from retrieval.embeddings.vector_models import EmbeddedChunk, VectorIndexStats, VectorSearchResult

logger = logging.getLogger(__name__)


class FAISSStoreError(RuntimeError):
    """Raised when FAISS vector store operations fail."""


class FAISSVectorStore:
    """FAISS-backed vector store with metadata sidecar storage."""

    def __init__(self, dimensions: int = 384, index_name: str = "placement_faiss_index") -> None:
        self.dimensions = dimensions
        self.index_name = index_name
        self.index = faiss.IndexFlatIP(dimensions)
        self._records: list[EmbeddedChunk] = []

    @property
    def count(self) -> int:
        return int(self.index.ntotal)

    @property
    def embedding_model(self) -> str | None:
        if not self._records:
            return None
        return self._records[0].model_name

    def add(self, records: list[EmbeddedChunk]) -> VectorIndexStats:
        if not records:
            return self.stats()

        vectors = self._records_to_matrix(records)
        self.index.add(vectors)
        self._records.extend(records)

        logger.info("Added %s vectors to FAISS index; total=%s", len(records), self.count)
        return self.stats()

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        filters: dict[str, str | int | float | bool] | None = None,
    ) -> list[VectorSearchResult]:
        if self.count == 0:
            logger.warning("FAISS search requested against an empty index")
            return []
        if top_k < 1:
            raise FAISSStoreError("top_k must be at least 1.")

        query = np.asarray([query_vector], dtype=np.float32)
        if query.shape[1] != self.dimensions:
            raise FAISSStoreError(f"Query dimension mismatch: expected {self.dimensions}, received {query.shape[1]}")

        faiss.normalize_L2(query)
        search_k = min(max(top_k * 8, top_k), self.count)
        logger.info(
            "Executing FAISS search: requested_top_k=%s expanded_search_k=%s index_vectors=%s dimensions=%s",
            top_k,
            search_k,
            self.count,
            self.dimensions,
        )
        scores, indices = self.index.search(query, search_k)
        logger.info(
            "FAISS raw search scores: %s",
            [round(float(score), 6) for score in scores[0][:top_k]],
        )

        results: list[VectorSearchResult] = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            record = self._records[int(index)]
            if filters and not self._matches_filters(record, filters):
                continue

            results.append(
                VectorSearchResult(
                    id=record.id,
                    content=record.content,
                    score=float(score),
                    rank=len(results) + 1,
                    metadata=record.metadata,
                    semantic_type=record.chunk.semantic_type,
                    chunk_id=record.chunk.id,
                    source_object_id=record.chunk.source_object_id,
                )
            )

            if len(results) >= top_k:
                break

        logger.info(
            "FAISS search returned %s result(s) for top_k=%s; result_scores=%s",
            len(results),
            top_k,
            [round(result.score, 6) for result in results],
        )
        return results

    def save(self, index_path: str | Path, metadata_path: str | Path) -> VectorIndexStats:
        index_path = Path(index_path).expanduser().resolve()
        metadata_path = Path(metadata_path).expanduser().resolve()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            faiss.write_index(self.index, str(index_path))
            payload = {
                "dimensions": self.dimensions,
                "index_name": self.index_name,
                "records": [record.model_dump(mode="json") for record in self._records],
            }
            metadata_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception as exc:
            logger.exception("Failed to save FAISS index")
            raise FAISSStoreError("Failed to save FAISS index and metadata.") from exc

        logger.info("Saved FAISS index to %s and metadata to %s", index_path, metadata_path)
        stats = self.stats()
        stats.saved_index_path = str(index_path)
        stats.saved_metadata_path = str(metadata_path)
        return stats

    @classmethod
    def load(cls, index_path: str | Path, metadata_path: str | Path) -> "FAISSVectorStore":
        index_path = Path(index_path).expanduser().resolve()
        metadata_path = Path(metadata_path).expanduser().resolve()

        logger.info("Loading FAISS index from %s", index_path)
        logger.info("Loading FAISS metadata from %s", metadata_path)

        if not index_path.exists() or not metadata_path.exists():
            raise FAISSStoreError(
                f"Both FAISS index and metadata files are required. "
                f"index_exists={index_path.exists()} metadata_exists={metadata_path.exists()}"
            )

        try:
            logger.info(
                "FAISS index file size=%s bytes; metadata file size=%s bytes",
                index_path.stat().st_size,
                metadata_path.stat().st_size,
            )
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            store = cls(dimensions=int(payload["dimensions"]), index_name=payload.get("index_name", "placement_faiss_index"))
            store.index = faiss.read_index(str(index_path))
            store._records = [EmbeddedChunk.model_validate(record) for record in payload.get("records", [])]
        except Exception as exc:
            logger.exception("Failed to load FAISS index")
            raise FAISSStoreError("Failed to load FAISS index and metadata.") from exc

        if store.count != len(store._records):
            raise FAISSStoreError("FAISS index count does not match metadata record count.")
        if store.index.d != store.dimensions:
            raise FAISSStoreError(
                f"FAISS index dimension mismatch: index={store.index.d} metadata={store.dimensions}"
            )
        for record in store._records:
            if record.dimensions != store.dimensions or len(record.embedding) != store.dimensions:
                raise FAISSStoreError(
                    f"Metadata vector dimension mismatch for chunk {record.id}: "
                    f"record_dimensions={record.dimensions} vector_length={len(record.embedding)} "
                    f"index_dimensions={store.dimensions}"
                )

        logger.info(
            "Loaded FAISS index successfully: vectors=%s metadata_records=%s dimensions=%s embedding_model=%s",
            store.count,
            len(store._records),
            store.dimensions,
            store.embedding_model,
        )
        return store

    def stats(self) -> VectorIndexStats:
        return VectorIndexStats(
            index_name=self.index_name,
            vector_count=self.count,
            dimensions=self.dimensions,
            embedding_model=self.embedding_model,
            metadata_count=len(self._records),
        )

    def _records_to_matrix(self, records: list[EmbeddedChunk]) -> np.ndarray:
        matrix = np.asarray([record.embedding for record in records], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimensions:
            raise FAISSStoreError(
                f"Vector dimension mismatch: expected {self.dimensions}, received shape {matrix.shape}"
            )
        faiss.normalize_L2(matrix)
        return np.ascontiguousarray(matrix)

    def _matches_filters(self, record: EmbeddedChunk, filters: dict[str, Any]) -> bool:
        metadata = record.metadata.model_dump()
        metadata.update(record.metadata.extra)
        metadata["semantic_type"] = record.chunk.semantic_type
        return all(metadata.get(key) == value for key, value in filters.items())

