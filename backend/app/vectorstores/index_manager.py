import logging
from pathlib import Path

from app.embeddings.embedder import SentenceTransformerEmbedder
from app.embeddings.vector_models import EmbeddedChunk, VectorIndexStats
from app.chunking.chunk_models import SemanticChunk
from app.vectorstores.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class VectorIndexManager:
    """Coordinates chunk embedding, FAISS indexing, and persistence."""

    def __init__(
        self,
        embedder: SentenceTransformerEmbedder,
        store: FAISSVectorStore | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store or FAISSVectorStore(dimensions=embedder.dimensions)

    def build_index(self, chunks: list[SemanticChunk]) -> tuple[FAISSVectorStore, list[EmbeddedChunk], VectorIndexStats]:
        logger.info("Building FAISS index from %s semantic chunk(s)", len(chunks))
        embedded_chunks = self.embedder.embed_chunks(chunks)
        stats = self.store.add(embedded_chunks)
        return self.store, embedded_chunks, stats

    def save(self, index_path: str | Path, metadata_path: str | Path) -> VectorIndexStats:
        return self.store.save(index_path=index_path, metadata_path=metadata_path)

    @classmethod
    def load(
        cls,
        embedder: SentenceTransformerEmbedder,
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> "VectorIndexManager":
        store = FAISSVectorStore.load(index_path=index_path, metadata_path=metadata_path)
        return cls(embedder=embedder, store=store)
