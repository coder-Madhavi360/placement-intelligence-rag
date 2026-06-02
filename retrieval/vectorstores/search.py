import logging

from retrieval.embeddings.embedder import SentenceTransformerEmbedder
from retrieval.embeddings.vector_models import VectorSearchResult
from retrieval.vectorstores.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class SemanticSearchService:
    """Embed natural language queries and search a FAISS vector index."""

    def __init__(self, embedder: SentenceTransformerEmbedder, store: FAISSVectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, str | int | float | bool] | None = None,
    ) -> list[VectorSearchResult]:
        logger.info("Running semantic vector search: top_k=%s filters=%s", top_k, filters or {})
        query_vector = self.embedder.embed_query(query)
        return self.store.search(query_vector=query_vector, top_k=top_k, filters=filters)
