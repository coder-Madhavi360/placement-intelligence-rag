import logging

from starlette.concurrency import run_in_threadpool

from app.embeddings.embedder import EmbeddingError, SentenceTransformerEmbedder
from app.retrieval.ranker import RetrievalRanker
from app.retrieval.search_models import RetrievalQueryRequest, RetrievalResponse
from app.vectorstores.faiss_store import FAISSStoreError, FAISSVectorStore

logger = logging.getLogger(__name__)


class RetrievalServiceError(RuntimeError):
    """Raised when retrieval cannot be completed."""


class RetrievalService:
    """Application service for query embedding, FAISS search, and ranking."""

    def __init__(
        self,
        embedder: SentenceTransformerEmbedder,
        store: FAISSVectorStore,
        ranker: RetrievalRanker | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.ranker = ranker or RetrievalRanker()

    async def query(self, request: RetrievalQueryRequest) -> RetrievalResponse:
        logger.info(
            "Retrieval query received: top_k=%s min_score=%s filters=%s index_vectors=%s index_dimensions=%s index_model=%s query_model=%s",
            request.top_k,
            request.min_score,
            request.filters,
            self.store.count,
            self.store.dimensions,
            self.store.embedding_model,
            self.embedder.config.model_name,
        )

        try:
            query_vector = await run_in_threadpool(self.embedder.embed_query, request.query)
            logger.info(
                "Generated query embedding: dimensions=%s model=%s",
                len(query_vector),
                self.embedder.config.model_name,
            )
            vector_results = await run_in_threadpool(
                self.store.search,
                query_vector,
                top_k=request.top_k,
                filters=request.filters,
            )
            logger.info(
                "Retrieved vector candidates: count=%s scores=%s",
                len(vector_results),
                [round(result.score, 6) for result in vector_results],
            )
        except (EmbeddingError, FAISSStoreError) as exc:
            logger.exception("Retrieval query failed")
            raise RetrievalServiceError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected retrieval failure")
            raise RetrievalServiceError("Unexpected retrieval failure.") from exc

        matches = self.ranker.rank(vector_results, min_score=request.min_score)
        stats = self.store.stats().model_dump()

        logger.info(
            "Retrieval query completed: matches=%s vector_count=%s",
            len(matches),
            stats.get("vector_count"),
        )

        return RetrievalResponse(
            query=request.query,
            top_k=request.top_k,
            match_count=len(matches),
            matches=matches,
            filters=request.filters,
            index=stats,
        )
