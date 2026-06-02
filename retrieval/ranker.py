import logging

from retrieval.embeddings.vector_models import VectorSearchResult
from retrieval.search_models import RetrievedChunk, RetrievalScoreMetadata

logger = logging.getLogger(__name__)


class RetrievalRanker:
    """Convert vector search results into stable ranked response objects."""

    def rank(
        self,
        results: list[VectorSearchResult],
        *,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        filtered = [result for result in results if min_score is None or result.score >= min_score]
        filtered.sort(key=lambda result: result.score, reverse=True)

        ranked: list[RetrievedChunk] = []
        for index, result in enumerate(filtered, start=1):
            normalized_score = self._normalize_score(result.score)
            metadata = result.metadata.model_copy(deep=True)
            metadata.extra.update(
                {
                    "retrieval_rank": index,
                    "retrieval_score": result.score,
                    "retrieval_normalized_score": normalized_score,
                    "retrieval_score_type": "cosine_similarity",
                }
            )

            ranked.append(
                RetrievedChunk(
                    id=result.id,
                    chunk_id=result.chunk_id,
                    source_object_id=result.source_object_id,
                    content=result.content,
                    rank=index,
                    score=result.score,
                    semantic_type=result.semantic_type,
                    metadata=metadata,
                    score_metadata=RetrievalScoreMetadata(
                        raw_score=result.score,
                        normalized_score=normalized_score,
                        rank=index,
                    ),
                    extra=result.extra,
                )
            )

        logger.info("Ranked %s retrieval result(s)", len(ranked))
        return ranked

    def _normalize_score(self, score: float) -> float:
        # FAISS IndexFlatIP over L2-normalized embeddings yields cosine scores.
        return round(max(0.0, min(1.0, (score + 1.0) / 2.0)), 6)
