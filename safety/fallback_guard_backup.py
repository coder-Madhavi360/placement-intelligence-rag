"""
safety/fallback_guard.py
Detects out-of-corpus queries and triggers graceful fallback.
"""
import logging
import math
from core.interfaces import BaseFallbackGuard, Chunk

logger = logging.getLogger(__name__)


MIN_RELEVANCE_SCORE = 0.05  # normalized reranker score below this = likely out-of-corpus


class FallbackGuard(BaseFallbackGuard):
    

    def is_out_of_corpus(self, query: str, chunks: list[Chunk]) -> bool:

        # Check retrieval score threshold
        if not chunks:
            logger.info("Fallback triggered: no chunks retrieved")
            return True

        top_score = max(c.score for c in chunks)
        normalized_score = self._sigmoid(top_score)

        print("TOP SCORE =", top_score)
        print("NORMALIZED SCORE =", normalized_score)
        print("MIN SCORE =", MIN_RELEVANCE_SCORE)
        if normalized_score < MIN_RELEVANCE_SCORE:
            logger.info(
                f"Fallback triggered: normalized top score "
                f"{normalized_score:.4f} < threshold"
            )
            return True

        return False

    def _sigmoid(self, score: float) -> float:
        if score >= 0:
            z = math.exp(-score)
            return 1 / (1 + z)
        z = math.exp(score)
        return z / (1 + z)
