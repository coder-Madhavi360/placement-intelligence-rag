"""
generation/refiner.py
Context pruning — removes low-quality chunks after reranking.
"""
import logging
from core.interfaces import RetrievalResult
from safety.overshadow_limiter import OvershadowLimiter

logger = logging.getLogger(__name__)


class ContextRefiner:
    """Wraps OvershadowLimiter with additional score-based pruning."""

    def __init__(self, limiter: OvershadowLimiter):
        self.limiter = limiter

    def refine(self, result: RetrievalResult, max_tokens: int = 2000) -> RetrievalResult:
        original_chunks = list(result.chunks)

        # Score-based pruning: remove chunks below 10% of a positive top score.
        # CrossEncoder scores can be negative logits, where relative positive
        # thresholding would incorrectly remove every retrieved chunk.
        if result.chunks:
            top = result.chunks[0].score
            if top > 0:
                threshold = top * 0.10
                kept = [c for c in result.chunks if c.score >= threshold]
                if kept:
                    result.chunks = kept
                else:
                    result.chunks = [result.chunks[0]]
                    logger.warning(
                        "Refiner score pruning would remove all chunks; "
                        "retained top chunk as safeguard"
                    )
            else:
                logger.info(
                    "Refiner skipped relative score pruning for non-positive "
                    "top rerank score %.4f",
                    top,
                )

        refined = self.limiter.refine(result, max_tokens)

        if original_chunks and not refined.chunks:
            refined.chunks = [original_chunks[0]]
            logger.warning(
                "Refiner safeguard restored top chunk after empty refinement"
            )

        return refined
