import logging
from difflib import SequenceMatcher

from ingestion.chunking.chunk_models import DeduplicationStats, SemanticChunk
from ingestion.chunking.utils import normalized_fingerprint_text

logger = logging.getLogger(__name__)


class ChunkDeduplicator:
    """Remove exact and near-duplicate chunks before embedding."""

    def __init__(self, similarity_threshold: float = 0.92) -> None:
        self.similarity_threshold = similarity_threshold

    def deduplicate(self, chunks: list[SemanticChunk]) -> tuple[list[SemanticChunk], DeduplicationStats]:
        unique_chunks: list[SemanticChunk] = []
        seen_fingerprints: set[str] = set()
        normalized_unique: list[str] = []
        duplicate_ids: list[str] = []

        for chunk in chunks:
            normalized = normalized_fingerprint_text(chunk.content)

            if chunk.fingerprint in seen_fingerprints or self._is_near_duplicate(normalized, normalized_unique):
                duplicate_ids.append(chunk.id)
                logger.debug("Dropping duplicate chunk %s", chunk.id)
                continue

            seen_fingerprints.add(chunk.fingerprint)
            normalized_unique.append(normalized)
            unique_chunks.append(chunk)

        stats = DeduplicationStats(
            input_count=len(chunks),
            unique_count=len(unique_chunks),
            duplicate_count=len(duplicate_ids),
            duplicate_ids=duplicate_ids,
        )

        logger.info(
            "Chunk deduplication complete: input=%s unique=%s duplicates=%s",
            stats.input_count,
            stats.unique_count,
            stats.duplicate_count,
        )
        return unique_chunks, stats

    def _is_near_duplicate(self, candidate: str, existing: list[str]) -> bool:
        if not candidate:
            return True

        for item in existing:
            length_ratio = min(len(candidate), len(item)) / max(len(candidate), len(item), 1)
            if length_ratio < 0.75:
                continue
            if SequenceMatcher(None, candidate, item).ratio() >= self.similarity_threshold:
                return True

        return False
