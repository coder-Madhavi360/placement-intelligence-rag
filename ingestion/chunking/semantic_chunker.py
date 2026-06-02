import logging
from collections.abc import Iterable

from ingestion.chunking.chunk_models import (
    ChunkingConfig,
    ChunkingResult,
    ChunkStatistics,
    DeduplicationStats,
    SemanticChunk,
)
from ingestion.chunking.deduplicator import ChunkDeduplicator
from ingestion.chunking.utils import (
    chunk_id,
    estimate_tokens,
    fingerprint_text,
    infer_semantic_type,
    is_semantic_boundary,
    merge_text,
    split_paragraphs,
    split_sentences,
)
from ingestion.metadata import ExtractedDocumentObject
from core.schemas.common import Metadata, Modality

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Paragraph-aware chunker optimized for placement intelligence PDFs."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        deduplicator: ChunkDeduplicator | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self.deduplicator = deduplicator or ChunkDeduplicator(self.config.duplicate_threshold)

    def chunk_documents(self, documents: Iterable[ExtractedDocumentObject]) -> ChunkingResult:
        document_list = list(documents)
        logger.info("Starting semantic chunking for %s extracted object(s)", len(document_list))

        raw_chunks: list[SemanticChunk] = []
        for document in document_list:
            raw_chunks.extend(self.chunk_document(document))

        if self.config.deduplicate:
            final_chunks, dedup_stats = self.deduplicator.deduplicate(raw_chunks)
        else:
            final_chunks = raw_chunks
            dedup_stats = DeduplicationStats(
                input_count=len(raw_chunks),
                unique_count=len(raw_chunks),
                duplicate_count=0,
            )

        statistics = self._build_statistics(
            source_objects=len(document_list),
            raw_chunks=raw_chunks,
            final_chunks=final_chunks,
            deduplication=dedup_stats,
        )

        logger.info(
            "Semantic chunking complete: raw=%s final=%s duplicates_removed=%s",
            statistics.raw_chunks,
            statistics.final_chunks,
            statistics.duplicate_chunks_removed,
        )

        return ChunkingResult(
            chunks=final_chunks,
            statistics=statistics,
            deduplication=dedup_stats,
            config=self.config,
        )

    def chunk_document(self, document: ExtractedDocumentObject) -> list[SemanticChunk]:
        semantic_type = infer_semantic_type(document.content, document.object_type)
        if self.config.preserve_tables and semantic_type == "table":
            return [self._make_chunk(document, document.content, 1, semantic_type, overlap=False)]

        units = self._semantic_units(document.content)
        grouped_units = self._group_units(units)
        chunks: list[SemanticChunk] = []

        for index, group in enumerate(grouped_units, start=1):
            content = merge_text(group)
            if not content:
                continue
            chunks.append(self._make_chunk(document, content, index, semantic_type, overlap=index > 1))

        logger.debug("Chunked source object %s into %s chunk(s)", document.id, len(chunks))
        return chunks

    def _semantic_units(self, text: str) -> list[str]:
        units: list[str] = []

        for paragraph in split_paragraphs(text):
            paragraph_tokens = estimate_tokens(paragraph)
            if paragraph_tokens <= self.config.max_tokens:
                units.append(paragraph)
                continue

            units.extend(self._split_long_paragraph(paragraph))

        return units

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        sentences = split_sentences(paragraph)
        units: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)
            if current and current_tokens + sentence_tokens > self.config.max_tokens:
                units.append(" ".join(current).strip())
                current = []
                current_tokens = 0

            if sentence_tokens > self.config.max_tokens:
                units.extend(self._split_by_token_window(sentence))
                continue

            current.append(sentence)
            current_tokens += sentence_tokens

        if current:
            units.append(" ".join(current).strip())

        return units

    def _split_by_token_window(self, text: str) -> list[str]:
        words = text.split()
        step = max(self.config.max_tokens - self.config.overlap_tokens, 1)
        windows = []

        for start in range(0, len(words), step):
            window = words[start : start + self.config.max_tokens]
            if window:
                windows.append(" ".join(window))
            if start + self.config.max_tokens >= len(words):
                break

        return windows

    def _group_units(self, units: list[str]) -> list[list[str]]:
        groups: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = estimate_tokens(unit)
            starts_new_topic = bool(current and is_semantic_boundary(unit))
            exceeds_limit = current and current_tokens + unit_tokens > self.config.max_tokens

            if starts_new_topic or exceeds_limit:
                groups.append(current)
                current = self._overlap_units(current)
                current_tokens = sum(estimate_tokens(item) for item in current)

            current.append(unit)
            current_tokens += unit_tokens

        if current:
            groups.append(current)

        return [group for group in groups if sum(estimate_tokens(item) for item in group) >= self.config.min_chunk_tokens]

    def _overlap_units(self, units: list[str]) -> list[str]:
        if self.config.overlap_tokens <= 0:
            return []

        overlap: list[str] = []
        token_total = 0

        for unit in reversed(units):
            unit_tokens = estimate_tokens(unit)
            if overlap and token_total + unit_tokens > self.config.overlap_tokens:
                break
            overlap.insert(0, unit)
            token_total += unit_tokens

        return overlap

    def _make_chunk(
        self,
        document: ExtractedDocumentObject,
        content: str,
        chunk_index: int,
        semantic_type: str,
        *,
        overlap: bool,
    ) -> SemanticChunk:
        token_count = estimate_tokens(content)
        fingerprint = fingerprint_text(content)
        metadata = self._chunk_metadata(document.metadata, chunk_index, semantic_type, overlap)

        return SemanticChunk(
            id=chunk_id(document.id, chunk_index, fingerprint),
            source_object_id=document.id,
            content=content,
            modality=document.modality or Modality.TEXT,
            chunk_index=chunk_index,
            token_count=token_count,
            char_count=len(content),
            metadata=metadata,
            semantic_type=semantic_type,
            fingerprint=fingerprint,
        )

    def _chunk_metadata(
        self,
        source_metadata: Metadata,
        chunk_index: int,
        semantic_type: str,
        overlap: bool,
    ) -> Metadata:
        extra = dict(source_metadata.extra)
        extra.update(
            {
                "chunk_index": chunk_index,
                "semantic_type": semantic_type,
                "has_overlap_context": overlap,
            }
        )

        tags = sorted(set([*source_metadata.tags, "chunk", semantic_type]))
        return Metadata(source=source_metadata.source, page=source_metadata.page, tags=tags, extra=extra)

    def _build_statistics(
        self,
        source_objects: int,
        raw_chunks: list[SemanticChunk],
        final_chunks: list[SemanticChunk],
        deduplication: DeduplicationStats,
    ) -> ChunkStatistics:
        token_counts = [chunk.token_count for chunk in final_chunks]
        by_semantic_type: dict[str, int] = {}

        for chunk in final_chunks:
            by_semantic_type[chunk.semantic_type] = by_semantic_type.get(chunk.semantic_type, 0) + 1

        total_tokens = sum(token_counts)
        return ChunkStatistics(
            source_objects=source_objects,
            raw_chunks=len(raw_chunks),
            final_chunks=len(final_chunks),
            duplicate_chunks_removed=deduplication.duplicate_count,
            total_tokens=total_tokens,
            average_tokens=round(total_tokens / len(token_counts), 2) if token_counts else 0.0,
            min_tokens=min(token_counts) if token_counts else 0,
            max_tokens=max(token_counts) if token_counts else 0,
            by_semantic_type=by_semantic_type,
        )
