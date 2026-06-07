"""
retrieval/reranker.py
CrossEncoder reranking for precision after broad retrieval.
"""
import logging
import re
from sentence_transformers import CrossEncoder
from core.interfaces import BaseReranker, RetrievalResult
from retrieval.rewriter import extract_entities

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Uses a CrossEncoder model (ms-marco) to rerank retrieved chunks.
    Much more accurate than bi-encoder similarity alone.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading CrossEncoder: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, result: RetrievalResult) -> RetrievalResult:
        if not result.chunks:
            return result

        pairs = [(query, chunk.text) for chunk in result.chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(result.chunks, scores):
            chunk.score = float(score)

        result.chunks.sort(key=lambda c: c.score, reverse=True)
        result.chunks = self._promote_required_coverage(query, result.chunks)
        result.rerank_scores = [c.score for c in result.chunks]

        # Update quality based on rerank scores
        if result.rerank_scores:
            top_score = result.rerank_scores[0]
            result.retrieval_quality = min(1.0, max(0.0, top_score / 10.0))

        logger.debug(f"Reranked {len(result.chunks)} chunks | top={result.rerank_scores[0]:.3f}")
        return result

    def _promote_required_coverage(self, query: str, chunks: list) -> list:
        if not chunks:
            return chunks

        q = query.lower()
        entities = extract_entities(query)
        promoted = []

        if len(entities) >= 2:
            preferred_sections = self._preferred_sections(q)
            for entity in entities:
                candidates = [
                    c for c in chunks
                    if c.company.lower() == entity.lower()
                ]
                if not candidates:
                    continue
                promoted.append(
                    self._best_by_section_priority(
                        candidates,
                        preferred_sections,
                    )
                )

        elif (
            not entities
            and re.search(r"\b(highest|lowest)\b", q)
            and any(w in q for w in ["package", "salary", "lpa", "pay"])
        ):
            eligibility = [
                c for c in chunks
                if c.section == "eligibility"
                and isinstance(c.metadata.get("package"), (int, float))
            ]
            if eligibility:
                reverse = "highest" in q
                promoted.append(
                    max(eligibility, key=lambda c: c.metadata["package"])
                    if reverse
                    else min(eligibility, key=lambda c: c.metadata["package"])
                )

        if not promoted:
            return chunks

        seen = set()
        ordered = []
        for chunk in promoted + chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            ordered.append(chunk)
        return ordered

    def _best_by_section_priority(self, chunks: list, sections: list[str]):
        if not sections:
            return max(chunks, key=lambda c: c.score)
        for section in sections:
            matches = [c for c in chunks if c.section == section]
            if matches:
                return max(matches, key=lambda c: c.score)
        return max(chunks, key=lambda c: c.score)

    def _preferred_sections(self, query: str) -> list[str]:
        if any(w in query for w in ["bond", "bond-free", "no bond"]):
            return ["eligibility"]
        if any(w in query for w in ["package", "salary", "lpa", "pay"]):
            return ["eligibility", "trend", "conflict"]
        if any(w in query for w in ["eligibility", "eligible", "qualify", "cgpa", "backlog"]):
            return ["eligibility", "conflict"]
        return []
