"""
retrieval/retriever.py
Hybrid retriever: dense (FAISS) + sparse (BM25) with RRF fusion.
Includes retrieval quality scoring and overshadow risk estimation.
"""
import logging
import copy
import re
from collections import defaultdict
from core.interfaces import BaseRetriever, RetrievalResult, Chunk
from retrieval.rewriter import QueryRewriter, extract_entities
from ingestion.embedder import HybridEmbedder

logger = logging.getLogger(__name__)

RRF_K = 60  # RRF constant


class HybridRetriever(BaseRetriever):
    """
    Reciprocal Rank Fusion of dense + sparse results.
    Quality score = mean reranked similarity.
    Overshadow risk = f(chunk_count, token_count).
    """

    def __init__(self, embedder: HybridEmbedder, alpha: float = 0.6):
        self.embedder = embedder
        self.rewriter = QueryRewriter()
        self.alpha = alpha  # weight for dense (1-alpha for sparse)

    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        variants = self.rewriter.rewrite(query)
        all_dense, all_sparse = [], []

        for variant in variants:
            dense = self.embedder.search_dense(variant, top_k)
            sparse = self.embedder.search_sparse(variant, top_k)
            all_dense.extend(dense)
            all_sparse.extend(sparse)

        fused = self._rrf_fuse(all_dense, all_sparse, top_k)
        fused = self._ensure_intent_coverage(query, fused, top_k)
        print("="*50)
        print("FUSED CHUNKS:", len(fused))
        for i, chunk in enumerate(fused[:10]):
            print(i, chunk.metadata)
        quality = self._compute_quality(fused)
        overshadow = self._overshadow_risk(fused)

        return RetrievalResult(
            chunks=fused,
            query_used=variants[0],
            dense_scores=[c.score for c in fused],
            retrieval_quality=quality,
            overshadow_risk=overshadow,
        )

    def _rrf_fuse(
        self,
        dense: list[Chunk],
        sparse: list[Chunk],
        top_k: int,
    ) -> list[Chunk]:
        scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, Chunk] = {}

        # Dense RRF
        seen_dense: dict[str, int] = {}
        for rank, chunk in enumerate(dense):
            cid = chunk.chunk_id
            if cid not in seen_dense:
                seen_dense[cid] = rank
                scores[cid] += self.alpha / (RRF_K + rank + 1)
                chunk_map[cid] = chunk

        # Sparse RRF
        seen_sparse: dict[str, int] = {}
        for rank, chunk in enumerate(sparse):
            cid = chunk.chunk_id
            if cid not in seen_sparse:
                seen_sparse[cid] = rank
                scores[cid] += (1 - self.alpha) / (RRF_K + rank + 1)
                if cid not in chunk_map:
                    chunk_map[cid] = chunk

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        result = []
        for cid, score in ranked:
            chunk = chunk_map[cid]
            chunk.score = score
            result.append(chunk)
        return result

    def _ensure_intent_coverage(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int,
    ) -> list[Chunk]:
        q = query.lower()
        entities = extract_entities(query)
        needs_eligibility = any(
            w in q
            for w in [
                "eligibility", "eligible", "qualify", "cgpa",
                "backlog", "bond", "bond-free", "package",
                "salary", "lpa", "pay", "highest", "lowest",
            ]
        )
        if not needs_eligibility:
            return chunks

        additions = []
        if entities:
            for entity in entities:
                match = self._official_eligibility_for(entity)
                if match:
                    additions.append(match)
        elif re.search(r"\b(highest|lowest)\b", q) and any(
            w in q for w in ["package", "salary", "lpa", "pay"]
        ):
            additions.extend(self._official_eligibility_chunks())

        if not additions:
            return chunks

        seen = set()
        covered = []
        for chunk in additions + chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            covered.append(chunk)
            if len(covered) >= top_k:
                break
        return covered

    def _official_eligibility_for(self, company: str) -> Chunk | None:
        for chunk in self._official_eligibility_chunks():
            if chunk.company.lower() == company.lower():
                return chunk
        return None

    def _official_eligibility_chunks(self) -> list[Chunk]:
        return [
            copy.copy(chunk)
            for chunk in self.embedder.chunks
            if chunk.section == "eligibility" and chunk.source == "official"
        ]

    def _compute_quality(self, chunks: list[Chunk]) -> float:
        if not chunks:
            return 0.0
        return min(1.0, sum(c.score for c in chunks) / max(len(chunks), 1) * 10)

    def _overshadow_risk(self, chunks: list[Chunk]) -> float:
        """
        Estimates hallucination/overshadowing risk.
        Risk increases with chunk count and drops when top scores cluster tightly.
        Binary-search analogy: we probe this in the refiner.
        """
        if not chunks:
            return 0.0
        total_tokens = sum(len(c.text.split()) for c in chunks)
        token_risk = min(1.0, total_tokens / 3000)
        score_spread = chunks[0].score - chunks[-1].score if len(chunks) > 1 else 0
        diversity_risk = 1.0 - min(1.0, score_spread * 5)
        return round((token_risk * 0.6 + diversity_risk * 0.4), 3)
