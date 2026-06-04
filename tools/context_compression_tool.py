"""
tools/context_compression_tool.py

Context Compression Tool — removes irrelevant sentences from retrieved
chunks before passing to the LLM.

Why: retrieved chunks often contain sentences unrelated to the query.
For example, a chunk about Amazon's eligibility might contain sentences
about interview rounds. These sentences consume tokens and distract the LLM.

How: uses sentence-level TF-IDF similarity to score each sentence
against the query and keep only the top-scoring ones.

Result: same information, fewer tokens, less hallucination risk.
"""
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ContextCompressionTool:
    """
    Compresses each chunk by keeping only sentences
    most relevant to the query.
    """

    def __init__(self, keep_ratio: float = 0.6, min_sentences: int = 1):
        """
        Args:
            keep_ratio: fraction of sentences to keep per chunk (0.6 = 60%)
            min_sentences: always keep at least this many sentences
        """
        self.keep_ratio    = keep_ratio
        self.min_sentences = min_sentences

    def compress(self, query: str, chunks: list) -> list:
        """
        Returns new chunk objects with compressed text.
        Original chunks are not modified.

        Args:
            query: the user's question
            chunks: list of Chunk objects

        Returns:
            list of Chunk objects with compressed .text
        """
        import copy
        compressed = []
        total_before = 0
        total_after  = 0

        for chunk in chunks:
            sentences = self._split_sentences(chunk.text)
            total_before += len(chunk.text.split())

            if len(sentences) <= 2:
                # Too short to compress — keep as is
                compressed.append(chunk)
                total_after += len(chunk.text.split())
                continue

            keep_count = max(
                self.min_sentences,
                int(len(sentences) * self.keep_ratio)
            )

            try:
                scores   = self._score_sentences(query, sentences)
                top_idxs = sorted(
                    range(len(scores)),
                    key=lambda i: scores[i],
                    reverse=True
                )[:keep_count]
                # Keep in original order (preserve coherence)
                top_idxs = sorted(top_idxs)
                kept = [sentences[i] for i in top_idxs]
            except Exception:
                kept = sentences

            new_chunk      = copy.copy(chunk)
            new_chunk.text = " ".join(kept)
            compressed.append(new_chunk)
            total_after += len(new_chunk.text.split())

        reduction = round(1 - total_after / max(total_before, 1), 2)
        logger.info(
            f"Context compression: {total_before} → {total_after} tokens "
            f"({reduction:.0%} reduction)"
        )
        return compressed

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 15]

    def _score_sentences(
        self, query: str, sentences: list[str]
    ) -> list[float]:
        docs = [query] + sentences
        try:
            tfidf  = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(docs)
            scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
            return scores.tolist()
        except Exception:
            return [1.0] * len(sentences)