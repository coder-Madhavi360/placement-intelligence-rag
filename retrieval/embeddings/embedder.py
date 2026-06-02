import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from ingestion.chunking.chunk_models import SemanticChunk
from retrieval.embeddings.vector_models import EmbeddedChunk, EmbeddingBatchStats, EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


class SentenceTransformerEmbedder:
    """Batch embedder backed by sentence-transformers.

    Uses all-MiniLM-L6-v2 by default, which produces 384-dimensional vectors
    and is a strong baseline for fast semantic retrieval over placement text.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._model: SentenceTransformer | None = None
        self._cache: dict[str, list[float]] = {}
        self._last_stats = EmbeddingBatchStats(
            model_name=self.config.model_name,
            dimensions=self.config.dimensions,
        )
        self._load_cache()

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    @property
    def last_stats(self) -> EmbeddingBatchStats:
        return self._last_stats

    def embed_chunks(self, chunks: list[SemanticChunk]) -> list[EmbeddedChunk]:
        """Generate embeddings for chunks, using cache when available."""
        if not chunks:
            self._last_stats = EmbeddingBatchStats(model_name=self.config.model_name, dimensions=self.dimensions)
            return []

        logger.info("Generating embeddings for %s chunk(s)", len(chunks))
        texts_to_embed: list[str] = []
        cache_keys_to_embed: list[str] = []
        embeddings_by_key: dict[str, list[float]] = {}
        cache_hits = 0

        for chunk in chunks:
            cache_key = self._cache_key(chunk)
            cached = self._cache.get(cache_key) if self.config.cache_enabled else None
            if cached is not None:
                embeddings_by_key[cache_key] = cached
                cache_hits += 1
                continue

            texts_to_embed.append(chunk.content)
            cache_keys_to_embed.append(cache_key)

        if texts_to_embed:
            generated = self._encode_texts(texts_to_embed)
            for cache_key, vector in zip(cache_keys_to_embed, generated, strict=True):
                vector_list = vector.astype(np.float32).tolist()
                embeddings_by_key[cache_key] = vector_list
                if self.config.cache_enabled:
                    self._cache[cache_key] = vector_list

        embedded = [
            EmbeddedChunk(
                id=chunk.id,
                chunk=chunk,
                embedding=embeddings_by_key[self._cache_key(chunk)],
                model_name=self.config.model_name,
                dimensions=self.dimensions,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]

        self._last_stats = EmbeddingBatchStats(
            input_chunks=len(chunks),
            embedded_chunks=len(embedded),
            cache_hits=cache_hits,
            cache_misses=len(texts_to_embed),
            model_name=self.config.model_name,
            dimensions=self.dimensions,
        )
        self._save_cache()

        logger.info(
            "Embedding complete: chunks=%s cache_hits=%s cache_misses=%s",
            self._last_stats.embedded_chunks,
            self._last_stats.cache_hits,
            self._last_stats.cache_misses,
        )
        return embedded

    def embed_query(self, query: str) -> list[float]:
        """Embed a user query for vector search."""
        if not query.strip():
            raise EmbeddingError("Cannot embed an empty query.")
        return self._encode_texts([query])[0].astype(np.float32).tolist()

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        try:
            model = self._get_model()
            vectors = model.encode(
                texts,
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.exception("Embedding generation failed")
            raise EmbeddingError("Failed to generate sentence-transformer embeddings.") from exc

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dimensions:
            raise EmbeddingError(
                f"Embedding dimension mismatch: expected {self.dimensions}, received {vectors.shape[1]}"
            )
        return vectors

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading sentence-transformer model %s", self.config.model_name)
            try:
                self._model = SentenceTransformer(self.config.model_name, device=self.config.device)
            except TypeError:
                self._model = SentenceTransformer(self.config.model_name)
            except Exception as exc:
                logger.exception("Failed to load sentence-transformer model %s", self.config.model_name)
                raise EmbeddingError(f"Failed to load embedding model: {self.config.model_name}") from exc
        return self._model

    def _cache_key(self, chunk: SemanticChunk) -> str:
        return f"{self.config.model_name}:{chunk.fingerprint}"

    def _load_cache(self) -> None:
        if not self.config.cache_enabled or self.config.cache_path is None:
            return
        if not self.config.cache_path.exists():
            return

        try:
            self._cache = json.loads(self.config.cache_path.read_text(encoding="utf-8"))
            logger.info("Loaded embedding cache with %s entries", len(self._cache))
        except Exception:
            logger.exception("Failed to load embedding cache from %s", self.config.cache_path)
            self._cache = {}

    def _save_cache(self) -> None:
        if not self.config.cache_enabled or self.config.cache_path is None:
            return

        try:
            self.config.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        except Exception:
            logger.exception("Failed to save embedding cache to %s", self.config.cache_path)

