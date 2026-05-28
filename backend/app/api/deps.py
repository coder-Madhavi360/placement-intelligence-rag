from functools import lru_cache

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.core.exceptions import ServiceUnavailableError
from app.chunking.chunk_models import ChunkingConfig
from app.embeddings.base import EmbeddingProvider
from app.embeddings.embedder import SentenceTransformerEmbedder
from app.embeddings.local import LocalEmbeddingProvider
from app.embeddings.vector_models import EmbeddingConfig
from app.retrieval.retriever import Retriever
from app.services.document_service import DocumentService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.vectorstores.base import VectorStore
from app.vectorstores.faiss_store import FAISSStoreError, FAISSVectorStore
from app.vectorstores.factory import build_vector_store


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Create the embedding provider once per process."""
    settings = get_settings()
    return LocalEmbeddingProvider(dimensions=settings.embedding_dimensions)


@lru_cache
def get_vector_store() -> VectorStore:
    """Create the configured vector store once per process."""
    settings = get_settings()
    return build_vector_store(settings)


@lru_cache
def get_sentence_transformer_embedder() -> SentenceTransformerEmbedder:
    """Create the production semantic embedder once per process."""
    settings = get_settings()
    return SentenceTransformerEmbedder(
        EmbeddingConfig(
            model_name=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            batch_size=settings.embedding_batch_size,
            cache_enabled=True,
            cache_path=settings.resolve_path(settings.embedding_cache_path),
        )
    )


def get_chunking_config(settings: Settings = Depends(get_settings)) -> ChunkingConfig:
    """Build chunking configuration from environment settings."""
    return ChunkingConfig(
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        min_chunk_tokens=settings.chunk_min_tokens,
    )


@lru_cache
def get_faiss_store() -> FAISSVectorStore:
    """Load the persisted FAISS index and metadata sidecar."""
    settings = get_settings()
    index_path = settings.resolve_path(settings.faiss_index_path)
    metadata_path = settings.resolve_path(settings.faiss_metadata_path)
    try:
        store = FAISSVectorStore.load(index_path=index_path, metadata_path=metadata_path)
    except FAISSStoreError as exc:
        raise ServiceUnavailableError(
            "FAISS retrieval index is unavailable. Build it with "
            "scripts/test_embeddings.py or your indexing job before querying. "
            f"index_path={index_path} metadata_path={metadata_path}"
        ) from exc

    if store.dimensions != settings.embedding_dimensions:
        raise ServiceUnavailableError(
            "FAISS index dimensions do not match embedding configuration. "
            f"index_dimensions={store.dimensions} configured_dimensions={settings.embedding_dimensions}"
        )
    if store.embedding_model and store.embedding_model != settings.embedding_model:
        raise ServiceUnavailableError(
            "FAISS index embedding model does not match query embedding model. "
            f"index_model={store.embedding_model} configured_model={settings.embedding_model}"
        )

    return store


def get_retriever(
    vector_store: VectorStore = Depends(get_vector_store),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> Retriever:
    return Retriever(vector_store=vector_store, embedding_provider=embedding_provider)


def get_document_service(
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    vector_store: VectorStore = Depends(get_vector_store),
) -> DocumentService:
    return DocumentService(embedding_provider=embedding_provider, vector_store=vector_store)


def get_retrieval_service(
    embedder: SentenceTransformerEmbedder = Depends(get_sentence_transformer_embedder),
    store: FAISSVectorStore = Depends(get_faiss_store),
) -> RetrievalService:
    return RetrievalService(embedder=embedder, store=store)


def get_llm_service(settings: Settings = Depends(get_settings)) -> LLMService:
    return LLMService(settings=settings)


def get_rag_service(
    settings: Settings = Depends(get_settings),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> RAGService:
    return RAGService(settings=settings, retrieval_service=retrieval_service, llm_service=llm_service)
