from functools import lru_cache

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.local import LocalEmbeddingProvider
from app.retrieval.retriever import Retriever
from app.services.document_service import DocumentService
from app.services.rag_service import RAGService
from app.vectorstores.base import VectorStore
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


def get_retriever(
    vector_store: VectorStore = Depends(get_vector_store),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> Retriever:
    return Retriever(vector_store=vector_store, embedding_provider=embedding_provider)


def get_rag_service(
    settings: Settings = Depends(get_settings),
    retriever: Retriever = Depends(get_retriever),
) -> RAGService:
    return RAGService(settings=settings, retriever=retriever)


def get_document_service(
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    vector_store: VectorStore = Depends(get_vector_store),
) -> DocumentService:
    return DocumentService(embedding_provider=embedding_provider, vector_store=vector_store)
