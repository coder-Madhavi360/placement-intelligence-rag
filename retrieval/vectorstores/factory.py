from core.config import Settings
from retrieval.vectorstores.base import VectorStore
from retrieval.vectorstores.memory import InMemoryVectorStore


def build_vector_store(settings: Settings) -> VectorStore:
    """Build the configured vector store adapter.

    The non-memory branches are placeholders for production adapters. Keeping
    construction behind a factory prevents provider-specific SDKs from leaking
    through the rest of the application.
    """
    if settings.vector_store == "memory":
        return InMemoryVectorStore()

    raise NotImplementedError(
        f"Vector store '{settings.vector_store}' is configured but its adapter is not installed."
    )
