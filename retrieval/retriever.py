from retrieval.embeddings.base import EmbeddingProvider
from core.schemas.rag import RAGQueryRequest, RetrievedContext
from retrieval.vectorstores.base import VectorSearchQuery, VectorStore


class Retriever:
    """Coordinates embedding generation and vector search."""

    def __init__(self, vector_store: VectorStore, embedding_provider: EmbeddingProvider) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    async def retrieve(self, request: RAGQueryRequest) -> list[RetrievedContext]:
        if request.modality.value == "image":
            embedding = await self.embedding_provider.embed_image(request.query)
        else:
            embedding = await self.embedding_provider.embed_text(request.query)

        matches = await self.vector_store.search(
            VectorSearchQuery(
                vector=embedding,
                top_k=request.top_k,
                filters=request.filters,
            )
        )

        return [
            RetrievedContext(
                id=match.id,
                content=match.content,
                score=match.score,
                modality=match.modality,
                metadata=match.metadata,
            )
            for match in matches
        ]

