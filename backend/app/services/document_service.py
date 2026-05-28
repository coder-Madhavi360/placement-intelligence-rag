from app.embeddings.base import EmbeddingProvider
from app.schemas.documents import DocumentChunk, IngestResult
from app.vectorstores.base import VectorRecord, VectorStore


class DocumentService:
    """Handles document normalization, embedding, and indexing."""

    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def ingest(self, documents: list[DocumentChunk]) -> IngestResult:
        records: list[VectorRecord] = []

        for document in documents:
            if document.modality.value == "image":
                vector = await self.embedding_provider.embed_image(document.content)
            else:
                vector = await self.embedding_provider.embed_text(document.content)

            records.append(
                VectorRecord(
                    id=document.id,
                    vector=vector,
                    content=document.content,
                    modality=document.modality,
                    metadata=document.metadata,
                )
            )

        indexed_count = await self.vector_store.upsert(records)
        return IngestResult(indexed_count=indexed_count)
