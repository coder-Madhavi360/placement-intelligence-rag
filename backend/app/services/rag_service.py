from app.core.config import Settings
from app.retrieval.retriever import Retriever
from app.schemas.rag import RAGQueryRequest, RAGQueryResponse


class RAGService:
    """High-level RAG workflow service.

    In production, call a generation provider here with the retrieved contexts.
    The starter response is intentionally deterministic and dependency-free.
    """

    def __init__(self, settings: Settings, retriever: Retriever) -> None:
        self.settings = settings
        self.retriever = retriever

    async def answer(self, request: RAGQueryRequest) -> RAGQueryResponse:
        contexts = await self.retriever.retrieve(request)

        if contexts:
            answer = "Retrieved relevant context. Connect an LLM provider to generate a grounded answer."
        else:
            answer = "No relevant context found. Try ingesting documents before querying."

        return RAGQueryResponse(
            answer=answer,
            contexts=contexts,
            model=self.settings.llm_model,
        )
