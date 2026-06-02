import logging
from datetime import UTC, datetime

from feedback.chat.chat_models import ChatRequest, ChatResponse
from feedback.chat.conversation_manager import ConversationManager
from core.schemas.rag import RAGQueryRequest
from generation.rag_service import RAGService

logger = logging.getLogger(__name__)


class ChatService:
    """Session-aware conversational wrapper around the RAG pipeline."""

    def __init__(
        self,
        conversation_manager: ConversationManager,
        rag_service: RAGService,
    ) -> None:
        self.conversation_manager = conversation_manager
        self.rag_service = rag_service

    async def answer(self, request: ChatRequest) -> ChatResponse:
        """Answer a chat query and persist the resulting conversation turn."""
        session_id = self.conversation_manager.resolve_session_id(request.session_id)
        history = self.conversation_manager.recent_history(session_id)

        logger.info(
            "Chat query received: session_id=%s history_messages=%s",
            session_id,
            len(history),
        )

        rag_response = await self.rag_service.answer(
            RAGQueryRequest(query=request.query),
            conversation_history=history,
        )
        conversation_length = self.conversation_manager.append_turn(
            session_id=session_id,
            query=request.query,
            answer=rag_response.answer,
        )

        logger.info(
            "Chat query completed: session_id=%s conversation_length=%s sources=%s",
            session_id,
            conversation_length,
            rag_response.sources,
        )

        return ChatResponse(
            session_id=session_id,
            answer=rag_response.answer,
            sources=rag_response.sources,
            conversation_length=conversation_length,
            timestamp=datetime.now(UTC),
        )

