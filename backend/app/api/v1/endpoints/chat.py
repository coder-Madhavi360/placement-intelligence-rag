import logging

from fastapi import APIRouter, Depends

from app.api.deps import get_chat_service
from app.chat.chat_models import ChatRequest, ChatResponse
from app.core.exceptions import ApplicationError
from app.services.chat_service import ChatService
from app.services.retrieval_service import RetrievalServiceError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Run a session-aware conversational RAG query."""
    try:
        return await chat_service.answer(payload)
    except RetrievalServiceError as exc:
        logger.exception("Chat endpoint retrieval failed")
        raise ApplicationError(str(exc)) from exc
