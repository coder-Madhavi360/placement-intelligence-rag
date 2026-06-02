from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ChatRole(StrEnum):
    """Supported roles stored in conversation memory."""

    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Single timestamped message stored in a chat session."""

    role: ChatRole
    content: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatRequest(BaseModel):
    """Request payload for session-based conversational RAG."""

    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    """Response payload for conversational RAG."""

    session_id: str
    answer: str
    sources: list[str] = Field(default_factory=list)
    conversation_length: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

