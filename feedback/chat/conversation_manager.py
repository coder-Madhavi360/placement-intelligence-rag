from uuid import uuid4

from feedback.chat.chat_models import ChatMessage, ChatRole
from feedback.chat.memory_store import InMemoryConversationStore


class ConversationManager:
    """Coordinates session IDs and bounded conversation memory access."""

    def __init__(
        self,
        store: InMemoryConversationStore,
        *,
        memory_window_conversations: int = 5,
    ) -> None:
        self.store = store
        self.memory_window_conversations = memory_window_conversations

    @property
    def memory_window_messages(self) -> int:
        return self.memory_window_conversations * 2

    def resolve_session_id(self, session_id: str | None) -> str:
        """Use the client session ID or generate a new opaque ID."""
        return session_id or str(uuid4())

    def recent_history(self, session_id: str) -> list[ChatMessage]:
        """Return the last configured number of user/assistant messages."""
        return self.store.get_recent(session_id, limit=self.memory_window_messages)

    def append_turn(self, session_id: str, query: str, answer: str) -> int:
        """Persist one user/assistant turn and return conversation length."""
        return self.store.append(
            session_id,
            [
                ChatMessage(role=ChatRole.USER, content=query),
                ChatMessage(role=ChatRole.ASSISTANT, content=answer),
            ],
        )

