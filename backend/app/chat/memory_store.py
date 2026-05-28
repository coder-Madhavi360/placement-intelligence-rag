from collections import defaultdict
from threading import RLock

from app.chat.chat_models import ChatMessage


class InMemoryConversationStore:
    """Thread-safe in-memory conversation store.

    The interface is intentionally small so it can be replaced by Redis,
    Postgres, or another durable backend without changing chat services.
    """

    def __init__(self) -> None:
        self._messages: dict[str, list[ChatMessage]] = defaultdict(list)
        self._lock = RLock()

    def append(self, session_id: str, messages: list[ChatMessage]) -> int:
        """Append messages to a session and return total message count."""
        with self._lock:
            self._messages[session_id].extend(messages)
            return len(self._messages[session_id])

    def get_recent(self, session_id: str, *, limit: int) -> list[ChatMessage]:
        """Return the most recent messages for a session."""
        with self._lock:
            return list(self._messages.get(session_id, [])[-limit:])

    def count(self, session_id: str) -> int:
        """Return total messages stored for a session."""
        with self._lock:
            return len(self._messages.get(session_id, []))

    def clear(self, session_id: str) -> None:
        """Remove a session from memory."""
        with self._lock:
            self._messages.pop(session_id, None)
