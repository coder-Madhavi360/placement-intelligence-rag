import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.api.deps import get_chat_service
from feedback.chat.conversation_manager import ConversationManager
from feedback.chat.memory_store import InMemoryConversationStore
from core.main import app
from core.schemas.rag import RAGQueryResponse
from feedback.chat_service import ChatService


class FakeRAGService:
    """Deterministic RAG stand-in for testing chat memory wiring."""

    async def answer(self, request, *, conversation_history=None) -> RAGQueryResponse:
        history_count = len(conversation_history or [])
        return RAGQueryResponse(
            answer=f"Answer for: {request.query}. History messages: {history_count}.",
            sources=["test-source"],
            model="test-model",
        )


def build_test_chat_service() -> ChatService:
    store = InMemoryConversationStore()
    manager = ConversationManager(store=store, memory_window_conversations=5)
    return ChatService(conversation_manager=manager, rag_service=FakeRAGService())  # type: ignore[arg-type]


def main() -> int:
    test_chat_service = build_test_chat_service()
    app.dependency_overrides[get_chat_service] = lambda: test_chat_service

    with TestClient(app) as client:
        first = client.post("/api/v1/chat", json={"query": "Am I eligible for Amazon?"})
        if first.status_code != 200:
            print(first.status_code, first.text)
            return 1

        first_payload = first.json()
        session_id = first_payload["session_id"]

        second = client.post(
            "/api/v1/chat",
            json={"session_id": session_id, "query": "What about my backlog?"},
        )
        if second.status_code != 200:
            print(second.status_code, second.text)
            return 1

        second_payload = second.json()
        assert second_payload["session_id"] == session_id
        assert first_payload["conversation_length"] == 2
        assert second_payload["conversation_length"] == 4
        assert "History messages: 2" in second_payload["answer"]

    app.dependency_overrides.clear()

    print("Chat memory test passed.")
    print(f"session_id={session_id}")
    print(f"conversation_length={second_payload['conversation_length']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


