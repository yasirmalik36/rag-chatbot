"""
Unit tests for the RAG pipeline.
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from app.services.rag_service import RAGService
from app.models.schemas import ChatRequest, MessageRole, ChatMessage


@pytest.fixture
def mock_vector_store():
    vs = MagicMock()
    vs.similarity_search.return_value = [
        {"content": "Refunds are processed within 5-7 business days.", "source": "policy.pdf", "score": 0.92},
        {"content": "Contact support@example.com for refund requests.", "source": "policy.pdf", "score": 0.85},
    ]
    vs.total_chunks = 100
    vs.total_documents = 5
    return vs


@pytest.fixture
def mock_llm_service():
    llm = MagicMock()
    llm.generate.return_value = {
        "answer": "Refunds are processed within 5-7 business days.",
        "tokens_used": 150,
        "model": "gpt-4o",
    }
    return llm


@pytest.fixture
def rag_service(mock_vector_store, mock_llm_service):
    return RAGService(vector_store=mock_vector_store, llm_service=mock_llm_service)


class TestRAGService:

    def test_chat_returns_answer(self, rag_service):
        req = ChatRequest(message="What is your refund policy?")
        response = rag_service.chat(req)
        assert response.answer == "Refunds are processed within 5-7 business days."
        assert len(response.sources) == 2

    def test_chat_assigns_session_id(self, rag_service):
        req = ChatRequest(message="Hello")
        response = rag_service.chat(req)
        assert response.session_id is not None
        assert len(response.session_id) > 0

    def test_chat_uses_provided_session_id(self, rag_service):
        req = ChatRequest(message="Hello", session_id="my-session-123")
        response = rag_service.chat(req)
        assert response.session_id == "my-session-123"

    def test_chat_includes_latency(self, rag_service):
        req = ChatRequest(message="Test question")
        response = rag_service.chat(req)
        assert response.latency_ms >= 0

    def test_chat_with_history(self, rag_service, mock_llm_service):
        history = [
            ChatMessage(role=MessageRole.USER, content="Hi"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hello! How can I help?"),
        ]
        req = ChatRequest(message="What is your refund policy?", history=history)
        response = rag_service.chat(req)
        # Verify history was passed to LLM
        call_args = mock_llm_service.generate.call_args
        assert len(call_args.kwargs.get("history", [])) == 2

    def test_chat_no_context_still_returns(self, rag_service, mock_vector_store):
        mock_vector_store.similarity_search.return_value = []
        req = ChatRequest(message="Something obscure")
        response = rag_service.chat(req)
        assert response.answer is not None
        assert response.sources == []

    def test_source_content_truncated(self, rag_service, mock_vector_store):
        long_content = "A" * 500
        mock_vector_store.similarity_search.return_value = [
            {"content": long_content, "source": "doc.pdf", "score": 0.9}
        ]
        req = ChatRequest(message="Test")
        response = rag_service.chat(req)
        assert len(response.sources[0].content) <= 303  # 300 + "..."
