"""
Unit tests for RAG Engine.
Run: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestRAGEngine:
    """Tests for core RAG pipeline."""

    def test_ingest_text_creates_chunks(self):
        """Ingesting text should produce at least one chunk."""
        with patch("app.services.rag_engine.OpenAIEmbeddings"), \
             patch("app.services.rag_engine.ChatOpenAI"), \
             patch("app.services.rag_engine.FAISS"):

            from app.services.rag_engine import RAGEngine
            engine = RAGEngine()

            mock_store = MagicMock()
            mock_store.index.ntotal = 3
            engine.vector_store = mock_store

            with patch.object(engine.text_splitter, "split_documents") as mock_split, \
                 patch.object(engine, "_save_index"):
                mock_split.return_value = [MagicMock(), MagicMock(), MagicMock()]
                count = engine.ingest_text("Some sample text content here.", "test.txt")
                assert count == 3

    def test_retrieve_returns_empty_when_no_index(self):
        """Retrieve should return empty list when no index exists."""
        with patch("app.services.rag_engine.OpenAIEmbeddings"), \
             patch("app.services.rag_engine.ChatOpenAI"), \
             patch("app.services.rag_engine.FAISS"):

            from app.services.rag_engine import RAGEngine
            engine = RAGEngine()
            engine.vector_store = None
            result = engine.retrieve("test query")
            assert result == []

    def test_session_history_accumulates(self):
        """Session history should grow with each chat turn."""
        with patch("app.services.rag_engine.OpenAIEmbeddings"), \
             patch("app.services.rag_engine.ChatOpenAI"), \
             patch("app.services.rag_engine.FAISS"):

            from app.services.rag_engine import RAGEngine
            engine = RAGEngine()

            mock_response = MagicMock()
            mock_response.content = "Test answer"
            mock_response.usage_metadata = {"total_tokens": 100}
            engine.llm.invoke = MagicMock(return_value=mock_response)

            with patch.object(engine, "retrieve", return_value=[]):
                engine.chat("Question 1", session_id="test-session")
                engine.chat("Question 2", session_id="test-session")

            assert len(engine.sessions.get("test-session", [])) == 4  # 2 turns × 2 messages

    def test_clear_session(self):
        """Clearing a session should remove its history."""
        with patch("app.services.rag_engine.OpenAIEmbeddings"), \
             patch("app.services.rag_engine.ChatOpenAI"), \
             patch("app.services.rag_engine.FAISS"):

            from app.services.rag_engine import RAGEngine
            engine = RAGEngine()
            engine.sessions["test-session"] = [MagicMock(), MagicMock()]

            result = engine.clear_session("test-session")
            assert result is True
            assert "test-session" not in engine.sessions

    def test_clear_nonexistent_session_returns_false(self):
        """Clearing non-existent session should return False."""
        with patch("app.services.rag_engine.OpenAIEmbeddings"), \
             patch("app.services.rag_engine.ChatOpenAI"), \
             patch("app.services.rag_engine.FAISS"):

            from app.services.rag_engine import RAGEngine
            engine = RAGEngine()
            result = engine.clear_session("ghost-session")
            assert result is False

    def test_get_index_size_zero_when_no_store(self):
        """Index size should be 0 when no vector store."""
        with patch("app.services.rag_engine.OpenAIEmbeddings"), \
             patch("app.services.rag_engine.ChatOpenAI"), \
             patch("app.services.rag_engine.FAISS"):

            from app.services.rag_engine import RAGEngine
            engine = RAGEngine()
            engine.vector_store = None
            assert engine.get_index_size() == 0


class TestSchemas:
    """Tests for request/response schemas."""

    def test_chat_request_valid(self):
        from app.models.schemas import ChatRequest
        req = ChatRequest(message="Hello", session_id="abc-123")
        assert req.message == "Hello"
        assert req.session_id == "abc-123"
        assert req.stream is False

    def test_chat_request_rejects_empty_message(self):
        from app.models.schemas import ChatRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_source_document_schema(self):
        from app.models.schemas import SourceDocument
        doc = SourceDocument(
            content="Sample content",
            source="test.pdf",
            score=0.92,
            chunk_id="chunk_001",
        )
        assert doc.score == 0.92
        assert doc.source == "test.pdf"
