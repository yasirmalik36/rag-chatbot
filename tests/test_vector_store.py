"""
Unit tests for the FAISS vector store service.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestVectorStoreService:

    def test_add_document_creates_chunks(self):
        """Documents should be split into chunks and indexed."""
        with patch("app.services.vector_store.faiss") as mock_faiss, \
             patch("app.services.vector_store.OpenAIEmbeddings") as mock_emb:

            mock_index = MagicMock()
            mock_index.ntotal = 0
            mock_faiss.IndexFlatIP.return_value = mock_index
            mock_faiss.read_index.side_effect = FileNotFoundError
            mock_emb.return_value.embed_documents.return_value = [
                [0.1] * 1536 for _ in range(3)
            ]

            from app.services.vector_store import VectorStoreService
            with patch.object(VectorStoreService, '_persist'):
                vs = VectorStoreService.__new__(VectorStoreService)
                vs._metadata = []
                vs._doc_registry = {}
                vs._index = mock_index
                vs.dimension = 1536
                vs.embeddings = mock_emb.return_value
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                vs.text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

                with patch.object(vs, '_persist'):
                    result = vs.add_document("Hello world. " * 50, "test.txt", "doc-001")

                assert result["document_id"] == "doc-001"
                assert result["chunks"] > 0

    def test_similarity_search_empty_store_returns_empty(self):
        """Empty vector store should return empty list."""
        with patch("app.services.vector_store.faiss") as mock_faiss, \
             patch("app.services.vector_store.OpenAIEmbeddings"):

            mock_index = MagicMock()
            mock_index.ntotal = 0
            mock_faiss.IndexFlatIP.return_value = mock_index

            from app.services.vector_store import VectorStoreService
            vs = VectorStoreService.__new__(VectorStoreService)
            vs._index = mock_index
            vs._metadata = []
            vs._doc_registry = {}
            vs.embeddings = MagicMock()

            results = vs.similarity_search("test query")
            assert results == []
