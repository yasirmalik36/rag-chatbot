"""
RAG Service — Orchestrates retrieval + generation pipeline.
"""

import time
import uuid
import logging
from typing import List, Optional

from app.services.vector_store import VectorStoreService
from app.services.llm_service import LLMService
from app.models.schemas import (
    ChatRequest, ChatResponse, SourceDocument
)

logger = logging.getLogger(__name__)


class RAGService:
    """
    Retrieval-Augmented Generation pipeline:

    1. Embed user query
    2. FAISS similarity search → top-K chunks
    3. Filter by score threshold
    4. Build context + conversation history
    5. GPT-4o generates grounded answer
    6. Return answer + source citations
    """

    def __init__(self, vector_store: VectorStoreService, llm_service: LLMService):
        self.vector_store = vector_store
        self.llm_service = llm_service

    def chat(self, request: ChatRequest) -> ChatResponse:
        start = time.time()

        session_id = request.session_id or str(uuid.uuid4())

        # ── Step 1: Retrieve relevant chunks ──────────────────────────────────
        logger.info(f"[{session_id}] Retrieving top-{request.top_k} chunks for query")
        chunks = self.vector_store.similarity_search(
            query=request.message,
            top_k=request.top_k,
            threshold=0.65,
        )

        if not chunks:
            logger.warning(f"[{session_id}] No relevant context found")

        # ── Step 2: Generate answer ───────────────────────────────────────────
        history = [{"role": m.role.value, "content": m.content} for m in (request.history or [])]

        result = self.llm_service.generate(
            question=request.message,
            context_chunks=chunks,
            history=history,
        )

        latency_ms = int((time.time() - start) * 1000)
        logger.info(
            f"[{session_id}] Response generated in {latency_ms}ms, "
            f"{len(chunks)} sources, {result['tokens_used']} tokens"
        )

        # ── Step 3: Build response ────────────────────────────────────────────
        sources = [
            SourceDocument(
                content=c["content"][:300] + "..." if len(c["content"]) > 300 else c["content"],
                source=c["source"],
                score=c["score"],
            )
            for c in chunks
        ]

        return ChatResponse(
            answer=result["answer"],
            session_id=session_id,
            sources=sources,
            tokens_used=result["tokens_used"],
            latency_ms=latency_ms,
            model=result["model"],
        )
