"""Chat endpoints — RAG question answering."""

import logging
from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_service import RAGService
from app.core.dependencies import get_rag_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=ChatResponse, summary="Ask a question")
async def chat(
    request: ChatRequest,
    rag: RAGService = Depends(get_rag_service),
):
    """
    Send a message and get a context-grounded answer with source citations.

    - **message**: Your question
    - **session_id**: Optional — pass the same ID to maintain conversation context
    - **history**: Previous messages for multi-turn conversations
    - **top_k**: Number of document chunks to retrieve (default: 5)
    """
    try:
        response = rag.chat(request)
        return response
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
