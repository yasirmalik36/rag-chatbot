"""Health check endpoint."""

from fastapi import APIRouter, Depends
from app.models.schemas import HealthResponse
from app.core.dependencies import get_vector_store
from app.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(vs=Depends(get_vector_store)):
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        vector_store="faiss",
        llm_model=settings.OPENAI_MODEL,
        total_documents=vs.total_documents,
        total_chunks=vs.total_chunks,
    )
