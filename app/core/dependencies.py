"""FastAPI dependency injection."""

from functools import lru_cache
from app.services.vector_store import VectorStoreService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreService:
    return VectorStoreService()


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService()


def get_rag_service() -> RAGService:
    return RAGService(
        vector_store=get_vector_store(),
        llm_service=get_llm_service(),
    )
