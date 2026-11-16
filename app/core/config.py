"""
Application configuration — loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # App
    APP_NAME: str = "RAG Chatbot API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # OpenAI
    OPENAI_API_KEY: str = "sk-placeholder"
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.2

    # FAISS Vector Store
    FAISS_INDEX_PATH: str = "data/faiss_index"
    EMBEDDING_DIMENSION: int = 1536

    # RAG
    RETRIEVAL_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    MAX_CONTEXT_LENGTH: int = 4000
    MAX_HISTORY_TURNS: int = 6  # conversation turns kept per session

    # Redis (optional caching)
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL_SECONDS: int = 3600

    # Security
    API_KEY_HEADER: str = "X-API-Key"
    API_KEYS: List[str] = []

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:4200"]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
