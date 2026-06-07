"""
RAG Chatbot — Production Entry Point
Author: Yasir Mehmood
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager

from app.api.routes import chat, documents, health
from app.core.config import settings
from app.core.logging import setup_logging

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    from app.services.vector_store import VectorStoreService
    from app.core.dependencies import get_vector_store
    print(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    yield
    print("🛑 Shutting down gracefully...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade RAG Chatbot API using LangChain, FAISS, and GPT-4",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(chat.router,      prefix="/api/v1/chat",      tags=["Chat"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
