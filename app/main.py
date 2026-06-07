"""
RAG Chatbot API — Production-grade Retrieval-Augmented Generation
Author: Yasir Mehmood
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time

from app.api.routes import chat, documents, health
from app.core.config import settings
from app.core.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("🚀 Starting RAG Chatbot API...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Model: {settings.OPENAI_MODEL}")
    yield
    logger.info("🛑 Shutting down RAG Chatbot API...")


app = FastAPI(
    title="RAG Chatbot API",
    description="""
    ## Production-grade Retrieval-Augmented Generation API
    
    Built by **Yasir Mehmood** — Senior Full-Stack & AI Engineer
    
    ### Features
    - 📄 **Document ingestion** — PDF, TXT, DOCX support
    - 🔍 **Hybrid search** — Dense (FAISS) + Sparse (BM25) retrieval
    - 🤖 **LLM generation** — GPT-4o with streaming support
    - 📊 **Evaluation** — RAGAS metrics built-in
    - 🔁 **Multi-turn** — Conversation history with memory
    - ⚡ **Fast** — Sub-2s response for most queries
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.3f}s"
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# Routers
app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level="info",
    )
