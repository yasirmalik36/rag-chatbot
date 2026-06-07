"""
Pydantic request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime


# ── Enums ─────────────────────────────────────────────────────────────────────

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


# ── Chat Schemas ──────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: Optional[str] = Field(None, description="Conversation session ID for memory")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Previous messages")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")
    temperature: Optional[float] = Field(0.2, ge=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "What are your refund policies?",
                "session_id": "user-123-session-456",
                "history": [],
                "top_k": 5,
            }
        }
    }


class SourceDocument(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    score: float = Field(..., description="Cosine similarity score 0–1")


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[SourceDocument]
    tokens_used: int
    latency_ms: int
    model: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Document Schemas ──────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    chunks_created: int
    message: str


class DocumentListItem(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    chunks: int
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    documents: List[DocumentListItem]
    total: int


class DeleteDocumentResponse(BaseModel):
    document_id: str
    message: str
    chunks_removed: int


# ── Health Schemas ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store: str
    llm_model: str
    total_documents: int
    total_chunks: int
