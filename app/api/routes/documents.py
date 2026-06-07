"""Document management endpoints — upload, list, delete."""

import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from app.models.schemas import (
    DocumentUploadResponse, DocumentListResponse,
    DeleteDocumentResponse, DocumentStatus, DocumentListItem
)
from app.services.vector_store import VectorStoreService
from app.services.document_processor import extract_text
from app.core.dependencies import get_vector_store
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
MAX_FILE_SIZE_MB = 10


@router.post("/upload", response_model=DocumentUploadResponse, summary="Upload a document")
async def upload_document(
    file: UploadFile = File(...),
    vs: VectorStoreService = Depends(get_vector_store),
):
    """
    Upload a document (PDF, TXT, MD, DOCX) to the knowledge base.
    The file is automatically chunked and indexed into the FAISS vector store.
    """
    import os
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large. Max size: {MAX_FILE_SIZE_MB}MB")

    try:
        text = extract_text(file_bytes, file.filename)
        if not text.strip():
            raise HTTPException(400, "Document appears to be empty or unreadable")

        doc_id = str(uuid.uuid4())
        doc_info = vs.add_document(text=text, filename=file.filename, document_id=doc_id)

        return DocumentUploadResponse(
            document_id=doc_id,
            filename=file.filename,
            status=DocumentStatus.INDEXED,
            chunks_created=doc_info["chunks"],
            message=f"Document indexed successfully with {doc_info['chunks']} chunks",
        )

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to process document: {str(e)}")


@router.get("/", response_model=DocumentListResponse, summary="List all documents")
async def list_documents(vs: VectorStoreService = Depends(get_vector_store)):
    docs = vs.list_documents()
    items = [
        DocumentListItem(
            document_id=d["document_id"],
            filename=d["filename"],
            status=DocumentStatus(d["status"]),
            chunks=d["chunks"],
            uploaded_at=datetime.fromisoformat(d["uploaded_at"]),
        )
        for d in docs
    ]
    return DocumentListResponse(documents=items, total=len(items))


@router.delete("/{document_id}", response_model=DeleteDocumentResponse, summary="Delete a document")
async def delete_document(
    document_id: str,
    vs: VectorStoreService = Depends(get_vector_store),
):
    try:
        chunks_removed = vs.delete_document(document_id)
        return DeleteDocumentResponse(
            document_id=document_id,
            message="Document deleted and index rebuilt",
            chunks_removed=chunks_removed,
        )
    except KeyError:
        raise HTTPException(404, f"Document {document_id} not found")
    except Exception as e:
        raise HTTPException(500, str(e))
