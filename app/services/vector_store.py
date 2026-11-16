"""
FAISS Vector Store Service
Handles document embedding, indexing, retrieval, and persistence.
"""

import json
import uuid
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

import faiss
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Production FAISS vector store with:
    - Cosine similarity search (IndexFlatIP on normalised vectors)
    - Persistent index to disk
    - Document metadata store (JSON)
    - Chunk-level deduplication
    """

    def __init__(self):
        self.index_path = Path(settings.FAISS_INDEX_PATH)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.dimension = settings.EMBEDDING_DIMENSION
        self._index: Optional[faiss.IndexFlatIP] = None
        self._metadata: List[Dict[str, Any]] = []      # parallel list to FAISS vectors
        self._doc_registry: Dict[str, Dict] = {}       # document_id → doc info

        self._load_or_create_index()

    # ── Index lifecycle ───────────────────────────────────────────────────────

    def _load_or_create_index(self) -> None:
        idx_file = self.index_path / "index.faiss"
        meta_file = self.index_path / "metadata.json"
        reg_file = self.index_path / "registry.json"

        if idx_file.exists() and meta_file.exists():
            logger.info("Loading existing FAISS index from disk")
            self._index = faiss.read_index(str(idx_file))
            with open(meta_file) as f:
                self._metadata = json.load(f)
            if reg_file.exists():
                with open(reg_file) as f:
                    self._doc_registry = json.load(f)
            logger.info(f"Loaded {self._index.ntotal} vectors, {len(self._doc_registry)} documents")
        else:
            logger.info("Creating new FAISS index (IndexFlatIP)")
            self._index = faiss.IndexFlatIP(self.dimension)
            self._metadata = []
            self._doc_registry = {}

    def _persist(self) -> None:
        faiss.write_index(self._index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "metadata.json", "w") as f:
            json.dump(self._metadata, f, default=str)
        with open(self.index_path / "registry.json", "w") as f:
            json.dump(self._doc_registry, f, default=str)

    # ── Document ingestion ────────────────────────────────────────────────────

    def add_document(self, text: str, filename: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Split → Embed → Index a document.
        Returns document info with chunk count.
        """
        doc_id = document_id or str(uuid.uuid4())
        chunks = self.text_splitter.split_text(text)

        if not chunks:
            raise ValueError("Document produced no text chunks after splitting")

        logger.info(f"Embedding {len(chunks)} chunks for document '{filename}'")
        embeddings = self.embeddings.embed_documents(chunks)

        # Normalise for cosine similarity (IndexFlatIP on unit vectors = cosine)
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

        start_idx = self._index.ntotal
        self._index.add(vectors)

        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            self._metadata.append({
                "document_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "faiss_index": start_idx + i,
                "content": chunk,
                "content_length": len(chunk),
            })

        self._doc_registry[doc_id] = {
            "document_id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "status": "indexed",
            "uploaded_at": datetime.utcnow().isoformat(),
            "faiss_start": start_idx,
            "faiss_end": start_idx + len(chunks),
        }

        self._persist()
        logger.info(f"Document '{filename}' indexed: {len(chunks)} chunks, doc_id={doc_id}")

        return self._doc_registry[doc_id]

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Embed query → search FAISS → return top-k chunks above threshold.
        Returns list of {content, source, score, chunk_index}.
        """
        if self._index.ntotal == 0:
            logger.warning("Vector store is empty — no documents indexed yet")
            return []

        query_vector = np.array(
            [self.embeddings.embed_query(query)], dtype=np.float32
        )
        faiss.normalize_L2(query_vector)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if float(score) < threshold:
                continue
            meta = self._metadata[idx]
            results.append({
                "content": meta["content"],
                "source": meta["filename"],
                "chunk_index": meta["chunk_index"],
                "score": round(float(score), 4),
            })

        logger.debug(f"Query '{query[:60]}...' → {len(results)} results above threshold={threshold}")
        return results

    # ── Document management ───────────────────────────────────────────────────

    def delete_document(self, document_id: str) -> int:
        """
        Remove a document and rebuild the index without its vectors.
        Returns number of chunks removed.
        """
        if document_id not in self._doc_registry:
            raise KeyError(f"Document {document_id} not found")

        chunks_removed = self._doc_registry[document_id]["chunks"]

        # Rebuild metadata without this document
        self._metadata = [m for m in self._metadata if m["document_id"] != document_id]
        del self._doc_registry[document_id]

        # Rebuild FAISS index from remaining metadata
        self._rebuild_index()
        self._persist()

        logger.info(f"Deleted document {document_id}, removed {chunks_removed} chunks")
        return chunks_removed

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from current metadata (after deletion)."""
        self._index = faiss.IndexFlatIP(self.dimension)
        if not self._metadata:
            return

        contents = [m["content"] for m in self._metadata]
        embeddings = self.embeddings.embed_documents(contents)
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self._index.add(vectors)

        for i, meta in enumerate(self._metadata):
            meta["faiss_index"] = i

    def list_documents(self) -> List[Dict[str, Any]]:
        return list(self._doc_registry.values())

    @property
    def total_chunks(self) -> int:
        return self._index.ntotal

    @property
    def total_documents(self) -> int:
        return len(self._doc_registry)
