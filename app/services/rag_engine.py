"""
RAG Engine — Core retrieval-augmented generation pipeline.

Pipeline:
  1. Document ingestion → chunking → embedding → FAISS index
  2. Query → embed → FAISS search → rerank → prompt → LLM → response

Author: Yasir Mehmood
"""

import logging
import time
import uuid
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from app.core.config import settings
from app.models.schemas import SourceDocument

logger = logging.getLogger(__name__)


# ─── System Prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful, accurate AI assistant. You answer questions \
based ONLY on the provided context documents.

Rules:
- Answer based solely on the provided context
- If the context doesn't contain enough information, say "I don't have enough \
information in the provided documents to answer this accurately"
- Never make up facts or hallucinate information
- Cite which document/source your answer comes from when possible
- Be concise but complete
- If asked about something outside the documents, acknowledge it politely

Context documents:
{context}
"""


class RAGEngine:
    """
    Production RAG engine with:
    - FAISS vector store for semantic search
    - Conversation memory per session
    - Source attribution
    - Streaming support
    """

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            openai_api_key=settings.OPENAI_API_KEY,
            streaming=False,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.vector_store: Optional[FAISS] = None
        self.sessions: Dict[str, List] = {}   # session_id → chat history
        self.document_registry: List[str] = []  # Track ingested doc names

        # Load existing index if available
        self._load_index()

    # ─── Document Ingestion ───────────────────────────────────────────────

    def ingest_document(self, file_path: str, filename: str) -> int:
        """
        Load, chunk, embed and store a document.
        Returns number of chunks created.
        """
        logger.info(f"Ingesting document: {filename}")
        start = time.time()

        # Load document based on type
        docs = self._load_document(file_path, filename)
        if not docs:
            raise ValueError(f"Could not load document: {filename}")

        # Split into chunks
        chunks = self.text_splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks from {filename}")

        # Add chunk IDs and source metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{filename}_{i}"
            chunk.metadata["source"] = filename

        # Add to vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

        # Track document
        if filename not in self.document_registry:
            self.document_registry.append(filename)

        # Persist index
        self._save_index()

        elapsed = time.time() - start
        logger.info(f"Ingested {filename} in {elapsed:.2f}s — {len(chunks)} chunks")
        return len(chunks)

    def ingest_text(self, text: str, source_name: str) -> int:
        """Ingest raw text string directly."""
        doc = Document(page_content=text, metadata={"source": source_name})
        chunks = self.text_splitter.split_documents([doc])

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{source_name}_{i}"

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

        if source_name not in self.document_registry:
            self.document_registry.append(source_name)

        self._save_index()
        return len(chunks)

    # ─── Retrieval ────────────────────────────────────────────────────────

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Semantic similarity search against FAISS index.
        Returns list of (document, score) tuples, score 0-1.
        """
        if self.vector_store is None:
            return []

        k = top_k or settings.TOP_K_RESULTS

        # FAISS returns (doc, distance) — convert distance to similarity
        results = self.vector_store.similarity_search_with_score(query, k=k)

        # Filter by threshold and normalize scores
        filtered = []
        for doc, distance in results:
            # FAISS L2 distance → approximate similarity (lower distance = higher similarity)
            similarity = max(0.0, 1.0 - (distance / 2.0))
            if similarity >= settings.SIMILARITY_THRESHOLD:
                filtered.append((doc, similarity))

        logger.info(f"Retrieved {len(filtered)}/{len(results)} chunks above threshold")
        return filtered

    # ─── Generation ──────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve → build prompt → generate.
        Returns answer, sources, token usage, latency.
        """
        start_time = time.time()
        session_id = session_id or str(uuid.uuid4())

        logger.info(f"Chat | session={session_id} | query='{message[:80]}...'")

        # 1. Retrieve relevant chunks
        retrieved = self.retrieve(message, top_k=top_k)

        # 2. Build context string
        context_parts = []
        source_docs = []
        for i, (doc, score) in enumerate(retrieved):
            context_parts.append(
                f"[Document {i+1} | Source: {doc.metadata.get('source', 'unknown')} "
                f"| Score: {score:.2f}]\n{doc.page_content}"
            )
            source_docs.append(
                SourceDocument(
                    content=doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page"),
                    score=round(score, 4),
                    chunk_id=doc.metadata.get("chunk_id", str(uuid.uuid4())),
                )
            )

        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

        # 3. Build messages with history
        messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

        # Add conversation history (last N turns)
        history = self.sessions.get(session_id, [])
        for hist_msg in history[-(settings.MAX_HISTORY_TURNS * 2):]:
            messages.append(hist_msg)

        messages.append(HumanMessage(content=message))

        # 4. Generate response
        response = self.llm.invoke(messages)
        answer = response.content
        tokens = response.usage_metadata.get("total_tokens", 0) if hasattr(response, "usage_metadata") else 0

        # 5. Update session history
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(HumanMessage(content=message))
        self.sessions[session_id].append(AIMessage(content=answer))

        # Trim history to max turns
        max_msgs = settings.MAX_HISTORY_TURNS * 2
        if len(self.sessions[session_id]) > max_msgs:
            self.sessions[session_id] = self.sessions[session_id][-max_msgs:]

        latency = (time.time() - start_time) * 1000
        logger.info(f"Generated response in {latency:.0f}ms | tokens={tokens}")

        return {
            "answer": answer,
            "session_id": session_id,
            "sources": source_docs,
            "tokens_used": tokens,
            "latency_ms": round(latency, 2),
            "model": settings.OPENAI_MODEL,
        }

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    # ─── Vector Store Management ─────────────────────────────────────────

    def get_index_size(self) -> int:
        """Return total number of chunks in vector store."""
        if self.vector_store is None:
            return 0
        return self.vector_store.index.ntotal

    def delete_document(self, filename: str) -> bool:
        """
        Remove a document from the registry.
        Note: FAISS doesn't support selective deletion natively —
        in production use Qdrant/Weaviate for this.
        """
        if filename in self.document_registry:
            self.document_registry.remove(filename)
            logger.warning(
                f"Removed {filename} from registry. "
                f"Note: chunks remain in FAISS index until full rebuild."
            )
            return True
        return False

    def _load_document(self, file_path: str, filename: str) -> List[Document]:
        """Load document based on file extension."""
        ext = Path(filename).suffix.lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
                return loader.load()
            elif ext in [".txt", ".md"]:
                loader = TextLoader(file_path, encoding="utf-8")
                return loader.load()
            else:
                # Fallback: read as text
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                return [Document(page_content=content, metadata={"source": filename})]
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return []

    def _save_index(self):
        """Persist FAISS index and metadata to disk."""
        try:
            os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)
            self.vector_store.save_local(settings.FAISS_INDEX_PATH)
            # Save document registry
            with open(f"{settings.FAISS_INDEX_PATH}/registry.pkl", "wb") as f:
                pickle.dump(self.document_registry, f)
            logger.debug("FAISS index saved to disk")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def _load_index(self):
        """Load existing FAISS index from disk if available."""
        index_file = Path(settings.FAISS_INDEX_PATH) / "index.faiss"
        if index_file.exists():
            try:
                self.vector_store = FAISS.load_local(
                    settings.FAISS_INDEX_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                # Load registry
                registry_file = Path(settings.FAISS_INDEX_PATH) / "registry.pkl"
                if registry_file.exists():
                    with open(registry_file, "rb") as f:
                        self.document_registry = pickle.load(f)
                logger.info(
                    f"Loaded FAISS index: {self.get_index_size()} chunks, "
                    f"{len(self.document_registry)} documents"
                )
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}. Starting fresh.")
                self.vector_store = None


# Singleton instance
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
