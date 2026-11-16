# 🤖 RAG Chatbot — Production AI Customer Support

> **Retrieval-Augmented Generation** chatbot built with **FastAPI**, **LangChain**, **FAISS**, and **GPT-4o**. Upload your documents, ask questions, get grounded answers with source citations.
>
> *A reference implementation of the pattern behind a production RAG chatbot I built at Markitech.AI, which cut support tickets by 35% and raised CSAT by 30%. This repository is an independent rebuild — no client code or data.*

[![Python](https://img.shields.io/badge/Python_3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![FAISS](https://img.shields.io/badge/FAISS-00599C?style=for-the-badge)](https://faiss.ai)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![CI](https://github.com/yasirmalik36/rag-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/yasirmalik36/rag-chatbot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 🔍 **Semantic Search** | FAISS IndexFlatIP with L2-normalised cosine similarity |
| 📄 **Multi-format Ingestion** | PDF, DOCX, TXT, Markdown — auto chunked + indexed |
| 🧠 **GPT-4o Generation** | Context-grounded answers with hallucination resistance |
| 💬 **Multi-turn Conversations** | Session-based conversation history (last 6 turns) |
| 📚 **Source Citations** | Every answer includes scored source chunks |
| 🗑️ **Document Management** | Upload, list, delete documents via REST API |
| ⚡ **Fast Retrieval** | In-memory FAISS search over normalised embeddings |
| 🐳 **Docker Ready** | Single command deployment with docker-compose |
| 🧪 **Tested** | Unit tests for RAG pipeline, vector store, API routes |
| 📊 **Observable** | Latency, token usage, session ID in every response |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Client Application                        │
│              (Angular / React / Postman / curl)                  │
└─────────────────────────┬────────────────────────────────────────┘
                          │ HTTP POST /api/v1/chat
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  /health    │  │  /chat       │  │  /documents            │  │
│  │             │  │  POST        │  │  POST /upload          │  │
│  │  status     │  │              │  │  GET  /               │  │
│  │  doc count  │  │  ChatRequest │  │  DELETE /{id}          │  │
│  └─────────────┘  └──────┬───────┘  └───────────┬────────────┘  │
└─────────────────────────-│──────────────────────-│───────────────┘
                           │                       │
                           ▼                       ▼
┌──────────────────────────────────┐  ┌────────────────────────────┐
│         RAG Service              │  │    Document Processor      │
│  1. Embed query (OpenAI)         │  │  extract_text()            │
│  2. FAISS similarity search      │  │  PDF / DOCX / TXT / MD     │
│  3. Filter by score threshold    │  └────────────┬───────────────┘
│  4. Build context string         │               │
│  5. Call LLM with context        │               ▼
│  6. Return answer + sources      │  ┌────────────────────────────┐
└──────────────────┬───────────────┘  │    Vector Store Service    │
                   │                  │  RecursiveCharacterSplitter│
                   │                  │  OpenAI Embeddings         │
                   ▼                  │  FAISS IndexFlatIP         │
┌──────────────────────────────────┐  │  JSON metadata store       │
│         LLM Service              │  │  Persist to disk           │
│  ChatOpenAI (GPT-4o)             │  └────────────────────────────┘
│  System prompt + context         │
│  Conversation history injection  │
│  Token usage tracking            │
└──────────────────────────────────┘
```

---

## 📁 Project Structure

```
rag-chatbot/
├── main.py                         # FastAPI app entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example                    # Environment variables template
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── chat.py             # POST /api/v1/chat
│   │       ├── documents.py        # Document CRUD endpoints
│   │       └── health.py           # GET /health
│   ├── core/
│   │   ├── config.py               # Pydantic settings (env vars)
│   │   ├── dependencies.py         # FastAPI DI — singleton services
│   │   └── logging.py              # Structured logging setup
│   ├── models/
│   │   └── schemas.py              # Pydantic request/response models
│   └── services/
│       ├── vector_store.py         # FAISS indexing + retrieval
│       ├── llm_service.py          # GPT-4o generation via LangChain
│       ├── rag_service.py          # RAG orchestration pipeline
│       └── document_processor.py  # PDF/DOCX/TXT text extraction
├── tests/
│   ├── test_rag_service.py
│   └── test_vector_store.py
└── data/
    └── faiss_index/                # Persisted FAISS index (git-ignored)
```

---

## 🚀 Getting Started

### Option A — Docker (Recommended)

```bash
git clone https://github.com/yasirmalik36/rag-chatbot.git
cd rag-chatbot

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

docker-compose up --build
```

API available at `http://localhost:8000` · Swagger UI at `http://localhost:8000/docs`

### Option B — Local Development

**Prerequisites:** Python 3.11+

```bash
git clone https://github.com/yasirmalik36/rag-chatbot.git
cd rag-chatbot

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY to .env

python main.py
```

---

## 📡 API Usage

### 1. Upload a Document

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@your-document.pdf"
```

**Response:**
```json
{
  "document_id": "a3f2c1d4-...",
  "filename": "your-document.pdf",
  "status": "indexed",
  "chunks_created": 42,
  "message": "Document indexed successfully with 42 chunks"
}
```

### 2. Ask a Question

```bash
curl -X POST "http://localhost:8000/api/v1/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the refund policy?",
    "session_id": "user-123",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "answer": "Based on the policy document, refunds are processed within 5-7 business days...",
  "session_id": "user-123",
  "sources": [
    {
      "content": "Refunds are processed within 5-7 business days of approval...",
      "source": "policy.pdf",
      "score": 0.923
    }
  ],
  "tokens_used": 312,
  "latency_ms": 847,
  "model": "gpt-4o",
  "timestamp": "2025-06-07T10:30:00Z"
}
```

### 3. Multi-turn Conversation

```bash
curl -X POST "http://localhost:8000/api/v1/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How long does that take?",
    "session_id": "user-123",
    "history": [
      {"role": "user", "content": "What is the refund policy?"},
      {"role": "assistant", "content": "Refunds are processed within 5-7 business days..."}
    ]
  }'
```

### 4. List Documents

```bash
curl "http://localhost:8000/api/v1/documents/"
```

### 5. Delete a Document

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/a3f2c1d4-..."
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

```
tests/test_rag_service.py::TestRAGService::test_chat_returns_answer         PASSED
tests/test_rag_service.py::TestRAGService::test_chat_assigns_session_id     PASSED
tests/test_rag_service.py::TestRAGService::test_chat_uses_provided_session_id PASSED
tests/test_rag_service.py::TestRAGService::test_chat_includes_latency       PASSED
tests/test_rag_service.py::TestRAGService::test_chat_with_history           PASSED
tests/test_rag_service.py::TestRAGService::test_chat_no_context_still_returns PASSED
tests/test_rag_service.py::TestRAGService::test_source_content_truncated    PASSED
```

---

## 🔑 Key Design Decisions

**Why FAISS over Pinecone/Weaviate?**
Zero external dependencies, sub-millisecond retrieval, full control over the index. For <1M chunks, FAISS in-memory outperforms managed services on latency. Swap to Pinecone by replacing `VectorStoreService` — the interface is the same.

**Why IndexFlatIP with L2 normalisation?**
Cosine similarity on normalised vectors = Inner Product. More semantically meaningful than Euclidean distance for text embeddings.

**Why chunk size 512 with 64 overlap?**
Balances semantic coherence (too small = fragmented context) vs retrieval precision (too large = irrelevant content dilutes the answer). Overlap prevents losing context at chunk boundaries.

**Why temperature 0.2?**
Low temperature = more deterministic, factual answers. For customer support, accuracy > creativity.

---

## 🚀 Production Deployment (Azure)

```bash
# Build and push to Azure Container Registry
az acr build --registry myregistry --image rag-chatbot:latest .

# Deploy to Azure Container Apps
az containerapp create \
  --name rag-chatbot \
  --resource-group my-rg \
  --image myregistry.azurecr.io/rag-chatbot:latest \
  --env-vars OPENAI_API_KEY=secretref:openai-key
```

---

## 🤝 Author

**Yasir Mehmood** — Senior Full-Stack & AI Engineer | Rawalpindi, Pakistan

[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-1B3A6B?style=flat-square)](https://yasir-portfolio-omega.vercel.app)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/yasir-mehmood-53549a1b3)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
