# AI Study Assistant

An AI-powered study assistant that lets you upload study documents and ask questions about them. It uses RAG (Retrieval-Augmented Generation) to answer questions grounded in your documents, a safe calculator tool, and a tool-calling agent that routes questions to the right tool automatically.

---

## Features

- 📄 **Document Upload** — PDF, TXT, DOCX, Markdown, CSV
- 🔀 **Hybrid Search** — BM25 keyword retrieval + FAISS dense semantic search
- 🎯 **Cross-Encoder Reranking** — `ms-marco-MiniLM-L-6-v2` reranks candidate chunks for maximum precision
- 🔍 **Grounded RAG Pipeline** — only top-reranked chunks passed to LLM with strict grounding
- 📚 **Source Citations** — every answer includes the document and page it came from
- 🧮 **Calculator Tool** — safe AST-based math evaluation (no `eval`)
- 🤖 **Tool-Calling Agent** — automatically routes to calculator, document search, or conversation
- 💬 **Conversation Memory** — follow-up questions work across multiple turns
- ⚡ **FastAPI Backend** — REST API with validation and error handling
- 🖥️ **Streamlit UI** — clean chat interface connected end-to-end to the backend

---

## Architecture

```
User
 │
 ▼
Streamlit UI  (frontend/app.py)
 │  HTTP
 ▼
FastAPI Backend  (backend/main.py)
 │
 ├─ POST /upload ──► Document Loader ──► Chunker ──► Embeddings ──► FAISS index
 │
 └─ POST /ask ────► Agent (LLM + tools)
                      │
                      ├─ Calculator Tool ──────────────────────────► Result
                      │   (safe AST math)
                      │
                      └─ Document Search Tool
                           │
                           ▼
                        Hybrid Search
                        ┌──────────────┐
                        │ BM25 + FAISS │
                        └──────┬───────┘
                               ▼
                        Candidate Chunks (Top 10)
                               ▼
                        Cross-Encoder Reranker
                        (ms-marco-MiniLM-L-6-v2)
                               ▼
                        Top Chunks (Top 4)
                               ▼
                        Grounded Prompt
                               ▼
                        LLM (Groq)
                               ▼
                        Answer + Sources

Conversation history is stored in-memory per session (session_id).
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API (`openai/gpt-oss-20b`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | FAISS (CPU) |
| Keyword Search | `rank-bm25` (BM25Okapi) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers) |
| RAG Framework | LangChain |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Language | Python 3.11+ |

---

## Project Structure

```
ai-study-assistant/
├── backend/
│   ├── main.py               # FastAPI app entry point
│   ├── api/
│   │   ├── routes.py         # /health, /upload, /ask, /chat, /documents
│   │   └── schemas.py        # Pydantic request/response models
│   ├── rag/
│   │   ├── document_loader.py  # PDF/TXT/DOCX/MD/CSV loaders
│   │   ├── chunks.py           # RecursiveCharacterTextSplitter
│   │   ├── embeddings.py       # HuggingFace sentence-transformers
│   │   ├── vectorstore.py      # FAISS create/save/load/delete
│   │   ├── hybrid_search.py    # BM25 + FAISS hybrid retrieval & deduplication
│   │   ├── reranker.py         # Cross-Encoder candidate reranker
│   │   ├── retriever.py        # Hybrid reranked retriever interface
│   │   ├── generator.py        # Grounded prompt + LLM answer
│   │   ├── conversation.py     # Question rewriting + history helpers
│   │   ├── session_store.py    # In-memory session history store
│   │   └── rag_chain.py        # Full conversational RAG pipeline
│   ├── agent/
│   │   └── agent.py            # Tool-calling agent (calculator + doc search)
│   └── tools/
│       ├── calculator.py        # Safe AST-based math calculator
│       └── document_search.py  # Hybrid search + reranker document tool
├── frontend/
│   └── app.py                # Streamlit UI
├── data/
│   ├── uploads/              # Uploaded documents (git-ignored)
│   └── faiss_index/          # Generated FAISS index (git-ignored)
├── tests/
│   ├── test_hybrid_search_reranking.py  # Hybrid search & reranking test suite
│   ├── test_calculator_conversation_delete.py  # API and agent integration tests
│   └── test_rag_manual.py               # Manual RAG evaluation script
├── .env.example              # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/yogesh9685/ai-study-assistant.git
cd ai-study-assistant
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
copy .env.example .env      # Windows
cp .env.example .env        # Linux / macOS
```

Open `.env` and set your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [https://console.groq.com](https://console.groq.com).

---

## Running Locally

### Start the FastAPI backend

```bash
# From the project root
uvicorn backend.main:app --reload --port 8000
```

### Start the Streamlit frontend

```bash
# In a separate terminal, from the project root
streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | — | Groq API key for LLM access |
| `API_BASE_URL` | No | `http://localhost:8000` | Backend URL (set when deploying) |
| `APP_NAME` | No | `AI Study Assistant` | Application name |
| `ENVIRONMENT` | No | `development` | Runtime environment |
| `DEBUG` | No | `false` | Enable debug mode |
| `BACKEND_PORT` | No | `8000` | FastAPI server port |
| `FRONTEND_PORT` | No | `8501` | Streamlit port |

---

## Supported File Types

| Extension | Loader |
|---|---|
| `.pdf` | PyPDFLoader |
| `.txt` | TextLoader |
| `.docx` | Docx2txtLoader |
| `.md` | UnstructuredMarkdownLoader (fallback: TextLoader) |
| `.csv` | CSVLoader |

Maximum file size: **10 MB**

---

## API Endpoints

### `GET /health`
Check backend availability.
```json
{ "status": "ok" }
```

### `POST /upload`
Upload and index a study document.
- **Body:** `multipart/form-data` with `file`
- **Response:** `{ "message": "...", "filename": "...", "chunks": 42 }`

### `POST /ask`
Ask a question with conversation memory.
```json
{ "question": "What is Python?", "session_id": "abc-123" }
```
- **Response:** `{ "answer": "...", "sources": [{"source": "file.pdf", "page": 1}] }`

### `POST /chat`
Ask a one-off question (no session memory required).
```json
{ "question": "What is 25 * 8?" }
```

### `DELETE /documents`
Clear all uploaded documents and reset the FAISS index.

---

## RAG Pipeline (Hybrid Retrieval + Cross-Encoder Reranking)

```
Uploaded Document
      ▼
Document Loader   (PDF/TXT/DOCX/MD/CSV)
      ▼
Text Chunker      (chunk_size=800, overlap=100)
      ▼
Embeddings        (all-MiniLM-L6-v2, 384-dim)
      ▼
FAISS Index       (saved to data/faiss_index/)

User Question
      ▼
┌─────────────────────────┐
│     Hybrid Retrieval    │
│  BM25           FAISS   │
│ (keyword)     (semantic)│
└────────────┬────────────┘
             ▼
Candidate Chunks (Top 10, Deduplicated)
             ▼
Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
             ▼
Top Chunks (Top 4)
             ▼
Grounded Prompt   ("Answer using only the context below...")
             ▼
Groq LLM
             ▼
Answer + Sources
```

---

## Agent & Tool Flow

```
User Question
      ▼
Agent (Groq LLM + tools)
      ├── "25 * 8"              ──► Calculator Tool  ──► 200
      ├── "Explain Python"      ──► Document Search  ──► Hybrid + Reranker ──► Chunks ──► Answer
      └── "Hello"               ──► Direct LLM response
```

The agent uses LangChain's `create_agent` with a system prompt that instructs it:
- Use the **calculator** for any math expression
- Use **document_search** for anything about uploaded documents
- Answer directly for simple conversational questions
- Never invent document content

---

## Conversation

Each Streamlit session gets a unique `session_id` (UUID). The backend stores conversation history in memory per session. Follow-up questions like "Who created **it**?" are rewritten into standalone questions before retrieval.

> **Note:** History is in-memory only — it resets when the server restarts.

---

## Testing

### 1. Hybrid Search & Cross-Encoder Reranking Suite

Verify BM25 keyword matching, FAISS semantic search, deduplication, Cross-Encoder score reordering, and top-k filtering:

```bash
python tests/test_hybrid_search_reranking.py
```

### 2. End-to-End API, Conversation & Deletion Suite

Verify calculator evaluation, multi-turn conversation memory, document upload, hybrid document search via `/ask`, document deletion, and post-deletion fallback:

```bash
python tests/test_calculator_conversation_delete.py
```

### 3. Manual RAG Evaluation Script

```bash
# Requires a document at data/python.txt
python tests/test_rag_manual.py
```

This tests: document load, chunking, FAISS indexing, hybrid retrieval, answer generation, source extraction, and out-of-scope question handling.

---

## Deployment Preparation

Before deploying:

1. Set `GROQ_API_KEY` as a secret/environment variable on your hosting platform.
2. Set `API_BASE_URL` in the Streamlit environment to the deployed backend URL.
3. Ensure the platform supports persistent filesystem storage for `data/faiss_index/` (or re-upload documents after each deploy).
4. The FastAPI server binds to `0.0.0.0:8000` by default — confirm your platform routes traffic correctly.

---

## Known Limitations

- **FAISS index is not persistent across server restarts** on platforms with ephemeral filesystems (e.g., Railway, Render free tier). Users must re-upload documents after a restart.
- **Conversation history is in-memory** — lost on server restart. Replace `session_store.py` with a Redis-backed store for production persistence.
- **One document at a time** — uploading a new document replaces the previous FAISS index.
- **Max file size:** 10 MB.
