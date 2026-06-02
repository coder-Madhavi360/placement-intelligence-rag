# Placement Intelligence RAG

Placement Intelligence RAG answers placement-related questions from the bundled PDF knowledge base. It preserves the existing ingestion, semantic chunking, FAISS retrieval, reranking, grounded LLM generation, conversational memory, FastAPI endpoints, and evaluation scripts in a flatter production layout.

## Architecture

```text
placement-rag/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── version.py
├── app.py
├── core/
├── ingestion/
├── retrieval/
├── generation/
├── safety/
├── tools/
├── evaluation/
├── feedback/
├── scripts/
└── data/
```

Runtime flow:

```text
PDFLoader
  -> SemanticChunker + deduplication
  -> SentenceTransformerEmbedder
  -> FAISSVectorStore
  -> RetrievalService + RetrievalRanker
  -> LLMService
  -> FastAPI / Streamlit responses
```

## Key Modules

- `core/`: FastAPI app factory, API routes, dependency injection, settings, logging, middleware, exceptions, and schemas.
- `ingestion/`: PDF loading, text cleaning, table extraction, metadata, document ingestion, and semantic chunking.
- `retrieval/`: embedding providers, FAISS vector store, index management, retrieval service, retriever, and reranker.
- `generation/`: grounded RAG orchestration and LLM answer generation with fallback behavior.
- `feedback/`: chat request/response models, conversation manager, in-memory chat store, and session-aware chat service.
- `evaluation/`: runnable validation scripts for ingestion, chunking, embeddings, retrieval, LLM pipeline, and chat memory.
- `app.py`: Streamlit chat frontend that calls the FastAPI `/api/v1/chat` endpoint.
- `data/`: source PDF and preserved FAISS artifacts under `data/vectorstores/`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Configure `.env` as needed:

```env
RAG_VECTOR_STORE=faiss
RAG_FAISS_INDEX_PATH=data/vectorstores/placement_intelligence.faiss
RAG_FAISS_METADATA_PATH=data/vectorstores/placement_intelligence.metadata.json
RAG_EMBEDDING_CACHE_PATH=data/vectorstores/all_minilm_l6_v2_cache.json
RAG_OPENAI_API_KEY=
RAG_LLM_MODEL=gpt-4o-mini
```

## Run

Start the FastAPI backend:

```bash
uvicorn core.main:app --reload
```

Start the Streamlit frontend in a second terminal:

```bash
streamlit run app.py
```

Useful URLs:

- FastAPI docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/api/v1/health`
- Version: `http://127.0.0.1:8000/api/v1/version`
- Streamlit: `http://localhost:8501`

## API Examples

Query RAG:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is Amazon eligibility criteria?\",\"top_k\":5}"
```

Conversational chat with memory:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Which companies allow one backlog?\"}"
```

Ingest documents:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingest ^
  -H "Content-Type: application/json" ^
  -d "{\"documents\":[{\"id\":\"doc-1\",\"content\":\"Placement preparation includes aptitude, coding, projects, and interview practice.\",\"modality\":\"text\",\"metadata\":{\"source\":\"manual\"}}]}"
```

## Evaluation

Run the validation scripts from the project root:

```bash
python evaluation/test_ingestion.py
python evaluation/test_chunking.py
python evaluation/test_embeddings.py
python evaluation/test_retrieval.py
python evaluation/test_llm_pipeline.py
python evaluation/test_chat_memory.py
```

`evaluation/test_embeddings.py` rebuilds and persists the FAISS index when needed. Existing FAISS files remain in `data/vectorstores/`.

## Migration Summary

| Old path | New path |
| --- | --- |
| `backend/app/main.py` | `core/main.py` |
| `backend/app/core/*` | `core/*` |
| `backend/app/api/*` | `core/api/*` |
| `backend/app/schemas/*` | `core/schemas/*` |
| `backend/app/ingestion/*` | `ingestion/*` |
| `backend/app/ingestion/chunking/*` | `ingestion/chunking/*` |
| `backend/app/retrieval/*` | `retrieval/*` |
| `backend/app/retrieval/embeddings/*` | `retrieval/embeddings/*` |
| `backend/app/retrieval/vectorstores/*` | `retrieval/vectorstores/*` |
| `backend/app/generation/*` | `generation/*` |
| `backend/app/feedback/*` | `feedback/*` |
| `backend/app/safety/*` | `safety/*` |
| `backend/app/tools/*` | `tools/*` |
| `scripts/test_*.py` | `evaluation/test_*.py` |
| `frontend/app.py` | `app.py` |
| `backend.main:app` | `core.main:app` |

Obsolete duplicate `backend/`, `frontend/`, and generated cache directories were removed after their implementations were moved.
