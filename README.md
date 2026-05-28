# Placement Intelligence RAG

Production-ready FastAPI backend for answering placement-intelligence questions from an ingested PDF knowledge base. The system extracts PDF content, chunks it semantically, embeds chunks, stores vectors in FAISS, retrieves relevant evidence, and generates grounded answers with an LLM fallback path.

## Architecture Overview

```text
PDF dataset
  -> PDFLoader / MetadataBuilder
  -> SemanticChunker
  -> SentenceTransformerEmbedder
  -> FAISSVectorStore
  -> RetrievalService + RetrievalRanker
  -> LLMService
  -> FastAPI response
```

The backend is layered so HTTP code stays thin and business logic stays testable:

- API routes validate requests and delegate to services.
- Services coordinate ingestion, retrieval, and answer generation.
- Retrieval modules handle query embedding, FAISS search, ranking, and score metadata.
- Embedding modules isolate model loading, batching, and cache behavior.
- Vector store adapters hide storage implementation details.
- Core modules own environment settings, logging, middleware, and exception handling.

## Folder Structure

```text
backend/
  main.py                    # ASGI import compatibility: backend.main:app
  app/
    main.py                  # FastAPI app factory and middleware
    api/
      deps.py                # Dependency injection providers
      v1/
        router.py            # Versioned API composition
        routes/              # Current route handlers
        endpoints/           # Backward-compatible legacy route module
    chunking/                # Semantic chunking and deduplication
    core/
      config.py              # Environment settings
      exceptions.py          # Centralized API exception handling
      logging.py             # Text/JSON logging setup
      middleware.py          # Request logging middleware
    embeddings/              # Embedding provider interfaces/adapters
    ingestion/               # PDF extraction, cleaning, table parsing, metadata
    retrieval/               # Retrieval orchestration and ranking
    schemas/                 # Pydantic request/response contracts
    services/                # Business workflows
    vectorstores/            # Vector database interfaces/adapters
data/
  Placement_RAG_Dataset_Enhanced.pdf
scripts/
  test_embeddings.py         # Build and persist the FAISS index
  test_retrieval.py          # Validate retrieval behavior
  test_llm_pipeline.py       # Validate grounded answer generation
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Configure `.env`:

```env
RAG_OPENAI_API_KEY=your_openai_key
RAG_LLM_MODEL=gpt-4o-mini
RAG_FAISS_INDEX_PATH=data/vectorstores/placement_intelligence.faiss
RAG_FAISS_METADATA_PATH=data/vectorstores/placement_intelligence.metadata.json
RAG_CHUNK_MAX_TOKENS=260
RAG_MAX_RETRIEVAL_RESULTS=5
```

Build the local FAISS index:

```bash
python scripts/test_embeddings.py
```

Run the API:

```bash
uvicorn backend.main:app --reload
```

Open:

- API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/api/v1/health`
- Version: `http://127.0.0.1:8000/api/v1/version`

## API Usage

Ingest documents:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingest ^
  -H "Content-Type: application/json" ^
  -d "{\"documents\":[{\"id\":\"doc-1\",\"content\":\"Campus placement preparation includes aptitude, coding, projects, and interview practice.\",\"modality\":\"text\",\"metadata\":{\"source\":\"starter\"}}]}"
```

Query the index:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"How should I prepare for placements?\",\"top_k\":3}"
```

Legacy-compatible RAG query endpoint:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/rag/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is Amazon eligibility criteria?\"}"
```

## Sample Queries

- `Which companies allow backlogs and offer a high package?`
- `What is Amazon eligibility criteria?`
- `Which companies are suitable for students with 7 CGPA?`
- `Summarize placement preparation advice from the dataset.`
- `Which roles mention coding or aptitude rounds?`

## Screenshots

Add screenshots before sharing or presenting:

- `docs/screenshots/swagger-ui.png` - FastAPI Swagger UI
- `docs/screenshots/query-response.png` - Sample RAG response
- `docs/screenshots/retrieval-logs.png` - Retrieval and answer-generation logs

## Production Notes

- Centralized exception handlers return a consistent error shape.
- Request, ingestion, retrieval, embedding, FAISS, and LLM paths emit structured logs.
- Startup validation warns when OpenAI or FAISS runtime prerequisites are missing.
- Environment variables use the `RAG_` prefix to avoid deployment collisions.
- The answer generator falls back to deterministic grounded output if OpenAI is unavailable.

## Future Improvements

- Add authentication, rate limiting, and per-request correlation IDs.
- Move PDF ingestion and index building to a background worker.
- Add CI with unit tests, integration tests, linting, and type checks.
- Add observability with OpenTelemetry traces and metrics.
- Add hosted vector database adapters for Qdrant, Pinecone, or Weaviate.
- Add an evaluation set for retrieval quality and hallucination checks.
