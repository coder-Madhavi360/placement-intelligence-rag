# Placement Intelligence RAG

Scalable FastAPI starter architecture for a multimodal retrieval augmented generation application.

## Architecture

```text
backend/
  main.py                    # ASGI import compatibility: backend.main:app
  app/
    main.py                  # FastAPI app factory and middleware
    api/
      deps.py                # FastAPI dependency providers
      v1/
        router.py            # Versioned API composition
        routes/              # Route handlers only
    core/
      config.py              # Environment settings
      logging.py             # Process logging setup
    schemas/                 # Pydantic request/response contracts
    services/                # Business workflows
    retrieval/               # Retrieval orchestration and ranking
    embeddings/              # Embedding provider interfaces/adapters
    vectorstores/            # Vector database interfaces/adapters
```

## Layering

- Routers validate HTTP input and delegate work.
- Services coordinate application use cases such as ingestion and answering.
- Retrieval owns query embedding and vector search orchestration.
- Embeddings hide provider-specific model calls.
- Vector stores hide database SDK details behind a common interface.
- Core settings and logging are imported by infrastructure code only.

The starter runs without external services by using a deterministic local embedding provider and an in-memory vector store. Runtime settings use the `RAG_` environment variable prefix to avoid collisions with platform-level variables. Production deployments should add concrete adapters for the selected embedding model, LLM, and vector database.

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Example Requests

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

## Production Notes

- Replace `LocalEmbeddingProvider` with a hosted or self-managed multimodal embedding model.
- Implement a concrete adapter under `app/vectorstores/` for Qdrant, Pinecone, Weaviate, or your preferred database.
- Add authentication, rate limiting, request IDs, tracing, and structured JSON logs before public exposure.
- Move long ingestion jobs to a worker queue when processing PDFs, images, video, or large crawls.
- Add tests around service behavior and each provider adapter contract.
