# Placement Intelligence RAG

Placement Intelligence RAG answers placement-related questions from the enhanced PDF dataset using a preserved end-to-end pipeline: PDF ingestion, semantic chunking, sentence-transformer embeddings, FAISS retrieval, grounded answer generation, chat memory, FastAPI, and Streamlit.

## Architecture

```text
placement-rag/
  README.md
  requirements.txt
  .env.example
  .gitignore
  version.py
  app.py                         # Streamlit frontend
  backend/
    main.py                      # compatibility ASGI shim to core.main:app
  core/                          # FastAPI app, API routes, config, schemas, errors, logging
  ingestion/                     # PDF loading, text/table extraction, semantic chunking, document ingestion
  retrieval/                     # embeddings, FAISS/vector stores, search, ranking, retrieval service
  generation/                    # LLM service and RAG orchestration
  feedback/                      # chat models, conversation memory, chat service
  safety/                        # safety package boundary for future grounded-answer guards
  tools/                         # tool package boundary
  evaluation/                    # evaluation package boundary
  scripts/                       # pipeline verification scripts
  data/                          # placement PDF and generated vector artifacts
```

Runtime flow:

```text
PDF dataset
  -> ingestion.PDFLoader / TableExtractor / MetadataBuilder
  -> ingestion.chunking.SemanticChunker
  -> retrieval.embeddings.SentenceTransformerEmbedder
  -> retrieval.vectorstores.FAISSVectorStore
  -> retrieval.service.RetrievalService + RetrievalRanker
  -> generation.LLMService / RAGService
  -> FastAPI API and Streamlit UI
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Configure `.env` as needed. For local retrieval with FAISS, keep:

```env
RAG_EMBEDDING_PROVIDER=sentence_transformers
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_EMBEDDING_DIMENSIONS=384
RAG_VECTOR_STORE=faiss
RAG_FAISS_INDEX_PATH=data/vectorstores/placement_intelligence.faiss
RAG_FAISS_METADATA_PATH=data/vectorstores/placement_intelligence.metadata.json
RAG_OPENAI_API_KEY=
RAG_API_BASE_URL=http://127.0.0.1:8000
```

If `RAG_OPENAI_API_KEY` is empty, generation uses the existing deterministic grounded fallback.

## Build The FAISS Index

```bash
python scripts/test_embeddings.py
```

This ingests `data/Placement_RAG_Dataset_Enhanced.pdf`, chunks the extracted objects, embeds them with `sentence-transformers/all-MiniLM-L6-v2`, and writes FAISS artifacts under `data/vectorstores/`.

## Run

FastAPI:

```bash
uvicorn core.main:app --reload
```

Compatibility entrypoint:

```bash
uvicorn backend.main:app --reload
```

Streamlit frontend:

```bash
streamlit run app.py
```

Open:

- API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/api/v1/health`
- Version: `http://127.0.0.1:8000/api/v1/version`
- Streamlit: the local URL printed by `streamlit run app.py`

## API Usage

Ingest pre-chunked documents into the lightweight in-memory vector store:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingest ^
  -H "Content-Type: application/json" ^
  -d "{\"documents\":[{\"id\":\"doc-1\",\"content\":\"Campus placement preparation includes aptitude, coding, projects, and interview practice.\",\"modality\":\"text\",\"metadata\":{\"source\":\"starter\"}}]}"
```

Query the FAISS-backed RAG pipeline:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is Amazon eligibility criteria?\",\"top_k\":5,\"modality\":\"text\"}"
```

Chat with memory:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Am I eligible for Amazon?\"}"
```

Legacy-compatible RAG endpoint remains available:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/rag/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Which companies allow backlogs?\"}"
```

## Verification

Run these scripts from the project root:

```bash
python scripts/test_ingestion.py
python scripts/test_chunking.py
python scripts/test_embeddings.py
python scripts/test_retrieval.py
python scripts/test_llm_pipeline.py
python scripts/test_chat_memory.py
```

What they cover:

- ingestion: PDF text/table extraction and metadata
- chunking: semantic chunk generation and deduplication
- embeddings/indexing: sentence-transformer embeddings and FAISS persistence
- retrieval/API: FAISS load, ranking, `/api/v1/query`, restart cache simulation
- generation: grounded answer generation with sources
- memory: `/api/v1/chat` session memory wiring

## Migration

See `MIGRATION_REPORT.md` for the old path to new path mapping.
