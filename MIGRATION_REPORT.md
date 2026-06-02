# Migration Report

| Old path | New path |
|---|---|
| `backend/app/main.py` | `core/main.py` |
| `backend/app/api/` | `core/api/` |
| `backend/app/core/` | `core/` |
| `backend/app/schemas/` | `core/schemas/` |
| `backend/app/ingestion/` | `ingestion/` |
| `backend/app/chunking/` | `ingestion/chunking/` |
| `backend/app/embeddings/` | `retrieval/embeddings/` |
| `backend/app/vectorstores/` | `retrieval/vectorstores/` |
| `backend/app/retrieval/` | `retrieval/` |
| `backend/app/services/retrieval_service.py` | `retrieval/service.py` |
| `backend/app/services/document_service.py` | `ingestion/service.py` |
| `backend/app/services/llm_service.py` | `generation/llm_service.py` |
| `backend/app/services/rag_service.py` | `generation/rag_service.py` |
| `backend/app/services/chat_service.py` | `feedback/chat_service.py` |
| `backend/app/chat/` | `feedback/chat/` |
| `backend/main.py` | `backend/main.py` compatibility shim to `core.main:app` |
| missing root Streamlit app | `app.py` |
| missing version module | `version.py` |

Unused duplicate folders under `backend/app/` were removed after their modules were moved.
