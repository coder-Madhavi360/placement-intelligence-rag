# Migration Report

The repository now uses the target top-level package structure.

| Current path | Final path | Status |
|---|---|---|
| `app.py` | `app.py` | Streamlit entrypoint preserved |
| `version.py` | `version.py` | Version metadata preserved |
| `core/` | `core/` | FastAPI app, API routes, config, schemas, errors, logging preserved |
| `ingestion/` | `ingestion/` | PDF loading, cleaning, metadata, table extraction, chunking preserved |
| `retrieval/` | `retrieval/` | Embeddings, vector stores, ranking, retrieval service preserved |
| `generation/` | `generation/` | LLM and RAG services preserved |
| `safety/` | `safety/` | Package boundary preserved |
| `tools/` | `tools/` | Package boundary preserved |
| `evaluation/` | `evaluation/` | Package boundary preserved |
| `feedback/` | `feedback/` | Chat memory and chat service preserved |
| `scripts/` | `scripts/` | Verification scripts preserved |
| `data/` | `data/` | Dataset and retrieval artifacts preserved |
| `backend/main.py` | removed | Compatibility shim removed; use `core.main:app` |

## Import Changes

No business-logic imports were changed. The supported FastAPI entrypoint is:

```bash
uvicorn core.main:app --reload
```

The removed compatibility import was:

```python
from core.main import app
```

from `backend/main.py`.
