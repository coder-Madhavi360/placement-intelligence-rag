import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "data" / "vectorstores"
INDEX_PATH = INDEX_DIR / "placement_intelligence.faiss"
METADATA_PATH = INDEX_DIR / "placement_intelligence.metadata.json"
CACHE_PATH = INDEX_DIR / "all_minilm_l6_v2_cache.json"

os.environ["RAG_EMBEDDING_PROVIDER"] = "sentence_transformers"
os.environ["RAG_EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
os.environ["RAG_EMBEDDING_DIMENSIONS"] = "384"
os.environ["RAG_EMBEDDING_BATCH_SIZE"] = "32"
os.environ["RAG_EMBEDDING_CACHE_PATH"] = str(CACHE_PATH)
os.environ["RAG_VECTOR_STORE"] = "faiss"
os.environ["RAG_FAISS_INDEX_PATH"] = str(INDEX_PATH)
os.environ["RAG_FAISS_METADATA_PATH"] = str(METADATA_PATH)
os.environ.setdefault("RAG_LLM_MODEL", "gpt-4o-mini")
os.environ["RAG_OPENAI_API_KEY"] = ""

from fastapi.testclient import TestClient

from core.api.deps import get_faiss_store, get_sentence_transformer_embedder
from core.config import get_settings
from core.main import app


def clear_runtime_caches() -> None:
    get_settings.cache_clear()
    get_sentence_transformer_embedder.cache_clear()
    get_faiss_store.cache_clear()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    print("\n===== LLM RAG PIPELINE TEST =====")
    print(f"FAISS index: {INDEX_PATH}")
    print(f"FAISS metadata: {METADATA_PATH}")

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        print("\n===== LLM PIPELINE FAILURE =====")
        print("Missing FAISS index artifacts. Run evaluation/test_embeddings.py first.")
        return 1

    payload = {
        "query": "What is Amazon eligibility criteria?",
        "top_k": 5,
        "filters": {},
        "modality": "text",
    }

    clear_runtime_caches()
    with TestClient(app) as client:
        response = client.post("/api/v1/query", json=payload)

    if response.status_code != 200:
        print("\n===== LLM PIPELINE FAILURE =====")
        print(f"Status code: {response.status_code}")
        print(response.text)
        return 1

    data = response.json()

    print("\n===== LLM PIPELINE SUCCESS =====")
    print(f"Answer: {data['answer']}")
    print(f"Model: {data['model']}")
    print(f"Contexts: {len(data['contexts'])}")
    print(f"Sources: {data['sources']}")
    print(f"Retrieval time ms: {data['retrieval_time_ms']}")
    print(f"Generation time ms: {data['generation_time_ms']}")

    print("\n===== TOP CONTEXTS =====")
    for index, context in enumerate(data["contexts"][:3], start=1):
        preview = context["content"][:500].replace("\n", "\n  ")
        print(f"\n--- Context {index} ---")
        print(f"score: {context['score']:.4f}")
        print(f"source: {context['metadata']['extra'].get('source_file')} Page {context['metadata'].get('page')}")
        print(f"content:\n  {preview}")

    answer_lower = data["answer"].lower()
    if "insufficient information" in answer_lower:
        print("\n===== LLM PIPELINE FAILURE =====")
        print("The answer generator did not use retrieved Amazon context.")
        return 1
    if "amazon" not in answer_lower:
        print("\n===== LLM PIPELINE FAILURE =====")
        print("Generated answer does not mention Amazon.")
        return 1
    if not data["contexts"]:
        print("\n===== LLM PIPELINE FAILURE =====")
        print("No contexts returned.")
        return 1
    if not data["sources"]:
        print("\n===== LLM PIPELINE FAILURE =====")
        print("No sources returned.")
        return 1

    print("\n===== TEST PASSED =====")
    print("LLM answer generation pipeline returned a grounded answer with sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
