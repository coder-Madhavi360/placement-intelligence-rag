import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "Placement_RAG_Dataset_Enhanced.pdf"
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

from fastapi.testclient import TestClient

from core.api.deps import get_faiss_store, get_sentence_transformer_embedder
from ingestion.chunking.chunk_models import ChunkingConfig
from ingestion.chunking.semantic_chunker import SemanticChunker
from core.config import get_settings
from retrieval.embeddings.embedder import EmbeddingError, SentenceTransformerEmbedder
from retrieval.embeddings.vector_models import EmbeddingConfig
from ingestion.pdf_loader import PDFIngestionError, PDFLoader
from core.main import app
from retrieval.vectorstores.faiss_store import FAISSVectorStore
from retrieval.vectorstores.index_manager import VectorIndexManager


def ensure_index_exists() -> None:
    """Build the local FAISS index when the retrieval test is run fresh."""
    if INDEX_PATH.exists() and METADATA_PATH.exists():
        return

    print("\nNo persisted FAISS index found. Building one from the placement PDF...")
    extracted = PDFLoader().load(PDF_PATH)
    chunks = SemanticChunker(
        ChunkingConfig(max_tokens=260, overlap_tokens=40, min_chunk_tokens=20)
    ).chunk_documents(extracted).chunks

    embedder = SentenceTransformerEmbedder(
        EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=384,
            batch_size=32,
            cache_enabled=True,
            cache_path=CACHE_PATH,
        )
    )
    manager = VectorIndexManager(embedder=embedder)
    manager.build_index(chunks)
    manager.save(INDEX_PATH, METADATA_PATH)


def validate_persisted_index() -> FAISSVectorStore:
    """Validate disk artifacts before exercising the API dependency path."""
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(f"Missing FAISS artifacts: {INDEX_PATH}, {METADATA_PATH}")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    records = metadata.get("records", [])
    if metadata.get("dimensions") != 384:
        raise RuntimeError(f"Expected metadata dimensions=384, received {metadata.get('dimensions')}")
    if not records:
        raise RuntimeError("Metadata sidecar contains no stored chunk records.")

    first_record = records[0]
    if first_record.get("model_name") != "sentence-transformers/all-MiniLM-L6-v2":
        raise RuntimeError(f"Unexpected embedding model in metadata: {first_record.get('model_name')}")
    if first_record.get("dimensions") != 384:
        raise RuntimeError(f"Unexpected record dimensions: {first_record.get('dimensions')}")

    store = FAISSVectorStore.load(INDEX_PATH, METADATA_PATH)
    if store.count != len(records):
        raise RuntimeError(f"FAISS vector count {store.count} does not match metadata count {len(records)}")
    if store.dimensions != 384:
        raise RuntimeError(f"Expected FAISS dimensions=384, received {store.dimensions}")
    if store.embedding_model != "sentence-transformers/all-MiniLM-L6-v2":
        raise RuntimeError(f"Unexpected FAISS embedding model: {store.embedding_model}")

    return store


def clear_runtime_caches() -> None:
    """Simulate a server restart by forcing dependencies to reload from disk."""
    get_settings.cache_clear()
    get_sentence_transformer_embedder.cache_clear()
    get_faiss_store.cache_clear()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    print("\n===== RAG RETRIEVAL API TEST =====")
    print(f"PDF path: {PDF_PATH}")
    print(f"FAISS index: {INDEX_PATH}")

    if not PDF_PATH.exists():
        print("\n===== RETRIEVAL FAILURE =====")
        print(f"Missing PDF file: {PDF_PATH}")
        return 1

    try:
        ensure_index_exists()
        validated_store = validate_persisted_index()
    except (PDFIngestionError, EmbeddingError) as exc:
        print("\n===== RETRIEVAL FAILURE =====")
        print(str(exc))
        return 1
    except Exception as exc:
        print("\n===== RETRIEVAL FAILURE =====")
        print(f"Failed to prepare FAISS index: {exc}")
        return 1

    payload = {
        "query": "Which companies can a student with 7.6 CGPA and 1 backlog apply to for a high package?",
        "top_k": 5,
        "filters": {},
        "min_score": None,
    }

    clear_runtime_caches()
    legacy_payload = {
        "query": payload["query"],
        "top_k": payload["top_k"],
        "filters": payload["filters"],
        "modality": "text",
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/query", json=legacy_payload)

    if response.status_code != 200:
        print("\n===== RETRIEVAL FAILURE =====")
        print(f"Status code: {response.status_code}")
        print(response.text)
        return 1

    data = response.json()

    print("\n===== RETRIEVAL SUCCESS =====")
    print(f"Answer: {data['answer']}")
    print(f"Contexts: {len(data['contexts'])}")
    print(f"Model: {data['model']}")
    print(f"Validated disk vector count: {validated_store.count}")
    print(f"Validated metadata path: {METADATA_PATH}")

    print("\n===== TOP MATCHES =====")
    for index, context in enumerate(data["contexts"], start=1):
        preview = context["content"][:650].replace("\n", "\n  ")
        metadata = context["metadata"]

        print(f"\n--- Rank {index} ---")
        print(f"score: {context['score']:.4f}")
        print(f"chunk_id: {context['id']}")
        print(f"source_file: {metadata['extra'].get('source_file')}")
        print(f"page: {metadata.get('page')}")
        print(f"retrieval_score_metadata: {metadata['extra'].get('retrieval_score')}")
        print(f"semantic_type: {metadata['extra'].get('semantic_type')}")
        print(f"content:\n  {preview}")

    if len(data["contexts"]) == 0:
        print("\n===== RETRIEVAL FAILURE =====")
        print("The API returned no matches.")
        return 1
    if data["model"] != "sentence-transformers/all-MiniLM-L6-v2":
        print("\n===== RETRIEVAL FAILURE =====")
        print(f"API used an unexpected model: {data['model']}")
        return 1
    if "No relevant context found" in data["answer"]:
        print("\n===== RETRIEVAL FAILURE =====")
        print("The legacy empty-retrieval answer is still being returned.")
        return 1
    if not any("Amazon" in context["content"] or "Deloitte" in context["content"] for context in data["contexts"]):
        print("\n===== RETRIEVAL FAILURE =====")
        print("Search results did not include expected stored placement chunks.")
        return 1
    if not all(context["metadata"]["extra"].get("retrieval_score") is not None for context in data["contexts"]):
        print("\n===== RETRIEVAL FAILURE =====")
        print("Retrieved contexts are missing retrieval score metadata.")
        return 1

    print("\n===== SERVER RESTART SIMULATION =====")
    clear_runtime_caches()
    with TestClient(app) as client:
        restart_response = client.post("/api/v1/query", json=legacy_payload)
    restart_data = restart_response.json() if restart_response.status_code == 200 else {}
    if restart_response.status_code != 200 or len(restart_data.get("contexts", [])) == 0:
        print("\n===== RETRIEVAL FAILURE =====")
        print("Retrieval failed after dependency cache reset.")
        print(restart_response.text)
        return 1
    print("Retrieval still works after dependency cache reset.")

    print("\n===== TEST PASSED =====")
    print("RAG retrieval endpoint returned ranked semantic matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
