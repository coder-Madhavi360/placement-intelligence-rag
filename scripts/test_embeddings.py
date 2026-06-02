import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "Placement_RAG_Dataset_Enhanced.pdf"
INDEX_DIR = PROJECT_ROOT / "data" / "vectorstores"
INDEX_PATH = INDEX_DIR / "placement_intelligence.faiss"
METADATA_PATH = INDEX_DIR / "placement_intelligence.metadata.json"
CACHE_PATH = INDEX_DIR / "all_minilm_l6_v2_cache.json"

sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.chunking.chunk_models import ChunkingConfig
from ingestion.chunking.semantic_chunker import SemanticChunker
from retrieval.embeddings.embedder import EmbeddingError, SentenceTransformerEmbedder
from retrieval.embeddings.vector_models import EmbeddingConfig
from ingestion.pdf_loader import PDFIngestionError, PDFLoader
from retrieval.vectorstores.index_manager import VectorIndexManager
from retrieval.vectorstores.search import SemanticSearchService


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    print("\n===== EMBEDDING + FAISS INDEX TEST =====")
    print(f"PDF path: {PDF_PATH}")

    if not PDF_PATH.exists():
        print("\n===== EMBEDDING FAILURE =====")
        print(f"Missing PDF file: {PDF_PATH}")
        return 1

    try:
        extracted = PDFLoader().load(PDF_PATH)
        chunks = SemanticChunker(
            ChunkingConfig(max_tokens=260, overlap_tokens=40, min_chunk_tokens=20)
        ).chunk_documents(extracted).chunks
    except PDFIngestionError as exc:
        print("\n===== EMBEDDING FAILURE =====")
        print(f"PDF ingestion failed: {exc}")
        return 1

    embedder = SentenceTransformerEmbedder(
        EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=384,
            batch_size=32,
            cache_enabled=True,
            cache_path=CACHE_PATH,
        )
    )

    try:
        manager = VectorIndexManager(embedder=embedder)
        store, embedded_chunks, index_stats = manager.build_index(chunks)
        saved_stats = manager.save(INDEX_PATH, METADATA_PATH)

        loaded_manager = VectorIndexManager.load(
            embedder=embedder,
            index_path=INDEX_PATH,
            metadata_path=METADATA_PATH,
        )
        search_service = SemanticSearchService(embedder=embedder, store=loaded_manager.store)
        results = search_service.search(
            "Which companies allow backlogs and offer high package?",
            top_k=5,
        )
    except EmbeddingError as exc:
        print("\n===== EMBEDDING FAILURE =====")
        print(str(exc))
        print("Model download may be required the first time all-MiniLM-L6-v2 is used.")
        return 1
    except Exception as exc:
        print("\n===== EMBEDDING FAILURE =====")
        print(str(exc))
        return 1

    print("\n===== EMBEDDING SUCCESS =====")
    print(f"Chunks embedded: {len(embedded_chunks)}")
    print(f"Model: {embedder.last_stats.model_name}")
    print(f"Dimensions: {embedder.last_stats.dimensions}")
    print(f"Cache hits: {embedder.last_stats.cache_hits}")
    print(f"Cache misses: {embedder.last_stats.cache_misses}")

    print("\n===== INDEX STATISTICS =====")
    print(f"Index name: {index_stats.index_name}")
    print(f"Vector count: {index_stats.vector_count}")
    print(f"Metadata count: {index_stats.metadata_count}")
    print(f"Dimensions: {index_stats.dimensions}")
    print(f"Metric: {index_stats.metric}")
    print(f"Saved index: {saved_stats.saved_index_path}")
    print(f"Saved metadata: {saved_stats.saved_metadata_path}")
    print(f"Loaded index vector count: {loaded_manager.store.stats().vector_count}")

    print("\n===== SIMILARITY SEARCH RESULTS =====")
    for result in results:
        preview = result.content[:600].replace("\n", "\n  ")
        print(f"\n--- Rank {result.rank} ---")
        print(f"score: {result.score:.4f}")
        print(f"chunk_id: {result.chunk_id}")
        print(f"semantic_type: {result.semantic_type}")
        print(f"page: {result.metadata.page}")
        print(f"source_file: {result.metadata.extra.get('source_file')}")
        print(f"content:\n  {preview}")

    if not results:
        print("\n===== EMBEDDING FAILURE =====")
        print("No search results were returned.")
        return 1

    print("\n===== TEST PASSED =====")
    print("Embeddings generated, FAISS index built, and semantic search completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


