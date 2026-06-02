import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "Placement_RAG_Dataset_Enhanced.pdf"

from ingestion.chunking.chunk_models import ChunkingConfig
from ingestion.chunking.semantic_chunker import SemanticChunker
from ingestion.pdf_loader import PDFIngestionError, PDFLoader


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    print("\n===== SEMANTIC CHUNKING TEST =====")
    print(f"PDF path: {PDF_PATH}")

    if not PDF_PATH.exists():
        print("\n===== CHUNKING FAILURE =====")
        print(f"Missing PDF file: {PDF_PATH}")
        return 1

    try:
        extracted_objects = PDFLoader().load(PDF_PATH)
    except PDFIngestionError as exc:
        print("\n===== CHUNKING FAILURE =====")
        print(f"PDF ingestion failed: {exc}")
        return 1

    config = ChunkingConfig(max_tokens=260, overlap_tokens=40, min_chunk_tokens=20)
    result = SemanticChunker(config=config).chunk_documents(extracted_objects)

    print("\n===== CHUNKING SUCCESS =====")
    print(f"Extracted document objects: {len(extracted_objects)}")
    print(f"Raw chunks: {result.statistics.raw_chunks}")
    print(f"Final chunks: {result.statistics.final_chunks}")
    print(f"Duplicates removed: {result.statistics.duplicate_chunks_removed}")
    print(f"Average tokens: {result.statistics.average_tokens}")
    print(f"Min tokens: {result.statistics.min_tokens}")
    print(f"Max tokens: {result.statistics.max_tokens}")
    print(f"By semantic type: {result.statistics.by_semantic_type}")

    print("\n===== DEDUPLICATION STATISTICS =====")
    print(f"Input chunks: {result.deduplication.input_count}")
    print(f"Unique chunks: {result.deduplication.unique_count}")
    print(f"Duplicate chunks: {result.deduplication.duplicate_count}")
    if result.deduplication.duplicate_ids:
        print("Sample duplicate IDs:")
        for duplicate_id in result.deduplication.duplicate_ids[:5]:
            print(f"- {duplicate_id}")

    print("\n===== SAMPLE CHUNKS =====")
    for index, chunk in enumerate(result.chunks[:5], start=1):
        preview = chunk.content[:700].replace("\n", "\n  ")
        print(f"\n--- Chunk {index} ---")
        print(f"id: {chunk.id}")
        print(f"type: {chunk.semantic_type}")
        print(f"tokens: {chunk.token_count}")
        print(f"page: {chunk.metadata.page}")
        print(f"tags: {chunk.metadata.tags}")
        print(f"content:\n  {preview}")

    if not result.chunks:
        print("\n===== CHUNKING FAILURE =====")
        print("No chunks were generated.")
        return 1

    print("\n===== TEST PASSED =====")
    print("Semantic chunks generated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
