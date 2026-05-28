import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
PDF_PATH = PROJECT_ROOT / "data" / "Placement_RAG_Dataset_Enhanced.pdf"

sys.path.insert(0, str(BACKEND_ROOT))

from app.ingestion.pdf_loader import PDFIngestionError, PDFLoader


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    print("\n===== PDF INGESTION TEST =====")
    print(f"PDF path: {PDF_PATH}")

    if not PDF_PATH.exists():
        print("\n===== EXTRACTION FAILURE =====")
        print(f"Missing PDF file: {PDF_PATH}")
        return 1

    loader = PDFLoader()

    try:
        document_objects = loader.load(PDF_PATH)
    except PDFIngestionError as exc:
        print("\n===== EXTRACTION FAILURE =====")
        print(str(exc))
        return 1

    text_objects = [item for item in document_objects if item.object_type == "page_text"]
    table_objects = [item for item in document_objects if item.object_type == "table"]

    combined_text = "\n\n".join(item.content for item in text_objects).strip()
    first_metadata = document_objects[0].metadata if document_objects else None

    print("\n===== EXTRACTION SUCCESS =====")
    print(f"Structured objects: {len(document_objects)}")
    print(f"Text pages extracted: {len(text_objects)}")
    print(f"Extracted tables count: {len(table_objects)}")

    print("\n===== CLEAN EXTRACTED TEXT =====")
    if combined_text:
        print(combined_text)
    else:
        print("No readable text extracted.")

    print("\n===== METADATA =====")
    if first_metadata:
        metadata = first_metadata.model_dump()
        print(f"source_file: {metadata['extra'].get('source_file')}")
        print(f"page_count: {metadata['extra'].get('page_count')}")
        print(f"extracted_at: {metadata['extra'].get('extracted_at')}")
        print(f"source: {metadata.get('source')}")
        print(f"tags: {metadata.get('tags')}")
    else:
        print("No metadata available.")

    if not combined_text or combined_text.startswith("%PDF"):
        print("\n===== EXTRACTION FAILURE =====")
        print("Readable PDF text was not extracted correctly.")
        return 1

    print("\n===== TEST PASSED =====")
    print("Readable placement dataset content extracted successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
