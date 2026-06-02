import logging
from pathlib import Path
from typing import Any

import pdfplumber

from ingestion.cleaner import TextCleaner
from ingestion.metadata import ExtractedDocumentObject, MetadataBuilder, PDFDocumentMetadata
from ingestion.table_extractor import TableExtractor
from core.schemas.common import Modality
from core.schemas.documents import DocumentChunk

logger = logging.getLogger(__name__)


class PDFIngestionError(RuntimeError):
    """Raised when a PDF cannot be loaded into structured document objects."""


class PDFLoader:
    """Production-oriented PDF loader for placement intelligence datasets.

    The loader extracts clean page text and table objects separately so RAG
    retrieval can target salary grids, eligibility matrices, company-role
    tables, and regular prose with different chunking strategies later.
    """

    def __init__(
        self,
        cleaner: TextCleaner | None = None,
        table_extractor: TableExtractor | None = None,
        metadata_builder: MetadataBuilder | None = None,
    ) -> None:
        self.cleaner = cleaner or TextCleaner()
        self.table_extractor = table_extractor or TableExtractor(cleaner=self.cleaner)
        self.metadata_builder = metadata_builder or MetadataBuilder()

    def load(
        self,
        pdf_path: str | Path,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[ExtractedDocumentObject]:
        """Load a PDF into structured text and table objects."""
        path = Path(pdf_path).expanduser().resolve()
        if not path.exists():
            raise PDFIngestionError(f"PDF file not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise PDFIngestionError(f"Expected a .pdf file, received: {path}")

        logger.info("Starting PDF ingestion for %s", path)

        try:
            with pdfplumber.open(path) as pdf:
                document_metadata = self.metadata_builder.build_document_metadata(
                    pdf_path=path,
                    page_count=len(pdf.pages),
                    extra=extra_metadata,
                )
                objects = self._extract_pdf_objects(pdf, document_metadata)
        except PDFIngestionError:
            raise
        except Exception as exc:
            logger.exception("PDF ingestion failed for %s", path)
            raise PDFIngestionError(f"Failed to ingest PDF: {path}") from exc

        logger.info("Completed PDF ingestion for %s with %s objects", path, len(objects))
        return objects

    def load_as_chunks(
        self,
        pdf_path: str | Path,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Load PDF objects into the existing ingestion service schema."""
        return [
            DocumentChunk(
                id=document_object.id,
                content=document_object.content,
                modality=document_object.modality,
                metadata=document_object.metadata,
            )
            for document_object in self.load(pdf_path, extra_metadata=extra_metadata)
        ]

    def _extract_pdf_objects(
        self,
        pdf: pdfplumber.PDF,
        document_metadata: PDFDocumentMetadata,
    ) -> list[ExtractedDocumentObject]:
        objects: list[ExtractedDocumentObject] = []

        for page_index, page in enumerate(pdf.pages, start=1):
            logger.info(
                "Processing page %s/%s for %s",
                page_index,
                document_metadata.page_count,
                document_metadata.file_name,
            )
            objects.extend(self._extract_page_text(page, document_metadata, page_index))
            objects.extend(self._extract_page_tables(page, document_metadata, page_index))

        return objects

    def _extract_page_text(
        self,
        page: pdfplumber.page.Page,
        document_metadata: PDFDocumentMetadata,
        page_number: int,
    ) -> list[ExtractedDocumentObject]:
        try:
            raw_text = page.extract_text(
                layout=False,
                x_tolerance=2,
                y_tolerance=3,
                keep_blank_chars=False,
                use_text_flow=True,
            )
        except Exception:
            logger.exception("Text extraction failed for %s page %s", document_metadata.file_name, page_number)
            return []

        cleaned_text = self.cleaner.clean_page_text(raw_text)
        if not cleaned_text:
            logger.debug("No text extracted for %s page %s", document_metadata.file_name, page_number)
            return []

        logger.info(
            "Extracted %s characters of text from %s page %s",
            len(cleaned_text),
            document_metadata.file_name,
            page_number,
        )

        metadata = self.metadata_builder.build_object_metadata(
            document_metadata=document_metadata,
            page_number=page_number,
            object_type="page_text",
            content=cleaned_text,
        )

        return [
            ExtractedDocumentObject(
                id=self.metadata_builder.object_id(document_metadata, page_number, "text", 1),
                content=cleaned_text,
                modality=Modality.TEXT,
                object_type="page_text",
                page_number=page_number,
                metadata=metadata,
            )
        ]

    def _extract_page_tables(
        self,
        page: pdfplumber.page.Page,
        document_metadata: PDFDocumentMetadata,
        page_number: int,
    ) -> list[ExtractedDocumentObject]:
        tables = self.table_extractor.extract_tables(page, page_number)
        objects: list[ExtractedDocumentObject] = []

        if tables:
            logger.info(
                "Preparing %s table object(s) from %s page %s",
                len(tables),
                document_metadata.file_name,
                page_number,
            )

        for table in tables:
            content = self._format_table_content(table.markdown, table.headers)
            metadata = self.metadata_builder.build_object_metadata(
                document_metadata=document_metadata,
                page_number=page_number,
                object_type="table",
                content=content,
                table_index=table.table_index,
            )

            objects.append(
                ExtractedDocumentObject(
                    id=self.metadata_builder.object_id(
                        document_metadata,
                        page_number,
                        "table",
                        table.table_index,
                    ),
                    content=content,
                    modality=Modality.TEXT,
                    object_type="table",
                    page_number=page_number,
                    metadata=metadata,
                )
            )

        return objects

    def _format_table_content(self, markdown: str, headers: list[str]) -> str:
        header_summary = ", ".join(headers)
        return f"Table columns: {header_summary}\n\n{markdown}"
