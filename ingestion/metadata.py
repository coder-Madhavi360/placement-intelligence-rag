from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.schemas.common import Metadata, Modality


PLACEMENT_KEYWORDS = {
    "company",
    "ctc",
    "stipend",
    "eligibility",
    "cgpa",
    "branch",
    "role",
    "package",
    "deadline",
    "interview",
    "aptitude",
    "coding",
    "round",
    "internship",
    "placement",
}


class PDFDocumentMetadata(BaseModel):
    """Metadata shared by every structured object extracted from one PDF."""

    source_path: str
    source_file: str
    file_name: str
    document_id: str
    page_count: int
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dataset: str = "placement_intelligence"
    extra: dict[str, Any] = Field(default_factory=dict)


class ExtractedDocumentObject(BaseModel):
    """Structured object ready for chunking, indexing, or audit storage."""

    id: str
    content: str
    modality: Modality = Modality.TEXT
    object_type: str
    page_number: int
    metadata: Metadata


class MetadataBuilder:
    """Build consistent metadata tags for placement intelligence retrieval."""

    def build_document_metadata(
        self,
        pdf_path: str | Path,
        page_count: int,
        extra: dict[str, Any] | None = None,
    ) -> PDFDocumentMetadata:
        path = Path(pdf_path)
        return PDFDocumentMetadata(
            source_path=str(path),
            source_file=path.name,
            file_name=path.name,
            document_id=path.stem,
            page_count=page_count,
            extra=extra or {},
        )

    def build_object_metadata(
        self,
        document_metadata: PDFDocumentMetadata,
        page_number: int,
        object_type: str,
        content: str,
        table_index: int | None = None,
    ) -> Metadata:
        tags = self._placement_tags(content)
        tags.extend([document_metadata.dataset, object_type])

        extra: dict[str, Any] = {
            "document_id": document_metadata.document_id,
            "source_file": document_metadata.source_file,
            "file_name": document_metadata.file_name,
            "object_type": object_type,
            "page_count": document_metadata.page_count,
            "extracted_at": document_metadata.extracted_at.isoformat(),
        }

        if table_index is not None:
            extra["table_index"] = table_index

        extra.update(document_metadata.extra)

        return Metadata(
            source=document_metadata.source_path,
            page=page_number,
            tags=sorted(set(tags)),
            extra=extra,
        )

    def object_id(
        self,
        document_metadata: PDFDocumentMetadata,
        page_number: int,
        object_type: str,
        ordinal: int,
    ) -> str:
        return f"{document_metadata.document_id}:p{page_number}:{object_type}:{ordinal}"

    def _placement_tags(self, content: str) -> list[str]:
        lowered = content.lower()
        return [keyword for keyword in PLACEMENT_KEYWORDS if keyword in lowered]
