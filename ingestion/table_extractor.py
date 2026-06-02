import logging
from dataclasses import dataclass
from typing import Any

import pdfplumber

from ingestion.cleaner import TextCleaner

logger = logging.getLogger(__name__)


DEFAULT_TABLE_SETTINGS: dict[str, Any] = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 3,
    "min_words_horizontal": 1,
    "intersection_tolerance": 3,
    "text_tolerance": 3,
}

TEXT_TABLE_SETTINGS: dict[str, Any] = {
    **DEFAULT_TABLE_SETTINGS,
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
}


@dataclass(frozen=True)
class ExtractedTable:
    """A table represented as headers, records, and markdown for retrieval."""

    page_number: int
    table_index: int
    headers: list[str]
    rows: list[dict[str, str]]
    markdown: str


class TableExtractor:
    """Extract and normalize tables while preserving row/column structure."""

    def __init__(
        self,
        cleaner: TextCleaner | None = None,
        table_settings: dict[str, Any] | None = None,
    ) -> None:
        self.cleaner = cleaner or TextCleaner()
        self.table_settings = table_settings or DEFAULT_TABLE_SETTINGS

    def extract_tables(self, page: pdfplumber.page.Page, page_number: int) -> list[ExtractedTable]:
        try:
            raw_tables = page.extract_tables(table_settings=self.table_settings)
        except Exception:
            logger.exception("Failed to extract tables from PDF page %s", page_number)
            return []

        if not raw_tables:
            raw_tables = self._extract_borderless_tables(page, page_number)

        extracted: list[ExtractedTable] = []

        for index, raw_table in enumerate(raw_tables, start=1):
            normalized = self._normalize_table(raw_table)
            if not normalized:
                logger.debug("Skipping empty or malformed table %s on page %s", index, page_number)
                continue

            headers, rows = normalized
            markdown = self._to_markdown(headers, rows)

            extracted.append(
                ExtractedTable(
                    page_number=page_number,
                    table_index=index,
                    headers=headers,
                    rows=rows,
                    markdown=markdown,
                )
            )

        logger.info("Extracted %s table(s) from page %s", len(extracted), page_number)
        return extracted

    def _extract_borderless_tables(
        self,
        page: pdfplumber.page.Page,
        page_number: int,
    ) -> list[list[list[object | None]]]:
        """Fallback for reports that use whitespace instead of table lines."""
        try:
            return page.extract_tables(table_settings=TEXT_TABLE_SETTINGS)
        except Exception:
            logger.exception("Fallback table extraction failed for PDF page %s", page_number)
            return []

    def _normalize_table(self, raw_table: list[list[object | None]]) -> tuple[list[str], list[dict[str, str]]] | None:
        if not raw_table:
            return None

        cleaned_rows = [
            [self.cleaner.clean_cell(cell) for cell in row]
            for row in raw_table
            if row and any(self.cleaner.clean_cell(cell) for cell in row)
        ]

        if not cleaned_rows:
            return None

        width = max(len(row) for row in cleaned_rows)
        padded_rows = [row + [""] * (width - len(row)) for row in cleaned_rows]
        headers = self._deduplicate_headers(padded_rows[0])

        data_rows = padded_rows[1:] if len(padded_rows) > 1 else []
        records = [
            {headers[column_index]: value for column_index, value in enumerate(row)}
            for row in data_rows
            if any(row)
        ]

        if not records:
            records = [{headers[column_index]: value for column_index, value in enumerate(padded_rows[0])}]

        return headers, records

    def _deduplicate_headers(self, headers: list[str]) -> list[str]:
        names: list[str] = []
        seen: dict[str, int] = {}

        for index, header in enumerate(headers, start=1):
            base = header or f"column_{index}"
            count = seen.get(base, 0) + 1
            seen[base] = count
            names.append(base if count == 1 else f"{base}_{count}")

        return names

    def _to_markdown(self, headers: list[str], rows: list[dict[str, str]]) -> str:
        safe_headers = [self._escape_markdown_cell(header) for header in headers]
        header_line = "| " + " | ".join(safe_headers) + " |"
        separator_line = "| " + " | ".join("---" for _ in headers) + " |"
        row_lines = [
            "| " + " | ".join(self._escape_markdown_cell(row.get(header, "")) for header in headers) + " |"
            for row in rows
        ]
        return "\n".join([header_line, separator_line, *row_lines])

    def _escape_markdown_cell(self, value: str) -> str:
        return value.replace("|", "\\|")
