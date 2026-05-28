import re


class TextCleaner:
    """Normalize extracted PDF text while preserving semantic boundaries."""

    _hyphenated_line_break = re.compile(r"(\w)-\n(\w)")
    _single_line_break = re.compile(r"(?<!\n)\n(?!\n)")
    _excess_whitespace = re.compile(r"[ \t]+")
    _excess_blank_lines = re.compile(r"\n{3,}")
    _malformed_chars = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _binary_noise_line = re.compile(
        r"^\s*(%PDF-\d\.\d|%%EOF|xref|trailer|startxref|stream|endstream|"
        r"endobj|\d+\s+\d+\s+obj|/[A-Z][A-Za-z]+\b|<<|>>).*$",
        re.IGNORECASE,
    )

    def clean_page_text(self, text: str | None) -> str:
        """Clean raw text from one PDF page.

        The cleaner is intentionally conservative. It removes PDF extraction
        artifacts without lowercasing or stripping domain terms such as CTC,
        CGPA, role names, company names, or eligibility criteria.
        """
        if not text:
            return ""

        cleaned = self._remove_pdf_binary_noise(text)
        cleaned = self._malformed_chars.sub("", cleaned)
        cleaned = self._hyphenated_line_break.sub(r"\1\2", cleaned)
        cleaned = self._normalize_lines(cleaned)
        cleaned = self._excess_blank_lines.sub("\n\n", cleaned)
        return cleaned.strip()

    def clean_cell(self, value: object | None) -> str:
        """Normalize a table cell without destroying row/column meaning."""
        if value is None:
            return ""

        text = self._malformed_chars.sub("", str(value))
        text = self._single_line_break.sub(" ", text)
        text = self._excess_whitespace.sub(" ", text)
        return text.strip()

    def clean_heading(self, text: str) -> str:
        return self._excess_whitespace.sub(" ", text).strip(" :-\t")

    def _remove_pdf_binary_noise(self, text: str) -> str:
        """Drop lines that look like PDF syntax rather than extracted content."""
        lines = []
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            stripped = line.strip()
            if self._binary_noise_line.match(stripped):
                continue
            lines.append(line)
        return "\n".join(lines)

    def _normalize_lines(self, text: str) -> str:
        """Preserve paragraphs while joining artificial intra-paragraph wraps."""
        normalized_lines = [self._excess_whitespace.sub(" ", line).strip() for line in text.split("\n")]

        paragraphs: list[str] = []
        current: list[str] = []

        for line in normalized_lines:
            if not line:
                if current:
                    paragraphs.append(" ".join(current).strip())
                    current = []
                continue

            current.append(line)

        if current:
            paragraphs.append(" ".join(current).strip())

        return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)
