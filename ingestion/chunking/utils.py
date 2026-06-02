import hashlib
import re


SECTION_PATTERNS = (
    re.compile(r"^\s*section\s+\d+\s*:", re.IGNORECASE),
    re.compile(r"^\s*[A-Z][A-Za-z0-9 &/()'-]{2,}\s*\|\s*", re.IGNORECASE),
    re.compile(r"^\s*Q\d+\s*:", re.IGNORECASE),
    re.compile(r"^\s*[EMHX]\d+\s+", re.IGNORECASE),
)

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
TOKEN_PATTERN = re.compile(r"\b[\w.+#/-]+\b", re.UNICODE)
MARKDOWN_TABLE_ROW = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)


def estimate_tokens(text: str) -> int:
    """Fast token estimate suitable for chunk sizing before model tokenizers."""
    return len(TOKEN_PATTERN.findall(text))


def split_paragraphs(text: str) -> list[str]:
    prepared = text or ""
    prepared = re.sub(r"\s+(Section\s+\d+\s*:)", r"\n\n\1", prepared)
    prepared = re.sub(r"\s+(Q\d+\s*:)", r"\n\n\1", prepared)
    prepared = re.sub(r"\s+([A-Z][A-Za-z& ]{1,30}\s+\|\s+Technical Focus:)", r"\n\n\1", prepared)
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", prepared)]
    return [paragraph for paragraph in paragraphs if paragraph]


def split_sentences(text: str) -> list[str]:
    """Split sentences conservatively to avoid shredding abbreviations/tables."""
    if "|" in text:
        return [line.strip() for line in text.splitlines() if line.strip()]
    return [sentence.strip() for sentence in SENTENCE_BOUNDARY.split(text) if sentence.strip()]


def is_semantic_boundary(paragraph: str) -> bool:
    return any(pattern.search(paragraph) for pattern in SECTION_PATTERNS)


def infer_semantic_type(text: str, object_type: str | None = None) -> str:
    lowered = text.lower()
    if object_type == "table" or text.lstrip().startswith("Table columns:") or MARKDOWN_TABLE_ROW.search(text):
        return "table"
    if "round details" in lowered or "technical focus" in lowered or "interview" in lowered:
        return "interview_experience"
    if "cgpa" in lowered or "eligibility" in lowered or "backlog" in lowered:
        return "eligibility"
    if "trend" in lowered or "2021" in lowered or "2024" in lowered:
        return "temporal_trend"
    if "conflict" in lowered or "official" in lowered and "portal" in lowered:
        return "conflict_record"
    return "placement_report"


def normalized_fingerprint_text(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\s.+#/-]", "", normalized)
    return normalized.strip()


def fingerprint_text(text: str) -> str:
    return hashlib.sha256(normalized_fingerprint_text(text).encode("utf-8")).hexdigest()


def chunk_id(source_object_id: str, chunk_index: int, fingerprint: str) -> str:
    return f"{source_object_id}:chunk:{chunk_index}:{fingerprint[:10]}"


def merge_text(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part.strip()).strip()

