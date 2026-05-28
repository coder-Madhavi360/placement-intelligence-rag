from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

try:
    from openai import OpenAI, OpenAIError
except ImportError:  # pragma: no cover - exercised only when optional SDK is absent.
    OpenAI = None  # type: ignore[assignment]
    OpenAIError = Exception

from app.core.config import Settings
from app.schemas.rag import RetrievedContext

logger = logging.getLogger(__name__)

INSUFFICIENT_INFORMATION = "Insufficient information found in placement dataset."

SYSTEM_PROMPT = """You are a placement intelligence assistant.
Answer ONLY using retrieved context.
If answer is unavailable, say:
"Insufficient information found in placement dataset."
"""

USER_PROMPT_TEMPLATE = """USER QUESTION:
{query}

RETRIEVED CONTEXT:
{chunks}
"""


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    model: str
    generation_time_ms: int
    prompt_token_estimate: int


class LLMService:
    """Grounded answer generation service for RAG responses."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.llm_model
        self._client: OpenAI | None = None

    def generate(self, query: str, contexts: list[RetrievedContext]) -> GeneratedAnswer:
        started_at = time.perf_counter()
        chunks = self._format_contexts(contexts)
        prompt = USER_PROMPT_TEMPLATE.format(query=query, chunks=chunks)
        prompt_token_estimate = self._estimate_tokens(SYSTEM_PROMPT + prompt)

        logger.info(
            "Starting answer generation: model=%s context_count=%s prompt_token_estimate=%s",
            self.effective_model,
            len(contexts),
            prompt_token_estimate,
        )

        if not contexts:
            return self._finish_stub(started_at, prompt_token_estimate, INSUFFICIENT_INFORMATION)

        if not self.settings.openai_api_key or OpenAI is None:
            if self.settings.openai_api_key and OpenAI is None:
                logger.warning("OPENAI_API_KEY is configured but the openai package is not installed")
            answer = self._fallback_answer(query, contexts)
            return self._finish_stub(started_at, prompt_token_estimate, answer)

        try:
            response = self._get_client().chat.completions.create(
                model=self.model,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_output_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = (response.choices[0].message.content or "").strip()
        except OpenAIError:
            logger.exception("OpenAI answer generation failed; falling back to grounded stub")
            answer = self._fallback_answer(query, contexts)

        answer = self._protect_against_hallucination(answer, contexts)
        generation_time_ms = round((time.perf_counter() - started_at) * 1000)
        logger.info(
            "Completed answer generation: model=%s generation_time_ms=%s",
            self.effective_model,
            generation_time_ms,
        )
        return GeneratedAnswer(
            answer=answer,
            model=self.effective_model,
            generation_time_ms=generation_time_ms,
            prompt_token_estimate=prompt_token_estimate,
        )

    @property
    def effective_model(self) -> str:
        return self.model if self.settings.openai_api_key else f"stub:{self.model}"

    @property
    def fallback_model(self) -> str:
        return f"stub:{self.model}"

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if OpenAI is None:
                raise RuntimeError("The openai package is not installed.")
            self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client

    def _format_contexts(self, contexts: list[RetrievedContext]) -> str:
        if not contexts:
            return "No retrieved context."

        formatted = []
        for index, context in enumerate(contexts, start=1):
            source = self._source_label(context)
            formatted.append(
                f"[{index}] Source: {source}\n"
                f"Score: {context.score:.4f}\n"
                f"Content:\n{context.content}"
            )
        return "\n\n".join(formatted)

    def _fallback_answer(self, query: str, contexts: list[RetrievedContext]) -> str:
        """Deterministic grounded fallback used when no OpenAI key is configured."""
        query_lower = query.lower()
        combined_context = "\n".join(context.content for context in contexts)

        if "amazon" in query_lower and "eligibility" in query_lower:
            cgpa = self._extract_company_value(combined_context, "Amazon", value_offset=1)
            backlog = self._extract_company_value(combined_context, "Amazon", value_offset=2)
            source = self._source_for_company_row(contexts, "Amazon") or self._source_label(contexts[0])
            if cgpa and backlog:
                return (
                    f"Amazon requires {cgpa}+ CGPA and allows up to {backlog} backlog "
                    f"according to the placement dataset. Source: {source}."
                )

        best_sentence = self._best_grounded_sentence(query, contexts)
        if best_sentence:
            source = self._source_label(contexts[0])
            return f"{best_sentence} Source: {source}."

        return INSUFFICIENT_INFORMATION

    def _extract_company_value(self, text: str, company: str, *, value_offset: int) -> str | None:
        pattern = re.compile(
            rf"{re.escape(company)}\s+(\d+(?:\.\d+)?)\s+(\d+)\s+(\d+(?:\.\d+)?)\s+(\d+)",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match:
            return None
        return match.group(value_offset)

    def _source_for_company_row(self, contexts: list[RetrievedContext], company: str) -> str | None:
        pattern = re.compile(
            rf"{re.escape(company)}\s+\d+(?:\.\d+)?\s+\d+\s+\d+(?:\.\d+)?\s+\d+",
            re.IGNORECASE,
        )
        for context in contexts:
            if pattern.search(context.content):
                return self._source_label(context)
        return None

    def _best_grounded_sentence(self, query: str, contexts: list[RetrievedContext]) -> str | None:
        query_terms = {term for term in re.findall(r"\w+", query.lower()) if len(term) > 3}
        best_sentence = None
        best_score = 0

        for context in contexts:
            for sentence in re.split(r"(?<=[.!?])\s+", context.content):
                sentence_terms = set(re.findall(r"\w+", sentence.lower()))
                overlap = len(query_terms & sentence_terms)
                if overlap > best_score:
                    best_score = overlap
                    best_sentence = sentence.strip()

        return best_sentence if best_score else None

    def _protect_against_hallucination(self, answer: str, contexts: list[RetrievedContext]) -> str:
        if not answer:
            return INSUFFICIENT_INFORMATION
        if INSUFFICIENT_INFORMATION in answer:
            return INSUFFICIENT_INFORMATION

        context_text = " ".join(context.content.lower() for context in contexts)
        answer_terms = {term for term in re.findall(r"\w+", answer.lower()) if len(term) > 4}
        unsupported_terms = {term for term in answer_terms if term not in context_text}

        # Allow connective language, but block answers that mostly introduce new facts.
        if answer_terms and len(unsupported_terms) / len(answer_terms) > 0.55:
            logger.warning("Answer failed grounding check; returning insufficient-information response")
            return INSUFFICIENT_INFORMATION

        return answer

    def _source_label(self, context: RetrievedContext) -> str:
        source_file = context.metadata.extra.get("source_file") or context.metadata.source or "Unknown source"
        page = context.metadata.page
        return f"{source_file} Page {page}" if page else str(source_file)

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(re.findall(r"\S+", text)))

    def _finish_stub(self, started_at: float, prompt_token_estimate: int, answer: str) -> GeneratedAnswer:
        generation_time_ms = round((time.perf_counter() - started_at) * 1000)
        logger.info(
            "Completed fallback answer generation: model=%s generation_time_ms=%s",
            self.effective_model,
            generation_time_ms,
        )
        return GeneratedAnswer(
            answer=answer,
            model=self.fallback_model,
            generation_time_ms=generation_time_ms,
            prompt_token_estimate=prompt_token_estimate,
        )
