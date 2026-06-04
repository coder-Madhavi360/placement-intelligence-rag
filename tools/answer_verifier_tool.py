"""
tools/answer_verifier_tool.py

Answer Verifier Tool — independently re-checks the generated answer
against the source chunks using a second LLM call.

Why: the main generation prompt is long and complex (system instructions
+ context + query type instructions). Sometimes the LLM follows the
instructions rather than the context. A simple second verification call
with a clean prompt catches these cases.

Process:
1. Take the generated answer
2. Take the retrieved chunks
3. Ask the LLM a simple yes/no: "Is this answer supported by this context?"
4. If no → flag the answer

This is different from recitation (which checks tokens) —
the verifier understands semantic meaning.
"""
import os
import logging
from tools.base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class AnswerVerifierTool(BaseTool):
    """
    Verifies a generated answer against source chunks
    using an independent LLM call with a minimal prompt.
    """

    @property
    def name(self) -> str:
        return "answer_verifier"

    @property
    def description(self) -> str:
        return (
            "Independently verifies whether a generated answer "
            "is supported by the retrieved chunks."
        )

    def execute(self, query: str) -> ToolResult:
        # This tool is called programmatically, not via router
        return ToolResult(
            tool_name=self.name,
            success=False,
            output="Use verify(answer, chunks) directly.",
        )

    def verify(
        self,
        question:   str,
        answer:     str,
        chunks:     list,
        api_key:    str = None,
        model:      str = "llama-3.1-8b-instant",
    ) -> dict:
        """
        Returns dict with keys: supported (bool), confidence (float),
        explanation (str).
        """
        context = "\n".join(
            f"[{c.section}|{c.company}]: {c.text[:200]}"
            for c in chunks[:4]
        )

        verify_prompt = f"""You are a fact-checker.

Context from documents:
{context}

Question: {question}
Answer to verify: {answer}

Is the answer fully supported by the context above?
Reply in exactly this format:
SUPPORTED: yes/no
CONFIDENCE: 0.0 to 1.0
REASON: one sentence

Do not add anything else."""

        try:
            from groq import Groq
            key    = api_key or os.getenv("GROQ_API_KEY")
            client = Groq(api_key=key)
            resp   = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": verify_prompt}],
                temperature=0.0,
                max_tokens=80,
            )
            raw = resp.choices[0].message.content.strip()
            return self._parse_verification(raw)

        except Exception as e:
            logger.warning(f"AnswerVerifier failed: {e}")
            return {
                "supported":   True,
                "confidence":  0.5,
                "explanation": f"Verification unavailable: {e}",
            }

    def _parse_verification(self, raw: str) -> dict:
        import re
        supported_match   = re.search(
            r"SUPPORTED:\s*(yes|no)", raw, re.IGNORECASE
        )
        confidence_match  = re.search(
            r"CONFIDENCE:\s*([\d\.]+)", raw
        )
        reason_match      = re.search(
            r"REASON:\s*(.+)", raw
        )
        supported   = (
            supported_match.group(1).lower() == "yes"
            if supported_match else True
        )
        confidence  = (
            float(confidence_match.group(1))
            if confidence_match else 0.5
        )
        explanation = (
            reason_match.group(1).strip()
            if reason_match else raw
        )
        return {
            "supported":   supported,
            "confidence":  min(1.0, confidence),
            "explanation": explanation,
        }