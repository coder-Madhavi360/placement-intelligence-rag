"""
safety/lookback_ratio_guard.py

Lookback Ratio — token-level context grounding measurement.

Definition:
    Lookback Ratio = |tokens(answer) ∩ tokens(context)| / |tokens(answer)|

Interpretation:
    1.0 → every word in the answer came from the context (fully grounded)
    0.0 → no word in the answer came from the context (pure parametric memory)
    0.5 → half the answer is grounded, half came from LLM memory

Why better than recitation:
    Recitation checks specific high-risk phrases (numbers, names).
    Lookback Ratio checks ALL tokens — gives a continuous grounding score
    rather than a binary pass/fail on selected patterns.

Camera sensor analogy (as per your professor's framework):
    Just like a camera sensor has a resolution spec that tells you exactly
    how sharp an image can be, the Lookback Ratio is the "resolution spec"
    of your RAG answer — it tells you exactly how much of the answer
    is real signal (from context) vs noise (from parametric memory).

    Lookback Ratio = your RAG's "megapixel count"
    Low ratio = blurry answer (hallucinated)
    High ratio = sharp answer (grounded)
"""
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Stopwords to exclude from lookback comparison
# These words appear in both context and any answer naturally
# and would inflate the ratio unfairly
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need",
    "to", "of", "in", "on", "at", "for", "with", "by", "from",
    "and", "or", "but", "not", "no", "so", "if", "as", "it",
    "its", "this", "that", "these", "those", "they", "them",
    "their", "there", "here", "then", "than", "which", "who",
    "what", "when", "where", "how", "all", "each", "more", "most",
    "i", "you", "we", "he", "she", "company", "companies",
}

# Thresholds
LOOKBACK_EXCELLENT  = 0.70   # Fully grounded — green
LOOKBACK_ACCEPTABLE = 0.45   # Mostly grounded — yellow
LOOKBACK_POOR       = 0.25   # Likely hallucinated — red


@dataclass
class LookbackReport:
    ratio:              float         # 0.0 – 1.0
    answer_tokens:      int           # total meaningful tokens in answer
    grounded_tokens:    int           # tokens found in context
    ungrounded_tokens:  list[str]     # top ungrounded content words
    verdict:            str           # EXCELLENT / ACCEPTABLE / POOR
    label:              str           # human-readable label
    bar:                str           # visual bar e.g. "████░░░░"


class LookbackRatioGuard:
    """
    Computes the Lookback Ratio for a generated answer against
    the retrieved context chunks.

    Usage:
        guard = LookbackRatioGuard()
        report = guard.compute(answer, chunks)
        print(report.ratio, report.verdict)
    """

    def compute(self, answer: str, chunks: list) -> LookbackReport:
        """
        Computes lookback ratio between answer and context.

        Args:
            answer: the generated answer string
            chunks: list of Chunk objects (must have .text attribute)

        Returns:
            LookbackReport with ratio, verdict, and diagnostics
        """
        # Build context token set from all chunks
        context_text  = " ".join(c.text for c in chunks).lower()
        context_tokens = self._tokenize(context_text)

        # Tokenize answer
        answer_tokens = self._tokenize(answer.lower())

        if not answer_tokens:
            return LookbackReport(
                ratio=1.0,
                answer_tokens=0,
                grounded_tokens=0,
                ungrounded_tokens=[],
                verdict="EXCELLENT",
                label="Empty answer",
                bar="█████████░",
            )

        # Count grounded tokens (appear in context)
        grounded    = [t for t in answer_tokens if t in context_tokens]
        ungrounded  = [t for t in answer_tokens if t not in context_tokens]
        ratio       = round(len(grounded) / len(answer_tokens), 4)

        # Determine verdict
        if ratio >= LOOKBACK_EXCELLENT:
            verdict = "EXCELLENT"
        elif ratio >= LOOKBACK_ACCEPTABLE:
            verdict = "ACCEPTABLE"
        else:
            verdict = "POOR"

        # Build visual bar (10 segments)
        filled  = int(ratio * 10)
        bar     = "█" * filled + "░" * (10 - filled)

        # Pick most suspicious ungrounded content words (non-stopword)
        suspicious = [
            t for t in ungrounded
            if t not in STOPWORDS
            and len(t) > 3
            and not t.isdigit()
        ][:5]

        label = {
            "EXCELLENT":  f"Fully grounded ({ratio:.0%} from context)",
            "ACCEPTABLE": f"Mostly grounded ({ratio:.0%} from context)",
            "POOR":       f"Low grounding ({ratio:.0%}) — possible hallucination",
        }[verdict]

        report = LookbackReport(
            ratio=ratio,
            answer_tokens=len(answer_tokens),
            grounded_tokens=len(grounded),
            ungrounded_tokens=suspicious,
            verdict=verdict,
            label=label,
            bar=bar,
        )

        logger.info(
            f"Lookback ratio: {ratio:.3f} [{verdict}] "
            f"({len(grounded)}/{len(answer_tokens)} tokens grounded)"
        )
        if suspicious:
            logger.debug(f"Ungrounded words: {suspicious}")

        return report

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenizes text into meaningful content words.
        Removes stopwords, punctuation, and very short tokens.
        """
        # Extract alphanumeric tokens
        raw = re.findall(r'\b[a-z0-9][a-z0-9\.\+\#]*\b', text.lower())
        return [
            t for t in raw
            if t not in STOPWORDS
            and len(t) >= 2
        ]

    def explain(self, report: LookbackReport) -> str:
        """Returns a human-readable explanation of the lookback report."""
        lines = [
            f"Lookback Ratio: {report.ratio:.3f} {report.bar}",
            f"Verdict: {report.verdict} — {report.label}",
            f"Grounded tokens: {report.grounded_tokens} / {report.answer_tokens}",
        ]
        if report.ungrounded_tokens:
            lines.append(
                f"Suspicious ungrounded words: {', '.join(report.ungrounded_tokens)}"
            )
        return "\n".join(lines)