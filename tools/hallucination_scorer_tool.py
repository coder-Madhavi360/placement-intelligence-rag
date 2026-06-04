"""
tools/hallucination_scorer_tool.py

Hallucination Scorer — combines all individual hallucination signals
into a single 0–100 trust score.

Signals combined:
    1. Lookback Ratio       (token-level grounding)    weight: 35%
    2. Self-consistency     (agreement across samples)  weight: 25%
    3. Recitation score     (phrase-level grounding)    weight: 20%
    4. Chain verified       (reasoning completeness)    weight: 10%
    5. Answer verification  (semantic support check)    weight: 10%

Interpretation:
    90–100  → High confidence — safe to use
    70–89   → Medium confidence — mostly reliable
    50–69   → Low confidence — verify key facts
    0–49    → Very low — likely hallucinated

This is your RAG pipeline's equivalent of a camera's
"image quality score" — a single number that summarises
how trustworthy the output is.
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

WEIGHTS = {
    "lookback":      0.35,
    "consistency":   0.25,
    "recitation":    0.20,
    "chain":         0.10,
    "verification":  0.10,
}


@dataclass
class TrustScore:
    score:          float      # 0–100
    grade:          str        # A / B / C / D / F
    label:          str        # human label
    color:          str        # green / yellow / orange / red
    components:     dict       # individual signal scores
    recommendation: str        # what to tell the user


class HallucinationScorerTool:
    """
    Computes a unified trust score from all hallucination signals.
    """

    def score(
        self,
        lookback_ratio:       float = 1.0,
        consistency_score:    float = 1.0,
        recitation_score:     float = 1.0,
        chain_verified:       bool  = True,
        verification_score:   float = 1.0,
    ) -> TrustScore:
        """
        Args:
            lookback_ratio:      from LookbackRatioGuard (0–1)
            consistency_score:   from SelfConsistencyChecker (0–1)
            recitation_score:    from RecitationChecker (0–1)
            chain_verified:      from ChainOfActionsVerifier (bool)
            verification_score:  from AnswerVerifierTool (0–1)

        Returns:
            TrustScore with unified 0–100 score
        """
        chain_val = 1.0 if chain_verified else 0.5

        components = {
            "lookback":     round(lookback_ratio,     3),
            "consistency":  round(consistency_score,  3),
            "recitation":   round(recitation_score,   3),
            "chain":        round(chain_val,          3),
            "verification": round(verification_score, 3),
        }

        weighted_sum = sum(
            components[k] * WEIGHTS[k]
            for k in WEIGHTS
        )
        score = round(weighted_sum * 100, 1)

        # Grade and label
        if score >= 90:
            grade          = "A"
            label          = "High confidence"
            color          = "green"
            recommendation = "Answer is well-grounded. Safe to use."
        elif score >= 70:
            grade          = "B"
            label          = "Medium confidence"
            color          = "yellow"
            recommendation = (
                "Answer is mostly reliable. "
                "Verify any specific numbers against the source."
            )
        elif score >= 50:
            grade          = "C"
            label          = "Low confidence"
            color          = "orange"
            recommendation = (
                "Answer may contain hallucinations. "
                "Check key facts in the source document."
            )
        else:
            grade          = "F"
            label          = "Very low confidence"
            color          = "red"
            recommendation = (
                "Answer likely contains hallucinations. "
                "Do not rely on this answer without verification."
            )

        logger.info(
            f"Trust score: {score}/100 [{grade}] — {label}"
        )

        return TrustScore(
            score=score,
            grade=grade,
            label=label,
            color=color,
            components=components,
            recommendation=recommendation,
        )

    def explain(self, trust: TrustScore) -> str:
        lines = [
            f"Trust Score: {trust.score}/100 (Grade {trust.grade})",
            f"Label: {trust.label}",
            f"",
            f"Signal breakdown:",
            f"  Lookback ratio      {trust.components['lookback']:.2f}  (weight 35%)",
            f"  Self-consistency    {trust.components['consistency']:.2f}  (weight 25%)",
            f"  Recitation score    {trust.components['recitation']:.2f}  (weight 20%)",
            f"  Chain verified      {trust.components['chain']:.2f}  (weight 10%)",
            f"  Verification score  {trust.components['verification']:.2f}  (weight 10%)",
            f"",
            f"Recommendation: {trust.recommendation}",
        ]
        return "\n".join(lines)