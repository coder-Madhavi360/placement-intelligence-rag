"""
feedback/loop.py
AIMD feedback controller — tracks answer quality and adjusts retrieval cap.
"""

import logging
from dataclasses import dataclass
from feedback.storage import save_feedback

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    query: str
    retrieval_quality: float
    context_tokens: int
    overshadow_risk: float
    user_rating: int = 0  # 1=good, -1=bad, 0=unrated


class FeedbackLoop:
    """
    Additive Increase / Multiplicative Decrease controller.
    Tells the overshadow limiter to grow or shrink context cap.
    """

    def __init__(self, limiter):
        self.limiter = limiter
        self.records: list[FeedbackRecord] = []

    def record(
        self,
        query: str,
        quality: float,
        tokens: int,
        risk: float,
    ) -> FeedbackRecord:
        rec = FeedbackRecord(
            query=query,
            retrieval_quality=quality,
            context_tokens=tokens,
            overshadow_risk=risk,
        )

        self.records.append(rec)
        return rec

    def good(self, rec: FeedbackRecord):
        rec.user_rating = 1

        if hasattr(self.limiter, "feedback_good"):
            self.limiter.feedback_good()

        save_feedback(
            {
                "query": rec.query,
                "rating": "good",
                "retrieval_quality": rec.retrieval_quality,
                "context_tokens": rec.context_tokens,
                "overshadow_risk": rec.overshadow_risk,
            }
        )

        logger.info("Feedback GOOD received")

    def bad(self, rec: FeedbackRecord):
        rec.user_rating = -1

        if hasattr(self.limiter, "feedback_bad"):
            self.limiter.feedback_bad()

        save_feedback(
            {
                "query": rec.query,
                "rating": "bad",
                "retrieval_quality": rec.retrieval_quality,
                "context_tokens": rec.context_tokens,
                "overshadow_risk": rec.overshadow_risk,
            }
        )

        logger.info("Feedback BAD received")

    def summary(self) -> dict:
        rated = [r for r in self.records if r.user_rating != 0]

        good = sum(1 for r in rated if r.user_rating == 1)
        bad = sum(1 for r in rated if r.user_rating == -1)

        return {
            "total_queries": len(self.records),
            "rated": len(rated),
            "good": good,
            "bad": bad,
            "accuracy": round(good / len(rated), 2) if rated else 0.0,
            "current_cap": getattr(self.limiter, "current_cap", "N/A"),
        }