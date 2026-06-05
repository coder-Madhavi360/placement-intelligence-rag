"""
tools/tool_router.py
Tool router — classifies query and dispatches to correct tool.
Acts as a simple agent: decide → act → observe → respond.
Interface Segregation: each tool is independent.
"""
import re
import logging
from dataclasses import dataclass
from tools.base_tool import BaseTool, ToolResult
from tools.web_search_tool import WebSearchTool
from tools.calculator_tool import CalculatorTool
from tools.date_tool import DateTool
from tools.opinion_guard_tool import OpinionGuardTool
from tools.intent_classifier import IntentClassifier

logger = logging.getLogger(__name__)


@dataclass
class RouterDecision:
    needs_tool: bool
    tool_name: str
    reason: str
    result: ToolResult | None = None



class ToolRouter:
    """
    Classifies queries and routes to appropriate tools.
    Falls back gracefully when no tool matches.
    """

    def __init__(self):
        self.tools: dict[str, BaseTool] = {
            "web_search":    WebSearchTool(),
            "calculator":    CalculatorTool(),
            "current_date":  DateTool(),
            "opinion_guard": OpinionGuardTool(),
        }
        self.classifier=IntentClassifier()

    def route(self, query: str) -> RouterDecision:
        """
        Returns a RouterDecision.
        If needs_tool=False, RAG pipeline handles it normally.
        """
        intent = self.classifier.classify(query)

        if intent in self.tools:
            result = self.tools[intent].execute(query)

            return RouterDecision(
        needs_tool=True,
        tool_name=intent,
        reason=f"Semantic intent: {intent}",
        result=result
    )

        return RouterDecision(
            needs_tool=False,
            tool_name="rag",
            reason="In-corpus query — handled by RAG pipeline",
        )

    def get_tool_info(self) -> list[dict]:
        """Returns info about all available tools — for UI display."""
        return [
            {"name": t.name, "description": t.description}
            for t in self.tools.values()
        ]