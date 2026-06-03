from dataclasses import dataclass


@dataclass
class RouterDecision:
    needs_tool: bool = False
    tool_name: str = ""
    reason: str = ""
    result = None


class ToolRouter:
    def route(self, query: str):
        return RouterDecision()