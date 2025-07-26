# src/llm/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

JSON = dict[str, Any]
MsgList = list[dict[str, Any]]
ToolList = list[dict[str, Any]] | None

class ProviderAdapter(ABC):
    """
    Strategy interface for each provider.
    Concrete adapters produce a request and normalize the response.
    """

    def __init__(self, cfg: dict[str, Any], api_key: str):
        self.cfg = cfg
        self.api_key = api_key     # each adapter decides which header to use

    # ---------- helpers ----------
    def _model(self)        -> str  : return self.cfg["model"]
    def _temperature(self)  -> float: return float(self.cfg.get("temperature", 0.7))
    def _max_tokens(self)   -> int  : return int(self.cfg.get("max_tokens", 4096))
    def _top_p(self)        -> float: return float(self.cfg.get("top_p", 1.0))

    # ---------- interface ----------
    @abstractmethod
    def build_request(
        self, messages: MsgList, tools: ToolList
    ) -> tuple[str, dict[str, str], dict[str, str], JSON]:
        """
        → (path, query_params, headers, json_payload)
        """
        ...

    @abstractmethod
    def parse_response(self, data: JSON) -> JSON:
        """
        Normalize provider response to OpenAI-style:
          {
            "message": {
                "content": str,
                "tool_calls": list|None,
                "finish_reason": str|None
            },
            "finish_reason": str|None,
            "usage": dict|None,
            "model": str|None
          }
        """
        ...
