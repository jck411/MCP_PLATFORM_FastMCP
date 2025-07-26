# src/llm/providers/openai_compat.py
from __future__ import annotations

from typing import Any

from ..base import ProviderAdapter

JSON = dict[str, Any]
MsgList = list[dict[str, Any]]
ToolList = list[dict[str, Any]] | None

class OpenAICompatibleAdapter(ProviderAdapter):
    """
    Works unchanged for:
      • api.openai.com
      • api.groq.com/openai/v1
      • openrouter.ai/api/v1
    """

    def build_request(
        self, messages: MsgList, tools: ToolList
    ) -> tuple[str, dict[str, str], dict[str, str], JSON]:
        payload: JSON = {
            "model": self._model(),
            "messages": messages,
            "temperature": self._temperature(),
            "max_tokens": self._max_tokens(),
            "top_p": self._top_p(),
        }
        if tools:
            payload["tools"] = tools

        headers = {"Authorization": f"Bearer {self.api_key}"}
        return ("/chat/completions", {}, headers, payload)

    def parse_response(self, data: JSON) -> JSON:
        choice = data["choices"][0]
        return {
            "message": choice["message"],
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
            "model": data.get("model", self._model()),
        }
