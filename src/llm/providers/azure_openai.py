# src/llm/providers/azure_openai.py
from __future__ import annotations

from typing import Any

from ..base import ProviderAdapter

JSON = dict[str, Any]
MsgList = list[dict[str, Any]]
ToolList = list[dict[str, Any]] | None

class AzureOpenAIAdapter(ProviderAdapter):
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

        params = {}
        if v := self.cfg.get("api_version"):
            params["api-version"] = v
        headers = {"api-key": self.api_key}

        return ("/chat/completions", params, headers, payload)

    def parse_response(self, data: JSON) -> JSON:
        choice = data["choices"][0]
        return {
            "message": choice["message"],
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
            "model": data.get("model", self._model()),
        }
