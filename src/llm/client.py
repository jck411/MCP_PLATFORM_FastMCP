# src/llm/client.py
from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from .base import ProviderAdapter
from .providers import (
    AnthropicAdapter,
    AzureOpenAIAdapter,
    OpenAICompatibleAdapter,
)

JSON = dict[str, Any]
MsgList = list[dict[str, Any]]
ToolList = list[dict[str, Any]] | None


class LLMClient:
    """
    Thin façade: choose adapter, forward calls.
    """

    def __init__(self, full_cfg: dict[str, Any], api_key: str) -> None:
        self.active_name = full_cfg["active"].lower()
        self.cfg = (full_cfg["providers"] or {}).get(self.active_name, {})
        self.base_url = self.cfg["base_url"].rstrip("/")
        self.api_key = api_key

        self.adapter: ProviderAdapter = self._make_adapter()
        self.http = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    # ---------- public ----------
    @property
    def model(self) -> str:
        """Get the model name from the current configuration."""
        return self.cfg.get("model", "")

    async def get_response_with_tools(
        self, messages: MsgList, tools: ToolList = None
    ) -> JSON:
        path, params, hdrs, payload = self.adapter.build_request(messages, tools)

        r = await self.http.post(path, params=params, headers=hdrs, json=payload)
        r.raise_for_status()
        return self.adapter.parse_response(r.json())

    # NOTE: A full streaming implementation per provider is more code.
    # For now we switch off streaming if provider !openai-compatible.
    async def get_streaming_response_with_tools(
        self, messages: MsgList, tools: ToolList = None
    ) -> AsyncGenerator[JSON]:
        if not isinstance(self.adapter, OpenAICompatibleAdapter):
            raise RuntimeError(
                f"Streaming not yet implemented for {self.active_name}"
            )

        path, params, hdrs, payload = self.adapter.build_request(messages, tools)
        payload["stream"] = True

        async with self.http.stream(
            "POST", path, params=params, headers=hdrs, json=payload
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    yield json.loads(data)

    async def close(self):
        await self.http.aclose()

    # ---------- helpers ----------
    def _make_adapter(self) -> ProviderAdapter:
        if self.active_name in ("openai", "groq", "openrouter"):
            return OpenAICompatibleAdapter(self.cfg, self.api_key)
        if self.active_name == "anthropic":
            return AnthropicAdapter(self.cfg, self.api_key)
        if self.active_name in ("azure", "azure_openai"):
            return AzureOpenAIAdapter(self.cfg, self.api_key)
        raise ValueError(f"Unknown LLM provider: {self.active_name}")
