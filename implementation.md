

---

## 0 ▪ Folder map we will create

```
src/
  llm/                     ← NEW  (all provider logic lives here)
    __init__.py            ← exports LLMClient for imports
    base.py                ← ProviderAdapter interface
    providers/
      __init__.py
      openai_compat.py     ← OpenAI, Groq, OpenRouter
      anthropic.py
      azure_openai.py
main.py                    ← small diff: import class from src.llm
```

---

## 1 ▪ Create `src/llm/base.py`

```python
# src/llm/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

JSON = Dict[str, Any]
MsgList = List[Dict[str, Any]]
ToolList = Optional[List[Dict[str, Any]]]

class ProviderAdapter(ABC):
    """
    Strategy interface for each provider.
    Concrete adapters produce a request and normalize the response.
    """

    def __init__(self, cfg: Dict[str, Any], api_key: str):
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
    ) -> Tuple[str, Dict[str, str], Dict[str, str], JSON]:
        """
        → (path, query_params, headers, json_payload)
        """
        ...

    @abstractmethod
    def parse_response(self, data: JSON) -> JSON:
        """
        Normalize provider response to OpenAI‑style:
          {
            "message": {"content": str, "tool_calls": list|None, "finish_reason": str|None},
            "finish_reason": str|None,
            "usage": dict|None,
            "model": str|None
          }
        """
        ...
```

---

## 2 ▪ Create `src/llm/providers/openai_compat.py`

```python
# src/llm/providers/openai_compat.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from ..base import ProviderAdapter

JSON      = Dict[str, Any]
MsgList   = List[Dict[str, Any]]
ToolList  = Optional[List[Dict[str, Any]]]

class OpenAICompatibleAdapter(ProviderAdapter):
    """
    Works unchanged for:
      • api.openai.com
      • api.groq.com/openai/v1
      • openrouter.ai/api/v1
    """

    def build_request(
        self, messages: MsgList, tools: ToolList
    ) -> Tuple[str, Dict[str, str], Dict[str, str], JSON]:
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
```

---

## 3 ▪ Create `src/llm/providers/anthropic.py`

```python
# src/llm/providers/anthropic.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Tuple
from ..base import ProviderAdapter

JSON      = Dict[str, Any]
MsgList   = List[Dict[str, Any]]
ToolList  = Optional[List[Dict[str, Any]]]

def _system_prompt(msgs: MsgList) -> Optional[str]:
    for m in msgs:
        if m["role"] == "system":
            return m["content"]
    return None

def _to_anthropic_msgs(msgs: MsgList) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in msgs:
        role = m["role"]
        if role == "system":
            continue
        out.append({"role": role, "content": [{"type": "text", "text": m["content"]}]})
    return out

def _to_anthropic_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    res = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t["function"]
        res.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn["parameters"],
            }
        )
    return res

class AnthropicAdapter(ProviderAdapter):
    def build_request(
        self, messages: MsgList, tools: ToolList
    ) -> Tuple[str, Dict[str, str], Dict[str, str], JSON]:
        payload: JSON = {
            "model": self._model(),
            "messages": _to_anthropic_msgs(messages),
            "max_tokens": self._max_tokens(),
            "temperature": self._temperature(),
        }
        sys_prompt = _system_prompt(messages)
        if sys_prompt:
            payload["system"] = sys_prompt
        if tools:
            payload["tools"] = _to_anthropic_tools(tools)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        return ("/messages", {}, headers, payload)

    def parse_response(self, data: JSON) -> JSON:
        text_parts, tool_calls = [], []
        for item in data["content"]:
            if item["type"] == "text":
                text_parts.append(item["text"])
            elif item["type"] == "tool_use":
                tool_calls.append(
                    {
                        "id": item["id"],
                        "type": "function",
                        "function": {
                            "name": item["name"],
                            "arguments": json.dumps(item["input"]),
                        },
                    }
                )

        usage = data.get("usage", {})
        normalized_usage = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        }
        stop_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        finish = stop_map.get(data.get("stop_reason"), None)

        return {
            "message": {
                "content": "\n".join(text_parts),
                "tool_calls": tool_calls or None,
                "finish_reason": finish,
            },
            "finish_reason": finish,
            "usage": normalized_usage,
            "model": data.get("model", self._model()),
        }
```

---

## 4 ▪ Create `src/llm/providers/azure_openai.py`

```python
# src/llm/providers/azure_openai.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from ..base import ProviderAdapter

JSON      = Dict[str, Any]
MsgList   = List[Dict[str, Any]]
ToolList  = Optional[List[Dict[str, Any]]]

class AzureOpenAIAdapter(ProviderAdapter):
    def build_request(
        self, messages: MsgList, tools: ToolList
    ) -> Tuple[str, Dict[str, str], Dict[str, str], JSON]:
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
```

---

## 5 ▪ Expose in `src/llm/providers/__init__.py`

```python
from .openai_compat import OpenAICompatibleAdapter
from .anthropic      import AnthropicAdapter
from .azure_openai   import AzureOpenAIAdapter

__all__ = [
    "OpenAICompatibleAdapter",
    "AnthropicAdapter",
    "AzureOpenAIAdapter",
]
```

---

## 6 ▪ Create `src/llm/__init__.py`

```python
from .client import LLMClient  # we will create this next
__all__ = ["LLMClient"]
```

---

## 7 ▪ Create `src/llm/client.py` (replaces old class)

```python
# src/llm/client.py
from __future__ import annotations
import logging, httpx
from typing import Any, Dict, List, Optional, AsyncGenerator
from .base import ProviderAdapter
from .providers import (
    OpenAICompatibleAdapter,
    AnthropicAdapter,
    AzureOpenAIAdapter,
)

JSON = Dict[str, Any]
MsgList = List[Dict[str, Any]]
ToolList = Optional[List[Dict[str, Any]]]


class LLMClient:
    """
    Thin façade: choose adapter, forward calls.
    """

    def __init__(self, full_cfg: Dict[str, Any], api_key: str) -> None:
        self.active_name = full_cfg["active"].lower()
        self.cfg = (full_cfg["providers"] or {}).get(self.active_name, {})
        self.base_url = self.cfg["base_url"].rstrip("/")
        self.api_key = api_key

        self.adapter: ProviderAdapter = self._make_adapter()
        self.http = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)

    # ---------- public ----------
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
    ) -> AsyncGenerator[JSON, None]:
        if not isinstance(self.adapter, OpenAICompatibleAdapter):
            raise RuntimeError(f"Streaming not yet implemented for {self.active_name}")

        path, params, hdrs, payload = self.adapter.build_request(messages, tools)
        payload["stream"] = True

        async with self.http.stream("POST", path, params=params, headers=hdrs, json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    yield httpx.JSONDecoder().decode(data)

    async def close(self): await self.http.aclose()

    # ---------- helpers ----------
    def _make_adapter(self) -> ProviderAdapter:
        if self.active_name in ("openai", "groq", "openrouter"):
            return OpenAICompatibleAdapter(self.cfg, self.api_key)
        if self.active_name == "anthropic":
            return AnthropicAdapter(self.cfg, self.api_key)
        if self.active_name in ("azure", "azure_openai"):
            return AzureOpenAIAdapter(self.cfg, self.api_key)
        raise ValueError(f"Unknown LLM provider: {self.active_name}")
```

---

## 8 ▪ Delete the **old** `LLMClient` in `main.py`

and instead import the new one:

```diff
-from src.main import LLMClient, MCPClient
+from src.llm import LLMClient
+from src.main import MCPClient   # keep MCPClient where it is
```

(You may also physically move `MCPClient` out of `main.py` to its own file, but that’s optional and unrelated.)

---

## 9 ▪ Update other imports

* `chat_service.py` (and any other file) should now import:

```python
from src.llm import LLMClient
```

---

## 10 ▪ (Temporary) streaming guard

Until you implement streaming adapters for Anthropic/Azure, your config should set:

```yaml
chat:
  service:
    streaming:
      enabled: true   # OK for OpenAI / Groq / OpenRouter
      # set to false when you switch to anthropic or azure
```

Or keep the flag `true` but add logic in `ChatService` to fall back on `.chat_once()` when `RuntimeError("Streaming not yet implemented")` bubbles up.

---

## 11 ▪ Environment variables / keys

Create:

| Provider     | Environment variable you’ll export | Header logic (already handled)     |
| ------------ | ---------------------------------- | ---------------------------------- |
| OpenAI       | `OPENAI_API_KEY`                   | `Authorization: Bearer`            |
| Groq         | `GROQ_API_KEY`                     | «same»                             |
| OpenRouter   | `OPENROUTER_API_KEY`               | «same» + *optional* `HTTP‑Referer` |
| Anthropic    | `ANTHROPIC_API_KEY`                | `x-api-key:`                       |
| Azure OpenAI | `AZURE_OPENAI_KEY`                 | `api-key:` + `api-version` param   |

Bind `config.llm_api_key` accordingly (you already have helper code for that).

---

## 12 ▪ Smoke‑test matrix

```bash
# OpenAI
yq -i '.llm.active = "openai"'  config.yaml
python -m src.main

# Groq
yq -i '.llm.active = "groq"'    config.yaml
python -m src.main   # should work immediately

# OpenRouter
yq -i '.llm.active = "openrouter"' config.yaml
python -m src.main

# Anthropic (non‑streaming)
yq -i '.llm.active = "anthropic"' config.yaml
yq -i '.chat.service.streaming.enabled = false' config.yaml
python -m src.main

# Azure (non‑streaming unless you implement)
yq -i '.llm.active = "azure"' config.yaml
yq -i '.chat.service.streaming.enabled = false' config.yaml
python -m src.main
```

---

### What did we actually do?

1. **Isolated provider quirks** in adapter classes—no more `if provider == …` all over the place.
2. `ChatService` remains pristine and agnostic.
3. Adding a new provider = one new adapter file + config stanza.
4. Streaming currently supported for OpenAI‑compatible endpoints; you can copy the same pattern into `AnthropicAdapter` (SSE) or `AzureOpenAIAdapter` later.

---
