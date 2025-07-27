

> **Goal:** Extract the *MCP Resource Catalog & System Prompt Composer* logic from `OpenAIOrchestrator` into a provider‑agnostic service.
> The new service must not reference OpenAI request/response shapes or HTTP clients. It may depend on `mcp.types` and `ToolSchemaManager`.
>
> **Create** a new file at `src/mcp/prompting.py` with the exact contents below.
> **Then** apply the edits to `OpenAIOrchestrator` exactly as in the diff below.
> **Finally**, add the unit test `tests/test_mcp_prompting.py` as provided.

---

#### New file: `src/mcp/prompting.py`

```python
from __future__ import annotations

import logging
from typing import Any

from mcp import types

# Deliberately no imports from provider/LLM code.
# This module is MCP-only and provider-agnostic.

logger = logging.getLogger(__name__)


class MCPResourcePromptService:
    """
    Handles:
      - Cataloging readable MCP resources (availability-aware)
      - Building the final system prompt by embedding resources and prompts

    Dependencies:
      - ToolSchemaManager-like object with:
          list_available_resources() -> list[str]
          read_resource(uri) -> object with .contents
          get_resource_info(uri) -> object with .resource.name (optional)
          list_available_prompts() -> list[str]
          get_prompt_info(name) -> object with .prompt.description (optional)
    """

    def __init__(self, tool_mgr: Any, chat_conf: dict[str, Any]) -> None:
        self.tool_mgr = tool_mgr
        self.chat_conf = chat_conf
        self.resource_catalog: list[str] = []

    async def update_resource_catalog_on_availability(self) -> None:
        """
        Build `self.resource_catalog` with only those resources that currently return content.
        """
        if not self.tool_mgr:
            self.resource_catalog = []
            return

        all_resource_uris = self.tool_mgr.list_available_resources()
        available_uris: list[str] = []

        for uri in all_resource_uris:
            try:
                resource_result = await self.tool_mgr.read_resource(uri)
                if getattr(resource_result, "contents", None):
                    available_uris.append(uri)
            except Exception:
                # Skip unavailable resources silently
                continue

        self.resource_catalog = available_uris
        logger.debug(
            "Updated resource catalog: %d of %d resources are available",
            len(available_uris),
            len(all_resource_uris),
        )

    async def get_available_resources(
        self,
    ) -> dict[str, list[types.TextResourceContents | types.BlobResourceContents]]:
        """
        Return a mapping of URI -> list(contents) for resources that are currently readable.
        Only URIs in `self.resource_catalog` are considered.
        """
        out: dict[
            str, list[types.TextResourceContents | types.BlobResourceContents]
        ] = {}

        if not self.resource_catalog or not self.tool_mgr:
            return out

        for uri in self.resource_catalog:
            try:
                resource_result = await self.tool_mgr.read_resource(uri)
                if resource_result.contents:
                    out[uri] = resource_result.contents
                    logger.debug("Resource %s is available and loaded", uri)
                else:
                    logger.debug("Resource %s has no content, skipping", uri)
            except Exception as e:
                # Log and skip — do not leak broken resources into the prompt.
                logger.warning(
                    "Resource %s is unavailable and will be excluded from system prompt: %s",
                    uri,
                    e,
                )
                continue

        if out:
            logger.info("Including %d available resources in system prompt", len(out))
        else:
            logger.info(
                "No resources are currently available - system prompt will not include resource section"
            )

        return out

    async def make_system_prompt(self) -> str:
        """
        Build the system prompt by embedding:
          - The base system prompt from chat_conf["system_prompt"]
          - The available resources' content (text lines or binary size)
          - The available prompt catalog (name + description)

        This function is stable even if portions of the MCP surface are missing.
        """
        base = str(self.chat_conf.get("system_prompt", "")).rstrip()

        if not self.tool_mgr:
            return base

        # Include available resources
        available_resources = await self.get_available_resources()
        if available_resources:
            base += "\n\n**Available Resources:**"
            for uri, contents in available_resources.items():
                resource_info = self.tool_mgr.get_resource_info(uri)
                name = (
                    getattr(getattr(resource_info, "resource", None), "name", None)
                    or uri
                )
                base += f"\n\n**{name}** ({uri}):"

                for content in contents:
                    if isinstance(content, types.TextResourceContents):
                        lines = (content.text or "").strip().split("\n")
                        for line in lines:
                            base += f"\n{line}"
                    elif isinstance(content, types.BlobResourceContents):
                        base += f"\n[Binary content: {len(content.blob)} bytes]"
                    else:
                        base += f"\n[{type(content).__name__} available]"

        # Include prompt catalog
        try:
            prompt_names = self.tool_mgr.list_available_prompts()
        except Exception:
            prompt_names = []

        if prompt_names:
            prompt_list: list[str] = []
            for pname in prompt_names:
                try:
                    pinfo = self.tool_mgr.get_prompt_info(pname)
                    desc = getattr(getattr(pinfo, "prompt", None), "description", None)
                    desc = desc or "No description available"
                except Exception:
                    desc = "No description available"
                prompt_list.append(f"• **{pname}**: {desc}")

            prompts_text = "\n".join(prompt_list)
            base += (
                f"\n\n**Available Prompts** (use apply_prompt method):\n"
                f"{prompts_text}"
            )

        return base
```

---

#### Edit `OpenAIOrchestrator` to use the new service

1. **Add import** near the top of the file:

```python
from src.mcp.prompting import MCPResourcePromptService
```

2. **Add a field** in `__init__` (after `self._ready = asyncio.Event()`):

```python
self.mcp_prompt: MCPResourcePromptService | None = None
self._system_prompt: str = ""
```

3. **In `initialize()`**, replace the system‑prompt setup block:

**Before:**

```python
# Use connected clients for tool management (empty list is acceptable)
self.tool_mgr = ToolSchemaManager(connected_clients)
await self.tool_mgr.initialize()

# Update resource catalog to only include available resources
await self._update_resource_catalog_on_availability()
self._system_prompt = await self._make_system_prompt()
```

**After:**

```python
# Use connected clients for tool management (empty list is acceptable)
self.tool_mgr = ToolSchemaManager(connected_clients)
await self.tool_mgr.initialize()

# Resource catalog & system prompt (MCP-only, provider-agnostic)
self.mcp_prompt = MCPResourcePromptService(self.tool_mgr, self.chat_conf)
await self.mcp_prompt.update_resource_catalog_on_availability()
self._system_prompt = await self.mcp_prompt.make_system_prompt()
```

4. **Adjust the “ready” log lines** to use the new catalog:

```python
logger.info(
    "OpenAIOrchestrator ready: %d tools, %d resources, %d prompts",
    len(self.tool_mgr.get_mcp_tools()),
    len(self.mcp_prompt.resource_catalog if self.mcp_prompt else []),
    len(self.tool_mgr.list_available_prompts()),
)

logger.info(
    "Resource catalog: %s",
    (self.mcp_prompt.resource_catalog if self.mcp_prompt else []),
)
```

5. **Delete** the following now‑unused methods from `OpenAIOrchestrator`:

* `_make_system_prompt`
* `_get_available_resources`
* `_update_resource_catalog_on_availability`

> Ensure there are **no** remaining references to `self._resource_catalog` anywhere in the class.

**Illustrative diff:**

```diff
--- a/src/openai_orchestrator.py
+++ b/src/openai_orchestrator.py
@@
-from mcp import types
+from mcp import types
+from src.mcp.prompting import MCPResourcePromptService
@@ class OpenAIOrchestrator:
-        self._init_lock = asyncio.Lock()
+        self._init_lock = asyncio.Lock()
         self._ready = asyncio.Event()
+        self.mcp_prompt: MCPResourcePromptService | None = None
+        self._system_prompt: str = ""
@@ async def initialize(self) -> None:
-            # Use connected clients for tool management (empty list is acceptable)
+            # Use connected clients for tool management (empty list is acceptable)
             self.tool_mgr = ToolSchemaManager(connected_clients)
             await self.tool_mgr.initialize()
-
-            # Update resource catalog to only include available resources
-            await self._update_resource_catalog_on_availability()
-            self._system_prompt = await self._make_system_prompt()
+            # Resource catalog & system prompt (MCP-only)
+            self.mcp_prompt = MCPResourcePromptService(self.tool_mgr, self.chat_conf)
+            await self.mcp_prompt.update_resource_catalog_on_availability()
+            self._system_prompt = await self.mcp_prompt.make_system_prompt()
@@
-            logger.info(
-                "OpenAIOrchestrator ready: %d tools, %d resources, %d prompts",
-                len(self.tool_mgr.get_mcp_tools()),
-                len(self._resource_catalog),
-                len(self.tool_mgr.list_available_prompts()),
-            )
+            logger.info(
+                "OpenAIOrchestrator ready: %d tools, %d resources, %d prompts",
+                len(self.tool_mgr.get_mcp_tools()),
+                len(self.mcp_prompt.resource_catalog if self.mcp_prompt else []),
+                len(self.tool_mgr.list_available_prompts()),
+            )
 
-            logger.info("Resource catalog: %s", self._resource_catalog)
+            logger.info(
+                "Resource catalog: %s",
+                (self.mcp_prompt.resource_catalog if self.mcp_prompt else []),
+            )
@@
-    async def _make_system_prompt(self) -> str:
-        ...
-    async def _get_available_resources(self, ...) -> dict[...]:
-        ...
-    async def _update_resource_catalog_on_availability(self) -> None:
-        ...
+    # (Removed: system prompt + resource catalog helpers; now provided by MCPResourcePromptService)
```

---

#### New test: `tests/test_mcp_prompting.py`

```python
import asyncio
import importlib
from types import SimpleNamespace

import pytest

# Module under test
from src.mcp.prompting import MCPResourcePromptService


# --- Test doubles ------------------------------------------------------------

class _DummyText:
    def __init__(self, text: str):
        self.text = text


class _DummyBlob:
    def __init__(self, blob: bytes):
        self.blob = blob


class _DummyReadResult:
    def __init__(self, contents):
        self.contents = contents


class _DummyResInfo:
    def __init__(self, name: str):
        self.resource = SimpleNamespace(name=name)


class FakeToolMgr:
    def __init__(self):
        self._resources = ["uri://a", "uri://b", "uri://c"]  # c will fail
        self._prompts = ["greet"]
        self._prompt_desc = {"greet": "Say hello politely."}
        self._names = {"uri://a": "A_Name", "uri://b": "B_Name", "uri://c": "C_Name"}

    def list_available_resources(self):
        return list(self._resources)

    async def read_resource(self, uri: str):
        if uri == "uri://c":
            raise RuntimeError("simulated failure")
        if uri == "uri://a":
            return _DummyReadResult([_DummyText("Hello\nWorld")])
        if uri == "uri://b":
            return _DummyReadResult([_DummyBlob(b"\x01\x02\x03")])
        return _DummyReadResult([])

    def get_resource_info(self, uri: str):
        return _DummyResInfo(self._names.get(uri, uri))

    def list_available_prompts(self):
        return list(self._prompts)

    def get_prompt_info(self, name: str):
        desc = self._prompt_desc.get(name, "No description available")
        return SimpleNamespace(prompt=SimpleNamespace(description=desc))


@pytest.mark.asyncio
async def test_catalog_and_prompt_building(monkeypatch):
    # Monkeypatch module's `types` to use dummy classes (no real mcp dependency needed)
    mod = importlib.import_module("src.mcp.prompting")

    class _DummyTypes:
        TextResourceContents = _DummyText
        BlobResourceContents = _DummyBlob

    monkeypatch.setattr(mod, "types", _DummyTypes, raising=True)

    chat_conf = {"system_prompt": "SYS"}
    svc = MCPResourcePromptService(FakeToolMgr(), chat_conf)

    # Build catalog with availability filter
    await svc.update_resource_catalog_on_availability()
    assert sorted(svc.resource_catalog) == ["uri://a", "uri://b"]

    # Collect available resources
    available = await svc.get_available_resources()
    assert set(available.keys()) == {"uri://a", "uri://b"}
    assert isinstance(available["uri://a"][0], _DummyText)
    assert isinstance(available["uri://b"][0], _DummyBlob)

    # Build system prompt
    prompt = await svc.make_system_prompt()
    # Base
    assert "SYS" in prompt
    # Resource section
    assert "**A_Name** (uri://a):" in prompt
    assert "Hello" in prompt and "World" in prompt
    assert "[Binary content: 3 bytes]" in prompt
    # Prompt catalog
    assert "**Available Prompts**" in prompt
    assert "**greet**" in prompt
    assert "Say hello politely." in prompt
```

---

#### Acceptance criteria

* `src/mcp/prompting.py` contains **no** provider/LLM code or HTTP calls; it only uses `mcp.types` and a ToolSchemaManager‑like interface.
* `OpenAIOrchestrator.initialize()` now delegates resource cataloging and prompt composition to `MCPResourcePromptService`.
* All references to `self._resource_catalog`, `_make_system_prompt`, `_get_available_resources`, and `_update_resource_catalog_on_availability` are removed from `OpenAIOrchestrator`.
* The new test passes (`pytest -q`), validating:

  * Availability‑filtered catalog (`["uri://a", "uri://b"]`)
  * System prompt includes base text, resource content lines, binary sizes, and prompt catalog entries.

---

