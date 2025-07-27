import importlib
from types import SimpleNamespace

import pytest

from src.mcp_services.prompting import MCPResourcePromptService

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
    mod = importlib.import_module("src.mcp_services.prompting")

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
