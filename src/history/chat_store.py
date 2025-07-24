#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

# ---------- Canonical models (Pydantic v2) ----------

Role = Literal["system", "user", "assistant", "tool"]

class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str

Part = TextPart  # extend later

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    schema_version: int = 1
    type: Literal[
        "user_message",
        "assistant_message",
        "tool_call",
        "tool_result",
        "system_update",
        "meta"
    ]
    role: Role | None = None
    content: str | list[Part] | None = None
    tool_calls: list[ToolCall] = []
    usage: Usage | None = None
    provider: str | None = None
    model: str | None = None
    stop_reason: str | None = None
    token_count: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)
    raw: Any | None = None  # keep small; move big things elsewhere later

# ---------- Token estimation helper ----------

def estimate_tokens(text: str) -> int:
    # crude placeholder; replace with tiktoken or provider counter later
    return max(1, int(len(text.split()) / 0.75))

# ---------- Repository interface ----------

class ChatRepository(Protocol):
    def add_event(self, event: ChatEvent) -> None: ...
    def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]: ...
    def last_n_tokens(
        self, conversation_id: str, max_tokens: int
    ) -> list[ChatEvent]: ...
    def list_conversations(self) -> list[str]: ...

# ---------- In-memory implementation ----------

class InMemoryRepo(ChatRepository):
    def __init__(self):
        self._by_conv: dict[str, list[ChatEvent]] = {}
        self._lock = threading.Lock()

    def add_event(self, event: ChatEvent) -> None:
        with self._lock:
            self._by_conv.setdefault(event.conversation_id, []).append(event)

    def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]:
        with self._lock:
            events = self._by_conv.get(conversation_id, [])
            return events[-limit:] if limit is not None else list(events)

    def last_n_tokens(
        self, conversation_id: str, max_tokens: int
    ) -> list[ChatEvent]:
        with self._lock:
            events = self._by_conv.get(conversation_id, [])
            acc: list[ChatEvent] = []
            total = 0
            for ev in reversed(events):
                tok = ev.token_count or 0
                if total + tok > max_tokens:
                    break
                acc.append(ev)
                total += tok
            return list(reversed(acc))

    def list_conversations(self) -> list[str]:
        with self._lock:
            return list(self._by_conv.keys())

# ---------- JSONL (append-only) implementation ----------

class JsonlRepo(ChatRepository):
    """
    Simple, dev-friendly persistence: one global JSONL file.
    If you prefer, create one file per conversation (easy tweak).
    """
    def __init__(self, path: str = "events.jsonl"):
        self.path = path
        self._lock = threading.Lock()
        self._by_conv: dict[str, list[ChatEvent]] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            for file_line in f:
                stripped_line = file_line.strip()
                if not stripped_line:
                    continue
                data = json.loads(stripped_line)
                ev = ChatEvent.model_validate(data)
                self._by_conv.setdefault(ev.conversation_id, []).append(ev)

    def _append(self, event: ChatEvent) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            data = json.dumps(
                event.model_dump(mode="json"), ensure_ascii=False
            )
            f.write(data + "\n")
            f.flush()

    def add_event(self, event: ChatEvent) -> None:
        with self._lock:
            self._by_conv.setdefault(event.conversation_id, []).append(event)
            self._append(event)

    def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]:
        with self._lock:
            events = self._by_conv.get(conversation_id, [])
            return events[-limit:] if limit is not None else list(events)

    def last_n_tokens(
        self, conversation_id: str, max_tokens: int
    ) -> list[ChatEvent]:
        with self._lock:
            events = self._by_conv.get(conversation_id, [])
            acc: list[ChatEvent] = []
            total = 0
            for ev in reversed(events):
                tok = ev.token_count or 0
                if total + tok > max_tokens:
                    break
                acc.append(ev)
                total += tok
            return list(reversed(acc))

    def list_conversations(self) -> list[str]:
        with self._lock:
            return list(self._by_conv.keys())

# ---------- Tiny demo ----------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    repo: ChatRepository = JsonlRepo("events.jsonl")
    conv_id = str(uuid.uuid4())

    user_ev = ChatEvent(
        conversation_id=conv_id,
        type="user_message",
        role="user",
        content="Hello!",
        token_count=2
    )
    repo.add_event(user_ev)

    asst_ev = ChatEvent(
        conversation_id=conv_id,
        type="assistant_message",
        role="assistant",
        content="Hi! How can I help?",
        token_count=5,
        provider="openai",
        model="gpt-4o-mini"
    )
    repo.add_event(asst_ev)

    logger.info(f"Conversation ID: {conv_id}")
    for ev in repo.get_events(conv_id):
        logger.info(f"- {ev.type} {ev.role} {ev.content}")
