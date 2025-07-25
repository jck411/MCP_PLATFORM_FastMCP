#!/usr/bin/env python3
"""
Chat History Storage Module

This module provides a comprehensive chat event storage system for the MCP Platform.
It manages conversation history, token counting, and persistent storage with multiple
backend implementations.

Key Components:
- ChatEvent: Pydantic model representing individual chat messages, tool calls,
  and system events
- ChatRepository: Protocol defining the interface for chat storage backends
- InMemoryRepo: Fast in-memory storage implementation for development/testing
- JsonlRepo: Persistent JSONL file storage for production use

Features:
- Comprehensive event tracking (user messages, assistant responses, tool calls,
  system updates)
- Token counting and usage tracking for cost monitoring
- Duplicate detection via request IDs
- Conversation-based organization
- Thread-safe operations
- Async/await support
- Backward compatibility exports

The module follows the MCP Platform's standards with:
- Pydantic v2 for data validation and serialization
- Type hints on all functions and methods
- Fail-fast error handling
- Modern Python syntax (union types with |)
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from datetime import datetime
from functools import partial
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from src.history.token_counter import estimate_tokens

# Re-export for backward compatibility
__all__ = [
    "ChatEvent",
    "ChatRepository",
    "InMemoryRepo",
    "JsonlRepo",
    "estimate_tokens",
]

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

    def compute_and_cache_tokens(self) -> int:
        """Compute and cache token count for this event's content."""
        if self.content is None:
            self.token_count = 0
            return 0

        if isinstance(self.content, str):
            text_content = self.content
        elif isinstance(self.content, list):
            # Handle list of Parts (for future extensibility)
            text_parts = []
            for part in self.content:
                if isinstance(part, TextPart):
                    text_parts.append(part.text)
            text_content = " ".join(text_parts)
        else:
            text_content = str(self.content)

        self.token_count = estimate_tokens(text_content)
        return self.token_count

    def ensure_token_count(self) -> int:
        """Ensure token count is computed and return it."""
        if self.token_count is None:
            return self.compute_and_cache_tokens()
        return self.token_count

# ---------- Repository interface ----------

class ChatRepository(Protocol):
    # Returns True if added, False if duplicate
    async def add_event(self, event: ChatEvent) -> bool: ...
    async def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]: ...
    async def last_n_tokens(
        self, conversation_id: str, max_tokens: int
    ) -> list[ChatEvent]: ...
    async def list_conversations(self) -> list[str]: ...
    async def get_event_by_request_id(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None: ...
    async def get_last_assistant_reply_id(
        self, conversation_id: str, user_request_id: str
    ) -> str | None: ...

# ---------- In-memory implementation ----------

class InMemoryRepo(ChatRepository):
    def __init__(self):
        self._by_conv: dict[str, list[ChatEvent]] = {}
        self._lock = threading.Lock()

    async def add_event(self, event: ChatEvent) -> bool:
        with self._lock:
            events = self._by_conv.setdefault(event.conversation_id, [])

            # Check for duplicate by request_id if present
            request_id = event.extra.get("request_id")
            if request_id:
                for existing_event in events:
                    if existing_event.extra.get("request_id") == request_id:
                        return False  # Duplicate found, don't add

            events.append(event)
            return True  # Successfully added

    async def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]:
        with self._lock:
            events = self._by_conv.get(conversation_id, [])
            return events[-limit:] if limit is not None else list(events)

    async def last_n_tokens(
        self, conversation_id: str, max_tokens: int
    ) -> list[ChatEvent]:
        with self._lock:
            events = self._by_conv.get(conversation_id, [])
            acc: list[ChatEvent] = []
            total = 0
            for ev in reversed(events):
                # Ensure token count is computed
                tok = ev.ensure_token_count()
                if total + tok > max_tokens:
                    break
                acc.append(ev)
                total += tok
            return list(reversed(acc))

    async def list_conversations(self) -> list[str]:
        with self._lock:
            return list(self._by_conv.keys())

    async def get_event_by_request_id(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None:
        with self._lock:
            events = self._by_conv.get(conversation_id, [])
            for event in events:
                if event.extra.get("request_id") == request_id:
                    return event
            return None

    async def get_last_assistant_reply_id(
        self, conversation_id: str, user_request_id: str
    ) -> str | None:
        with self._lock:
            events = self._by_conv.get(conversation_id, [])
            for event in reversed(events):
                if (
                    event.type == "assistant_message"
                    and event.extra.get("user_request_id") == user_request_id
                ):
                    return event.id
            return None

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

    def _append_sync(self, event: ChatEvent) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            data = json.dumps(
                event.model_dump(mode="json"), ensure_ascii=False
            )
            f.write(data + "\n")
            f.flush()

    def _add_event_sync(self, event: ChatEvent) -> bool:
        with self._lock:
            events = self._by_conv.setdefault(event.conversation_id, [])

            # Check for duplicate by request_id if present
            request_id = event.extra.get("request_id")
            if request_id:
                for existing_event in events:
                    if existing_event.extra.get("request_id") == request_id:
                        return False  # Duplicate found, don't add

            events.append(event)
            self._append_sync(event)
            return True  # Successfully added

    async def add_event(self, event: ChatEvent) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._add_event_sync, event))

    async def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]:
        with self._lock:
            events = self._by_conv.get(conversation_id, [])
            return events[-limit:] if limit is not None else list(events)

    async def last_n_tokens(
        self, conversation_id: str, max_tokens: int
    ) -> list[ChatEvent]:
        with self._lock:
            events = self._by_conv.get(conversation_id, [])
            acc: list[ChatEvent] = []
            total = 0
            for ev in reversed(events):
                # Ensure token count is computed
                tok = ev.ensure_token_count()
                if total + tok > max_tokens:
                    break
                acc.append(ev)
                total += tok
            return list(reversed(acc))

    async def list_conversations(self) -> list[str]:
        with self._lock:
            return list(self._by_conv.keys())

    async def get_event_by_request_id(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None:
        with self._lock:
            events = self._by_conv.get(conversation_id, [])
            for event in events:
                if event.extra.get("request_id") == request_id:
                    return event
            return None

    async def get_last_assistant_reply_id(
        self, conversation_id: str, user_request_id: str
    ) -> str | None:
        with self._lock:
            events = self._by_conv.get(conversation_id, [])
            for event in reversed(events):
                if (
                    event.type == "assistant_message"
                    and event.extra.get("user_request_id") == user_request_id
                ):
                    return event.id
            return None

# ---------- Tiny demo ----------

if __name__ == "__main__":
    import asyncio
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    async def main():
        repo: ChatRepository = JsonlRepo("events.jsonl")
        conv_id = str(uuid.uuid4())

        user_ev = ChatEvent(
            conversation_id=conv_id,
            type="user_message",
            role="user",
            content="Hello!",
            token_count=2
        )
        await repo.add_event(user_ev)

        asst_ev = ChatEvent(
            conversation_id=conv_id,
            type="assistant_message",
            role="assistant",
            content="Hi! How can I help?",
            token_count=5,
            provider="openai",
            model="gpt-4o-mini"
        )
        await repo.add_event(asst_ev)

        logger.info(f"Conversation ID: {conv_id}")
        events = await repo.get_events(conv_id)
        for ev in events:
            logger.info(f"- {ev.type} {ev.role} {ev.content}")

    asyncio.run(main())
