#!/usr/bin/env python3
"""
Chat History Storage Module

This module provides a comprehensive chat event storage system for the MCP Platform.
It manages conversation history and persistent storage with multiple
backend implementations.

Key Components:
- ChatEvent: Pydantic model representing individual chat messages, tool calls,
  and system events
- ChatRepository: Protocol defining the interface for chat storage backends
- JsonlRepo: Persistent JSONL file storage for production use

Features:
- Comprehensive event tracking (user messages, assistant responses, tool calls,
  system updates)
- Usage tracking for cost monitoring
- Duplicate detection via request IDs
- Conversation-based organization
- Thread-safe operations
- Async/await support

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
from datetime import UTC, datetime
from functools import partial
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

# ---------- Canonical models (Pydantic v2) ----------

Role = Literal["system", "user", "assistant", "tool"]

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0  # For reasoning models like o3

class ChatEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    type: Literal[
        "user_message",
        "assistant_message",
        "tool_call",
        "tool_result",
        "system_update",
        "meta"
    ]
    role: Role | None = None
    content: str | None = None
    tool_calls: list[ToolCall] = []
    usage: Usage | None = None
    provider: str | None = None
    model: str | None = None
    stop_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    extra: dict[str, Any] = Field(default_factory=dict)

# ---------- Repository interface ----------

class ChatRepository(Protocol):
    # Returns True if added, False if duplicate
    async def add_event(self, event: ChatEvent) -> bool: ...
    async def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]: ...
    async def list_conversations(self) -> list[str]: ...
    async def get_event_by_request_id(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None: ...
    async def get_last_assistant_reply_id(
        self, conversation_id: str, user_request_id: str
    ) -> str | None: ...

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
