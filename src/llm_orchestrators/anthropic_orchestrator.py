"""
Anthropic Orchestrator for MCP Platform

This module handles the business logic for chat sessions with integrated
Anthropic API compatibility:
- Conversation management with simple default prompts
- Tool orchestration
- MCP client coordination
- Direct Anthropic API interactions (no adapter layer)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import httpx
from mcp import types

from src.history.chat_store import ChatEvent, ChatRepository, ToolCall, Usage
from src.history.persistence import ConversationPersistenceService
from src.mcp_services.prompting import MCPResourcePromptService
from src.schema_manager import SchemaManager

from .http_resilience import ResilientHttpClient, create_http_config_from_dict

if TYPE_CHECKING:
    from src.main import MCPClient  # pragma: no cover

logger = logging.getLogger(__name__)

# Maximum number of tool call hops to prevent infinite recursion
MAX_TOOL_HOPS = 8

# Maximum size for a single text delta to prevent memory issues (10KB)
MAX_TEXT_DELTA_SIZE = 10_000


# -----------------------------
# Exceptions
# -----------------------------
class OrchestratorError(Exception):
    """Base class for orchestrator errors."""


class InitializationError(OrchestratorError):
    """Raised when orchestrator is used before initialization."""


class TransportError(OrchestratorError):
    """Raised when HTTP/stream transport contracts are violated."""


class UpstreamProtocolError(OrchestratorError):
    """Raised when upstream provider protocol is violated/unexpected."""


class ToolExecutionError(OrchestratorError):
    """Raised for tool execution / argument parse failures."""


# -----------------------------
# Data Models
# -----------------------------
@dataclass(slots=True)
class ChatMessage:
    """
    Represents a chat message with metadata.
    """

    mtype: Literal["text", "error"]
    content: str
    meta: dict[str, Any]

    # Backwards accessor parity with your prior code
    @property
    def type(self) -> str:  # pragma: no cover
        return self.mtype


class AnthropicOrchestrator:
    """
    Conversation orchestrator with integrated Anthropic API handling.
    1. Takes your message
    2. Figures out what tools might be needed
    3. Asks the AI to respond (and use tools if needed) via Anthropic API
    4. Sends you back the response
    """

    def __init__(
        self,
        clients: list[MCPClient],
        llm_config: dict[str, Any],
        config: dict[str, Any],
        repo: ChatRepository,
        max_history_messages: int = 50,
    ):
        self.clients = clients
        self.llm_config = llm_config
        self.config = config
        self.repo = repo
        self.persist = ConversationPersistenceService(repo)
        self.max_history_messages = max_history_messages
        self.chat_conf = config.get("chat", {}).get("service", {}) or {}
        self.tool_mgr: SchemaManager | None = None
        self._init_lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self.mcp_prompt: MCPResourcePromptService | None = None
        self._system_prompt: str = ""

        # Establish provider fields first
        self.active_name = (llm_config.get("active") or "").lower()
        providers = llm_config.get("providers") or {}
        self.provider_cfg = providers.get(self.active_name, {}) or {}

        # Extract API key and base URL
        self.api_key = self._extract_api_key()
        self.base_url = (self.provider_cfg.get("base_url") or "").rstrip("/")

        # Validate required configuration
        if not self.api_key:
            raise InitializationError(
                f"Missing API key: set {self.active_name.upper()}_API_KEY or "
                f"providers['{self.active_name}']['api_key']"
            )
        if not self.base_url:
            raise InitializationError("Missing base_url in provider config")

        # Create HTTP resilience configuration
        http_config = create_http_config_from_dict(llm_config)

        # Override timeout if specified in provider config
        if "timeout" in self.provider_cfg:
            http_config.timeout = self.provider_cfg["timeout"]

        # Configure Anthropic version from provider config, fallback to default
        anthropic_version = self.provider_cfg.get("anthropic_version", "2023-06-01")

        # Configure resilient HTTP client with Anthropic-specific headers
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": anthropic_version,
            "content-type": "application/json",
            "User-Agent": "MCP-Platform/1.0",
        }

        self.http_client = ResilientHttpClient(
            base_url=self.base_url,
            headers=headers,
            config=http_config,
        )

    def _extract_api_key(self) -> str:
        """Extract API key from env or provider config; raise if absent."""
        provider_name = (self.llm_config.get("active") or "").upper()
        env_key = f"{provider_name}_API_KEY"
        return os.getenv(env_key) or self.provider_cfg.get("api_key", "") or ""

    @property
    def model(self) -> str:
        """Get the model name from the current configuration."""
        return self.provider_cfg.get("model", "")

    def _system_prompt_from_messages(
        self, messages: list[dict[str, Any]]
    ) -> str | None:
        """Extract the most recent system prompt from messages if present."""
        for m in messages:
            if m.get("role") == "system":
                return m.get("content") or None
        return None

    @staticmethod
    def _normalize_text_content_to_blocks(content: Any) -> list[dict[str, Any]]:
        """
        Ensure assistant/user content is a list of content blocks.
        Accepts either a plain string or an existing list of blocks.
        """
        if content is None:
            return []
        if isinstance(content, list):
            return content
        # Fallback: treat as text
        return [{"type": "text", "text": str(content)}]

    def _to_anthropic_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert OpenAI-style messages (plus our tool-call shape) to Anthropic format.

        Rules:
          - System messages are dropped here; they are sent via payload["system"].
          - Tool results are turned into user-side `tool_result` blocks attached
            to the next/last user message.
          - Assistant messages always use a list of content blocks.
        """
        out: list[dict[str, Any]] = []

        for m in messages:
            role = m.get("role")
            if role == "system":
                # Exclude; handled by payload["system"]
                continue

            if role == "tool":
                # Convert to a tool_result block and attach to the most recent user
                # message, or create a new user message if needed.
                tool_call_id = m.get("tool_call_id")
                content_text = m.get("content", "")
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": (
                        content_text
                        if isinstance(content_text, str)
                        else json.dumps(content_text)
                    ),
                }

                if out and out[-1]["role"] == "user":
                    if isinstance(out[-1]["content"], list):
                        out[-1]["content"].append(tool_result_block)
                    else:
                        out[-1]["content"] = [
                            {"type": "text", "text": str(out[-1]["content"])},
                            tool_result_block,
                        ]
                else:
                    out.append({"role": "user", "content": [tool_result_block]})
                continue

            if role == "assistant":
                content_blocks: list[dict[str, Any]] = []
                if m.get("content"):
                    content_blocks.extend(
                        self._normalize_text_content_to_blocks(m.get("content"))
                    )

                tool_calls = m.get("tool_calls") or []
                for call in tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": call["function"]["name"],
                            # Note: Anthropic expects structured input;
                            # we store as parsed JSON
                            "input": json.loads(call["function"]["arguments"]),
                        }
                    )

                out.append({"role": "assistant", "content": content_blocks})
                continue

            # user or other roles -> normalize to blocks if text, pass through
            if role == "user":
                out.append(
                    {
                        "role": "user",
                        "content": self._normalize_text_content_to_blocks(
                            m.get("content")
                        ),
                    }
                )
            else:
                # Fallback: pass through (rare)
                out.append({"role": role, "content": m.get("content", "")})
        return out

    def _to_anthropic_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert MCP tools to Anthropic format."""
        res = []
        for t in tools or []:
            res.append(
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t["inputSchema"],
                }
            )
        return res

    async def initialize(self) -> None:
        """Initialize the orchestrator and all MCP clients."""
        async with self._init_lock:
            if self._ready.is_set():
                return

            # Connect to all clients and collect results
            connection_results = await asyncio.gather(
                *(c.connect() for c in self.clients), return_exceptions=True
            )

            # Filter out only successfully connected clients
            connected_clients = []
            for i, result in enumerate(connection_results):
                if isinstance(result, Exception):
                    logger.warning(
                        "Client '%s' failed to connect: %s",
                        self.clients[i].name,
                        result,
                    )
                else:
                    connected_clients.append(self.clients[i])

            if not connected_clients:
                logger.warning(
                    "No MCP clients connected - running with basic functionality"
                )
            else:
                logger.info(
                    "Successfully connected to %d out of %d MCP clients",
                    len(connected_clients),
                    len(self.clients),
                )

            # Use connected clients for tool management (empty list is acceptable)
            self.tool_mgr = SchemaManager(connected_clients)
            await self.tool_mgr.initialize()

            # Resource catalog & system prompt (MCP-only, provider-agnostic)
            self.mcp_prompt = MCPResourcePromptService(self.tool_mgr, self.chat_conf)
            await self.mcp_prompt.update_resource_catalog_on_availability()
            self._system_prompt = await self.mcp_prompt.make_system_prompt()

            logger.info(
                "AnthropicOrchestrator ready: %d tools, %d resources, %d prompts",
                len(self.tool_mgr.get_mcp_tools()),
                len(self.mcp_prompt.resource_catalog if self.mcp_prompt else []),
                len(self.tool_mgr.list_available_prompts()),
            )

            logger.info(
                "Resource catalog: %s",
                (self.mcp_prompt.resource_catalog if self.mcp_prompt else []),
            )

            # Configurable system prompt logging
            if self.chat_conf.get("logging", {}).get("system_prompt", True):
                logger.info("System prompt being used:\n%s", self._system_prompt)
            else:
                logger.debug("System prompt logging disabled in configuration")

            self._ready.set()

    def _build_anthropic_request(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Build Anthropic API request payload."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_anthropic_messages(messages),
            "max_tokens": self.provider_cfg.get("max_tokens", 2048),
            "temperature": self.provider_cfg.get("temperature", 0.1),
        }

        # Extract system prompt from messages (most recent wins)
        sys_prompt = self._system_prompt_from_messages(messages)
        if sys_prompt:
            payload["system"] = sys_prompt

        if tools:
            payload["tools"] = self._to_anthropic_tools(tools)

        return payload

    def _parse_anthropic_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Parse Anthropic API response."""
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for item in data.get("content", []):
            if item.get("type") == "text":
                text_content = item.get("text", "")
                if text_content:
                    text_parts.append(text_content)
            elif item.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": item["id"],
                        "type": "function",
                        "function": {
                            "name": item["name"],
                            "arguments": json.dumps(item.get("input", {})),
                        },
                    }
                )

        usage = data.get("usage", {})
        normalized_usage = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0)
            + usage.get("output_tokens", 0),
        }

        stop_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        stop_reason = data.get("stop_reason")
        finish_reason = stop_map.get(stop_reason) if stop_reason else None

        return {
            "message": {
                "content": "\n".join(text_parts) if text_parts else None,
                "tool_calls": tool_calls or None,
                "finish_reason": finish_reason,
                "anthropic_stop_reason": stop_reason,  # keep original for debugging
            },
            "finish_reason": finish_reason,
            "usage": normalized_usage,
            "model": data.get("model", self.model),
        }

    async def _call_anthropic_api(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Make direct Anthropic API call with retry logic."""
        payload = self._build_anthropic_request(messages, tools)
        response = await self.http_client.post_json("/messages", payload)

        # For non-streaming, we expect an httpx.Response
        if isinstance(response, httpx.Response):
            try:
                return self._parse_anthropic_response(response.json())
            except Exception as e:  # pragma: no cover
                raise UpstreamProtocolError(
                    f"Invalid JSON in upstream response: {e}"
                ) from e

        raise TransportError("Expected httpx.Response for non-streaming request")

    async def _stream_anthropic_api(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream responses from Anthropic API with retry logic.

        Contract:
          - The HTTP client should return an async-iterable of SSE/event dicts
            when stream=True.
          - Each item is either a dict (already parsed) or a string SSE line
            "data: {...}".
        """
        payload = self._build_anthropic_request(messages, tools)
        payload["stream"] = True

        stream_obj = await self.http_client.post_json("/messages", payload, stream=True)

        # path 1: some clients improperly return a Response even for stream=True
        if isinstance(stream_obj, httpx.Response):  # pragma: no cover
            raise TransportError(
                "Got httpx.Response for streaming request; async iterator expected"
            )

        # path 2: async iterator contract
        if hasattr(stream_obj, "__aiter__"):
            async for chunk in stream_obj:  # type: ignore[assignment]
                try:
                    # If chunk comes as dict -> yield the 'data' field or
                    # the dict itself
                    if isinstance(chunk, dict):
                        yield chunk.get("data", chunk)
                        continue

                    # If chunk comes as SSE line string
                    if isinstance(chunk, str) and chunk.startswith("data: "):
                        data_str = chunk[6:].strip()
                        if data_str and data_str != "[DONE]":
                            with contextlib.suppress(json.JSONDecodeError):
                                yield json.loads(data_str)
                        continue

                    # Unknown type: ignore, but log at debug
                    logger.debug("Ignoring unknown stream chunk type: %r", type(chunk))
                except Exception as e:  # Robust to malformed chunks
                    logger.warning("Skipping malformed SSE chunk: %s", e)
                    continue
            return

        # Otherwise: we don't know how to read this stream
        raise TransportError("Streaming transport did not return an async iterator")

    async def _build_conversation_history(
        self, conversation_id: str
    ) -> list[dict[str, Any]]:
        """
        Build conversation history including tool calls and results.

        Canonical transcript representation used for both streaming and
        non-streaming paths.
        """
        events = await self.repo.get_events(conversation_id, self.max_history_messages)
        conv: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt}
        ]

        for ev in events:
            if ev.type in ("user_message", "assistant_message", "system_update"):
                conv.append({"role": ev.role, "content": ev.content})

            elif ev.type == "tool_call" and ev.tool_calls:
                tool_calls = []
                for tc in ev.tool_calls:
                    tool_calls.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                    )
                conv.append(
                    {"role": "assistant", "content": None, "tool_calls": tool_calls}
                )

            elif ev.type == "tool_result":
                tool_call_id = ev.extra.get("tool_call_id") if ev.extra else None
                if tool_call_id:
                    conv.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": ev.content or "",
                        }
                    )

        return conv

    async def process_message(
        self,
        conversation_id: str,
        user_msg: str,
        request_id: str,
    ) -> AsyncGenerator[ChatMessage]:
        """
        Process a user message with streaming response.
        Follows same persistence pattern as chat_once for consistency.
        """
        await self._ready.wait()

        if not self.tool_mgr:
            raise InitializationError("Tool manager not initialized")

        # Check for existing response to prevent double-processing
        existing_response = await self.persist.get_existing_assistant_response(
            conversation_id, request_id
        )
        if existing_response:
            logger.info("Returning cached response for request_id: %s", request_id)
            cached_text = self.persist.get_cached_text(existing_response)
            if cached_text:
                yield ChatMessage("text", cached_text, {"type": "cached"})
            return

        # 1) Persist user message and check if we should continue
        should_continue = await self.persist.ensure_user_message_persisted(
            conversation_id, user_msg, request_id
        )
        if not should_continue:
            # User message existed and assistant response found
            existing_response = await self.persist.get_existing_assistant_response(
                conversation_id, request_id
            )
            if existing_response:
                cached_text = self.persist.get_cached_text(existing_response)
                if cached_text:
                    yield ChatMessage("text", cached_text, {"type": "cached"})
                return

        # 2) Build conversation history from persistent store (canonical)
        conv = await self._build_conversation_history(conversation_id)

        # 3) Generate streaming response with tool handling
        async for msg in self._process_streaming_with_tools(
            conv, conversation_id, request_id
        ):
            yield msg

    async def _process_streaming_with_tools(  # noqa: PLR0912, PLR0915
        self, conv: list[dict[str, Any]], conversation_id: str, request_id: str
    ) -> AsyncGenerator[ChatMessage]:
        """Handle the streaming response generation and tool execution."""
        if not self.tool_mgr:
            raise InitializationError("Tool manager not initialized")

        tools_payload = self.tool_mgr.get_mcp_tools()
        full_assistant_content = ""
        assistant_msg: dict[str, Any] = {}

        try:
            # Initial response - stream and collect
            async for chunk in self._stream_llm_response(conv, tools_payload):
                if isinstance(chunk, ChatMessage):
                    # User-facing text deltas
                    yield chunk
                    if chunk.mtype == "text":
                        full_assistant_content += chunk.content
                else:
                    assistant_msg = chunk
        except asyncio.CancelledError:
            # Respect cancellation and attempt minimal cleanup semantics if needed
            logger.info("Streaming cancelled by upstream")
            raise
        except Exception as e:
            logger.error("Streaming error: %s", e)
            yield ChatMessage(
                "error",
                f"Streaming interrupted: {e!s}",
                {"type": "error", "recoverable": True},
            )

        # Log LLM reply if configured (streaming usage unknown)
        reply_data = {"message": assistant_msg, "usage": None, "model": self.model}
        self._log_llm_reply(reply_data, "Streaming initial response")

        # Handle tool calls in a loop with cycle detection
        hops = 0
        seen_calls: set[tuple[str, str]] = set()

        while assistant_msg.get("tool_calls"):
            calls: list[dict[str, Any]] = assistant_msg["tool_calls"]

            if hops >= MAX_TOOL_HOPS:
                logger.warning(
                    "Maximum tool hops (%d) reached, stopping recursion", MAX_TOOL_HOPS
                )
                error_msg = (
                    f"⚠️ Reached maximum tool call limit ({MAX_TOOL_HOPS}). "
                    f"Stopping to prevent infinite recursion."
                )
                yield ChatMessage(
                    "text", error_msg, {"finish_reason": "tool_limit_reached"}
                )
                full_assistant_content += error_msg
                break

            # Add assistant message to conversation
            conv.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.get("content") or "",
                    "tool_calls": calls,
                }
            )

            # Execute tool calls (with parse guards & persistence)
            await self._execute_tool_calls(
                conv, calls, conversation_id, request_id, seen_calls
            )

            # Get follow-up response - stream and collect
            assistant_msg = {}
            try:
                async for chunk in self._stream_llm_response(conv, tools_payload):
                    if isinstance(chunk, ChatMessage):
                        yield chunk
                        if chunk.mtype == "text":
                            full_assistant_content += chunk.content
                    else:
                        assistant_msg = chunk
            except asyncio.CancelledError:
                logger.info("Streaming cancelled by upstream during tool follow-up")
                raise
            except Exception as e:
                logger.error("Streaming error during tool follow-up: %s", e)
                yield ChatMessage(
                    "error",
                    f"Streaming interrupted: {e!s}",
                    {"type": "error", "recoverable": True},
                )

            # Log LLM reply if configured
            reply_data = {"message": assistant_msg, "usage": None, "model": self.model}
            self._log_llm_reply(
                reply_data, f"Streaming tool follow-up (hop {hops + 1})"
            )

            hops += 1

        # 4) Persist final assistant message
        await self.persist.persist_final_assistant_message(
            conversation_id, full_assistant_content, request_id
        )

    def _handle_stream_chunk(
        self,
        chunk: dict[str, Any],
        message_buffer: str,
        current_tool_calls: list[dict[str, Any]],
        current_tool_use: dict[str, Any] | None,
        finish_reason: str | None,
    ) -> tuple[
        str,
        list[dict[str, Any]],
        dict[str, Any] | None,
        str | None,
        str | None,
    ]:
        """Handle a single chunk from the Anthropic streaming response."""
        text_delta: str | None = None
        chunk_type = chunk.get("type")

        if chunk_type == "message_start":
            pass

        elif chunk_type == "content_block_start":
            block = chunk.get("content_block", {})
            if block.get("type") == "tool_use":
                current_tool_use = {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {"name": block.get("name", ""), "arguments": ""},
                }

        elif chunk_type == "content_block_delta":
            delta = chunk.get("delta", {})
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text_delta = delta.get("text", "")
                # Buffering handled by _stream_llm_response
            elif delta_type == "input_json_delta" and current_tool_use:
                # Accumulate raw JSON string fragments; errors handled on parse
                partial_json = delta.get("partial_json", "")
                current_tool_use["function"]["arguments"] += partial_json

        elif chunk_type == "content_block_stop":
            if current_tool_use:
                current_tool_calls.append(current_tool_use)
                current_tool_use = None

        elif chunk_type == "message_delta":
            delta = chunk.get("delta", {})
            if "stop_reason" in delta:
                stop_map = {
                    "end_turn": "stop",
                    "tool_use": "tool_calls",
                    "max_tokens": "length",
                    "stop_sequence": "stop",
                }
                finish_reason = stop_map.get(delta["stop_reason"])

        elif chunk_type == "message_stop":
            # caller breaks the loop when it sees this
            pass

        return (
            message_buffer,
            current_tool_calls,
            current_tool_use,
            finish_reason,
            text_delta,
        )

    def _validate_text_delta(self, text_delta: str) -> bool:
        """Validate text delta before immediate yielding."""
        if not isinstance(text_delta, str):
            logger.warning("Invalid text delta type: %s", type(text_delta))
            return False
        if len(text_delta) > MAX_TEXT_DELTA_SIZE:
            logger.warning("Text delta too large: %d chars", len(text_delta))
            return False
        return True

    def _should_accumulate_buffer(self) -> bool:
        """Whether to accumulate the full message buffer (needed for
        persistence & tools)."""
        return True

    async def _stream_llm_response(  # noqa: PLR0912
        self, conv: list[dict[str, Any]], tools_payload: list[dict[str, Any]]
    ) -> AsyncGenerator[ChatMessage | dict[str, Any]]:
        """Stream response from LLM and yield chunks to user, ending with the
        final assistant message dict."""
        message_buffer = ""
        current_tool_calls: list[dict[str, Any]] = []
        finish_reason: str | None = None
        current_tool_use: dict[str, Any] | None = None

        try:
            async for chunk in self._stream_anthropic_api(conv, tools_payload):
                chunk_type = chunk.get("type")

                # Fast path: text deltas, validate + yield
                if chunk_type == "content_block_delta":
                    delta = chunk.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text_delta = delta.get("text", "")
                        if text_delta and self._validate_text_delta(text_delta):
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(
                                    "Yielding text delta: %r...", text_delta[:50]
                                )
                            if self._should_accumulate_buffer():
                                message_buffer += text_delta
                            yield ChatMessage("text", text_delta, {"type": "delta"})
                            continue
                        elif text_delta:
                            logger.warning(
                                "Skipping invalid text delta: %r", text_delta[:100]
                            )
                            continue

                # Other chunk handling
                (
                    message_buffer,
                    current_tool_calls,
                    current_tool_use,
                    finish_reason,
                    _,
                ) = self._handle_stream_chunk(
                    chunk,
                    message_buffer,
                    current_tool_calls,
                    current_tool_use,
                    finish_reason,
                )

                if chunk_type == "message_stop":
                    break

        except asyncio.CancelledError:
            logger.info("Streaming cancelled")
            raise
        except Exception as e:
            logger.error("Streaming error: %s", e)
            yield ChatMessage(
                "error",
                f"Streaming interrupted: {e!s}",
                {"type": "error", "recoverable": True},
            )

        # Final assistant message for internal control-flow
        assistant_message = {
            "content": message_buffer or None,
            "tool_calls": current_tool_calls if current_tool_calls else None,
            "finish_reason": finish_reason,
        }

        if assistant_message.get("tool_calls"):
            logger.info("Tool calls detected: %d", len(assistant_message["tool_calls"]))
            for i, call in enumerate(assistant_message["tool_calls"]):
                func_name = call["function"]["name"]
                func_args = call["function"]["arguments"]
                logger.info("  Tool %d: %s with args: %s", i + 1, func_name, func_args)
        else:
            logger.info("No tool calls detected in response")
            logger.debug("current_tool_calls buffer: %s", current_tool_calls)
            logger.debug("finish_reason: %s", finish_reason)

        yield assistant_message

    async def _execute_tool_calls(
        self,
        conv: list[dict[str, Any]],
        calls: list[dict[str, Any]],
        conversation_id: str,
        request_id: str,
        seen_calls: set[tuple[str, str]],
    ) -> None:
        """
        Execute tool calls and add results to conversation and history.

        Includes:
          - Persistence of tool_call and tool_result events
          - Cycle detection via (tool_name, args_json)
          - Graceful handling of invalid JSON args from streamed deltas
        """
        if not self.tool_mgr:
            raise InitializationError("Tool manager not initialized")

        logger.info("Executing %d tool calls", len(calls))

        for call in calls:
            tool_name = call["function"]["name"]
            raw_args = call["function"]["arguments"] or "{}"

            # Try to parse arguments; on failure, surface the error to the model
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                err_text = (
                    f"[tool-args-error] Failed to parse JSON arguments for "
                    f"'{tool_name}': {e}"
                )
                logger.warning(err_text)
                # Persist the tool call with raw args for audit
                tool_call_event = ChatEvent(
                    conversation_id=conversation_id,
                    type="tool_call",
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id=call["id"],
                            name=tool_name,
                            arguments={"__raw__": raw_args},
                        )
                    ],
                    extra={"user_request_id": request_id, "tool_call_id": call["id"]},
                )
                await self.repo.add_event(tool_call_event)

                # Persist and add an explicit tool_result error so the model can recover
                tool_result_content = json.dumps(
                    {"error": "invalid_tool_arguments", "detail": str(e)}
                )
                tool_result_event = ChatEvent(
                    conversation_id=conversation_id,
                    type="tool_result",
                    role="tool",
                    content=tool_result_content,
                    extra={
                        "user_request_id": request_id,
                        "tool_call_id": call["id"],
                        "tool_name": tool_name,
                    },
                )
                await self.repo.add_event(tool_result_event)

                conv.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": tool_result_content,
                    }
                )
                continue

            # Cycle detection: avoid ping-pong loops
            call_key = (tool_name, json.dumps(args, sort_keys=True))
            if call_key in seen_calls:
                logger.warning(
                    "Cycle detected for tool '%s' with args %s; skipping",
                    tool_name,
                    call_key[1],
                )
                cycle_msg = json.dumps(
                    {"warning": "tool_cycle_detected", "tool": tool_name, "args": args},
                    ensure_ascii=False,
                )
                # Persist call + result (cycle notice)
                tool_call_event = ChatEvent(
                    conversation_id=conversation_id,
                    type="tool_call",
                    role="assistant",
                    tool_calls=[
                        ToolCall(id=call["id"], name=tool_name, arguments=args)
                    ],
                    extra={"user_request_id": request_id, "tool_call_id": call["id"]},
                )
                await self.repo.add_event(tool_call_event)
                tool_result_event = ChatEvent(
                    conversation_id=conversation_id,
                    type="tool_result",
                    role="tool",
                    content=cycle_msg,
                    extra={
                        "user_request_id": request_id,
                        "tool_call_id": call["id"],
                        "tool_name": tool_name,
                    },
                )
                await self.repo.add_event(tool_result_event)
                conv.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": cycle_msg}
                )
                continue

            seen_calls.add(call_key)

            # 1. Persist the tool call event
            tool_call_event = ChatEvent(
                conversation_id=conversation_id,
                type="tool_call",
                role="assistant",  # Tool calls are initiated by assistant
                tool_calls=[ToolCall(id=call["id"], name=tool_name, arguments=args)],
                extra={"user_request_id": request_id, "tool_call_id": call["id"]},
            )
            await self.repo.add_event(tool_call_event)

            logger.info("Calling tool '%s' with args: %s", tool_name, args)

            # 2. Execute the tool
            try:
                result = await self.tool_mgr.call_tool(tool_name, args)
            except Exception as e:
                logger.exception("Tool '%s' execution failed: %s", tool_name, e)
                # Surface execution error to model
                content = json.dumps(
                    {"error": "tool_execution_failed", "detail": str(e)}
                )
            else:
                # 2b. Handle structured content
                content = self._pluck_content(result)

            # 3. Persist the tool result event
            tool_result_event = ChatEvent(
                conversation_id=conversation_id,
                type="tool_result",
                role="tool",
                content=content,
                extra={
                    "user_request_id": request_id,
                    "tool_call_id": call["id"],
                    "tool_name": tool_name,
                },
            )
            await self.repo.add_event(tool_result_event)

            # 4. Add to conversation for immediate LLM context
            conv.append(
                {"role": "tool", "tool_call_id": call["id"], "content": content}
            )

    async def chat_once(
        self,
        conversation_id: str,
        user_msg: str,
        request_id: str,
    ) -> ChatEvent:
        """Non-streaming: persist user msg first, call LLM once, persist assistant."""
        await self._ready.wait()

        if not self.tool_mgr:
            raise InitializationError("Tool manager not initialized")

        # Check for existing response to prevent double-billing
        existing_response = await self.persist.get_existing_assistant_response(
            conversation_id, request_id
        )
        if existing_response:
            logger.info("Returning cached response for request_id: %s", request_id)
            return existing_response

        # 1) persist user message FIRST with idempotency check
        should_continue = await self.persist.ensure_user_message_persisted(
            conversation_id, user_msg, request_id
        )
        if not should_continue:
            existing_response = await self.persist.get_existing_assistant_response(
                conversation_id, request_id
            )
            if existing_response:
                logger.info(
                    "Returning existing response for duplicate request_id: %s",
                    request_id,
                )
                return existing_response

        # 2) build canonical history from repo (includes tool calls/results)
        conv = await self._build_conversation_history(conversation_id)

        # 3) Generate assistant response with usage tracking
        (
            assistant_full_text,
            total_usage,
            model,
        ) = await self._generate_assistant_response(conv)

        # 4) persist assistant message with usage and reference to user request
        assistant_ev = ChatEvent(
            conversation_id=conversation_id,
            type="assistant_message",
            role="assistant",
            content=assistant_full_text,
            usage=total_usage,
            provider="anthropic",
            model=model,
            extra={"user_request_id": request_id},
        )
        await self.repo.add_event(assistant_ev)

        return assistant_ev

    def _convert_usage(self, api_usage: dict[str, Any] | None) -> Usage:
        """Convert LLM API usage to our Usage model."""
        if not api_usage:
            return Usage()
        return Usage(
            prompt_tokens=api_usage.get("prompt_tokens", 0),
            completion_tokens=api_usage.get("completion_tokens", 0),
            total_tokens=api_usage.get("total_tokens", 0),
        )

    async def _generate_assistant_response(
        self, conv: list[dict[str, Any]]
    ) -> tuple[str, Usage, str]:
        """Generate assistant response using tools if needed (non-streaming path)."""
        if not self.tool_mgr:
            raise InitializationError("Tool manager not initialized")

        tools_payload = self.tool_mgr.get_mcp_tools()

        assistant_full_text = ""
        total_usage = Usage()
        model = ""

        reply = await self._call_anthropic_api(conv, tools_payload)
        assistant_msg = reply["message"]

        # Log LLM reply if configured
        self._log_llm_reply(reply, "Initial LLM response")

        # Track usage from this API call
        if reply.get("usage"):
            call_usage = self._convert_usage(reply["usage"])
            total_usage.prompt_tokens += call_usage.prompt_tokens
            total_usage.completion_tokens += call_usage.completion_tokens
            total_usage.total_tokens += call_usage.total_tokens

        model = reply.get("model", "") or self.model

        if txt := assistant_msg.get("content"):
            assistant_full_text += txt

        hops = 0
        seen_calls: set[tuple[str, str]] = set()

        while assistant_msg.get("tool_calls"):
            calls: list[dict[str, Any]] = assistant_msg["tool_calls"]

            if hops >= MAX_TOOL_HOPS:
                logger.warning(
                    "Maximum tool hops (%d) reached, stopping recursion", MAX_TOOL_HOPS
                )
                assistant_full_text += (
                    f"\n\n⚠️ Reached maximum tool call limit ({MAX_TOOL_HOPS}). "
                    "Stopping to prevent infinite recursion."
                )
                break

            conv.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.get("content") or "",
                    "tool_calls": calls,
                }
            )

            # Execute tools (non-streaming path)
            await self._execute_tool_calls(
                conv,
                calls,
                conversation_id="__inline__",
                request_id="__inline__",
                seen_calls=seen_calls,
            )

            # Follow-up call
            reply = await self._call_anthropic_api(conv, tools_payload)
            assistant_msg = reply["message"]

            # Log LLM reply if configured
            self._log_llm_reply(reply, f"Tool call follow-up response (hop {hops + 1})")

            # Track usage from subsequent API calls
            if reply.get("usage"):
                call_usage = self._convert_usage(reply["usage"])
                total_usage.prompt_tokens += call_usage.prompt_tokens
                total_usage.completion_tokens += call_usage.completion_tokens
                total_usage.total_tokens += call_usage.total_tokens

            if txt := assistant_msg.get("content"):
                assistant_full_text += txt

            hops += 1

        return assistant_full_text, total_usage, model

    def _log_llm_reply(self, reply: dict[str, Any], context: str) -> None:
        """Log LLM reply if configured."""
        if not self.chat_conf.get("logging", {}).get("llm_replies", False):
            return

        message = reply.get("message", {}) or {}
        content = message.get("content", "") or ""
        tool_calls = message.get("tool_calls", []) or []

        # Truncate content if configured
        logging_config = self.chat_conf.get("logging", {})
        truncate_length = int(logging_config.get("llm_reply_truncate_length", 500))
        if content and len(content) > truncate_length:
            content = content[:truncate_length] + "..."

        parts = [f"LLM Reply ({context}):"]
        if content:
            parts.append(f"Content: {content}")

        if tool_calls:
            parts.append(f"Tool calls: {len(tool_calls)}")
            for i, call in enumerate(tool_calls):
                func_name = call.get("function", {}).get("name", "unknown")
                parts.append(f"  - Tool {i + 1}: {func_name}")

        usage = reply.get("usage", {}) or {}
        if usage:
            p = usage.get("prompt_tokens", 0)
            c = usage.get("completion_tokens", 0)
            t = usage.get("total_tokens", 0)
            parts.append(f"Usage: {p}p + {c}c = {t}t")

        model = reply.get("model", "unknown")
        parts.append(f"Model: {model}")

        logger.info(" | ".join(parts))

    def _pluck_content(self, res: types.CallToolResult) -> str:
        """Extract content from a tool call result."""
        if not getattr(res, "content", None):
            return "✓ done"

        # Handle structured content if present
        if hasattr(res, "structuredContent") and res.structuredContent:
            try:
                return json.dumps(res.structuredContent, indent=2)
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to serialize structured content: %s", e)

        # Extract text from each piece of content
        out: list[str] = []
        for item in res.content:
            if isinstance(item, types.TextContent):
                out.append(item.text)
            elif isinstance(item, types.ImageContent):
                out.append(f"[Image: {item.mimeType}, {len(item.data)} bytes]")
            elif isinstance(item, types.BlobResourceContents):
                out.append(f"[Binary content: {len(item.blob)} bytes]")
            elif isinstance(item, types.EmbeddedResource):
                if isinstance(item.resource, types.TextResourceContents):
                    out.append(f"[Embedded resource: {item.resource.text}]")
                else:
                    out.append(f"[Embedded resource: {type(item.resource).__name__}]")
            else:
                out.append(f"[{type(item).__name__}]")

        return "\n".join(out)

    async def cleanup(self) -> None:
        """Clean up resources by closing all connected MCP clients and HTTP
        client (idempotent)."""
        # Close HTTP client first
        if getattr(self, "http_client", None):
            with contextlib.suppress(Exception):
                await self.http_client.close()

        # Close MCP clients (only if tool_mgr initialized)
        tool_mgr = getattr(self, "tool_mgr", None)
        if tool_mgr and getattr(tool_mgr, "clients", None):
            for client in tool_mgr.clients:
                try:
                    await client.close()
                except Exception as e:  # pragma: no cover
                    logger.warning(
                        "Error closing client %s: %s",
                        getattr(client, "name", "<unknown>"),
                        e,
                    )

    async def apply_prompt(
        self, name: str, args: dict[str, str]
    ) -> list[dict[str, Any]]:
        """Apply a parameterized prompt and return conversation messages."""
        if not self.tool_mgr:
            raise InitializationError("Tool manager not initialized")

        res = await self.tool_mgr.get_prompt(name, args)

        return [
            {"role": m.role, "content": m.content.text}
            for m in res.messages
            if isinstance(m.content, types.TextContent)
        ]
