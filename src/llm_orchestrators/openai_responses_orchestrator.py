# openai_responses_orchestrator.py
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncGenerator
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
    from src.main import MCPClient

logger = logging.getLogger(__name__)

MAX_TOOL_HOPS = 8


class OrchestratorError(Exception): ...


class InitializationError(OrchestratorError): ...


class TransportError(OrchestratorError): ...


class UpstreamProtocolError(OrchestratorError): ...


class ToolExecutionError(OrchestratorError): ...


@dataclass(slots=True)
class ChatMessage:
    mtype: Literal["text", "error"]
    content: str
    meta: dict[str, Any]

    def __init__(
        self,
        mtype: Literal["text", "error"],
        content: str,
        meta: dict[str, Any] | None = None,
    ):
        self.mtype = mtype
        self.content = content
        self.meta = meta if meta is not None else {}

    @property
    def type(self) -> str:
        return self.mtype


class OpenAIResponsesOrchestrator:
    """
    Orchestrator specialized for OpenAI's Responses API (/v1/responses).
    - Works great for o-series models (o3/o4/o1), including streaming & tool calls
    - Keeps your Chat Completions orchestrator unchanged
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
        self.chat_conf = config.get("chat", {}).get("service", {})
        self.tool_mgr: SchemaManager | None = None
        self._init_lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self.mcp_prompt: MCPResourcePromptService | None = None
        self._system_prompt: str = ""

        self.active_name = (llm_config.get("active") or "").lower()
        providers = llm_config.get("providers") or {}
        self.provider_cfg = providers.get(self.active_name, {})

        # Extract API key and base URL
        self.api_key = self._extract_api_key()
        self.base_url = (self.provider_cfg.get("base_url") or "").rstrip("/")
        if not self.api_key:
            raise InitializationError(
                f"Missing API key: set {self.active_name.upper()}_API_KEY or "
                f"providers['{self.active_name}']['api_key']"
            )
        if not self.base_url:
            raise InitializationError("Missing base_url in provider config")

        # HTTP client
        http_config = create_http_config_from_dict(llm_config)
        if "timeout" in self.provider_cfg:
            http_config.timeout = self.provider_cfg["timeout"]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "MCP-Platform/1.0",
        }
        self.http_client = ResilientHttpClient(
            base_url=self.base_url, headers=headers, config=http_config
        )

    def _extract_api_key(self) -> str:
        provider_name = (self.llm_config.get("active") or "").lower()

        # Map provider names to environment variable names
        provider_key_map = {
            "openai": "OPENAI_API_KEY",
            "openai_responses": "OPENAI_API_KEY",  # Uses same OpenAI API key
            "groq": "GROQ_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "mistral": "MISTRAL_API_KEY",
        }

        env_key = provider_key_map.get(provider_name)
        if not env_key:
            raise InitializationError(
                f"Unknown provider '{provider_name}' - no API key mapping found"
            )

        return os.getenv(env_key) or self.provider_cfg.get("api_key", "")

    @property
    def model(self) -> str:
        return self.provider_cfg.get("model", "")

    async def initialize(self) -> None:
        async with self._init_lock:
            if self._ready.is_set():
                return
            try:
                # Connect MCP clients
                connection_results = await asyncio.gather(
                    *(c.connect() for c in self.clients), return_exceptions=True
                )
                connected_clients = []
                for i, result in enumerate(connection_results):
                    if isinstance(result, Exception):
                        client_name = self.clients[i].name
                        logger.warning(
                            f"Client '{client_name}' failed to connect: {result}"
                        )
                    else:
                        connected_clients.append(self.clients[i])

                if not connected_clients:
                    logger.warning(
                        "No MCP clients connected - running with basic functionality"
                    )
                else:
                    logger.info(
                        f"Successfully connected to {len(connected_clients)} out of "
                        f"{len(self.clients)} MCP clients"
                    )

                self.tool_mgr = SchemaManager(connected_clients)
                await self.tool_mgr.initialize()

                self.mcp_prompt = MCPResourcePromptService(
                    self.tool_mgr, self.chat_conf
                )
                await self.mcp_prompt.update_resource_catalog_on_availability()
                self._system_prompt = await self.mcp_prompt.make_system_prompt()

                logger.info(
                    "OpenAIResponsesOrchestrator ready: %d tools, %d res, %d prompts",
                    len(self.tool_mgr.get_mcp_tools()),
                    len(self.mcp_prompt.resource_catalog if self.mcp_prompt else []),
                    len(
                        self.tool_mgr.list_available_prompts() if self.tool_mgr else []
                    ),
                )
                if self.chat_conf.get("logging", {}).get("system_prompt", True):
                    logger.info("System prompt being used:\n%s", self._system_prompt)

                self._ready.set()
            except Exception as e:
                logger.error(f"Failed to initialize Responses orchestrator: {e}")
                raise InitializationError(
                    f"Responses orchestrator initialization failed: {e}"
                ) from e

    # --------- Public entrypoints ---------

    async def process_message(
        self,
        conversation_id: str,
        user_msg: str,
        request_id: str,
    ) -> AsyncGenerator[ChatMessage]:
        """
        Streaming path with tool orchestration (Responses API).
        """
        await self._ready.wait()
        if not self.tool_mgr:
            raise RuntimeError("Tool manager not initialized")

        # Check for cached response first
        cached_response = await self._get_cached_response(conversation_id, request_id)
        if cached_response:
            yield cached_response
            return

        # Ensure user message is persisted
        if not await self._ensure_user_message_ready(
            conversation_id, user_msg, request_id
        ):
            cached_response = await self._get_cached_response(
                conversation_id, request_id
            )
            if cached_response:
                yield cached_response
                return

        # Generate streaming response with tool orchestration
        conv = await self._build_conversation_history(conversation_id)
        full_assistant_content = ""

        async for message in self._process_with_tools(
            conv, conversation_id, request_id
        ):
            yield message
            if message.type == "text":
                full_assistant_content += message.content

        # Persist final response
        await self.persist.persist_final_assistant_message(
            conversation_id, full_assistant_content, request_id
        )

    async def _get_cached_response(
        self, conversation_id: str, request_id: str
    ) -> ChatMessage | None:
        """Get cached response if available."""
        existing_response = await self.persist.get_existing_assistant_response(
            conversation_id, request_id
        )
        if existing_response:
            cached_text = self.persist.get_cached_text(existing_response)
            if cached_text:
                return ChatMessage("text", cached_text, {"type": "cached"})
        return None

    async def _ensure_user_message_ready(
        self, conversation_id: str, user_msg: str, request_id: str
    ) -> bool:
        """Ensure user message is persisted and ready for processing."""
        return await self.persist.ensure_user_message_persisted(
            conversation_id, user_msg, request_id
        )

    async def _process_with_tools(
        self, conv: list[dict[str, Any]], conversation_id: str, request_id: str
    ) -> AsyncGenerator[ChatMessage]:
        """Process conversation with tool orchestration."""
        tools_payload = self.tool_mgr.get_mcp_tools() if self.tool_mgr else []
        hops = 0

        # Generate initial response
        assistant_msg: dict[str, Any] = {}
        async for chunk in self._stream_llm_response(conv, tools_payload):
            if isinstance(chunk, ChatMessage):
                yield chunk
            else:
                assistant_msg = chunk

        # Tool call loop
        while assistant_msg.get("tool_calls") and hops < MAX_TOOL_HOPS:
            conv.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.get("content") or "",
                    "tool_calls": assistant_msg["tool_calls"],
                }
            )

            await self._execute_tool_calls(
                conv, assistant_msg["tool_calls"], conversation_id, request_id
            )

            assistant_msg = {}
            async for chunk in self._stream_llm_response(conv, tools_payload):
                if isinstance(chunk, ChatMessage):
                    yield chunk
                else:
                    assistant_msg = chunk
            hops += 1

        # Handle tool limit reached
        if assistant_msg.get("tool_calls") and hops >= MAX_TOOL_HOPS:
            warn = (
                f"⚠️ Reached maximum tool call limit ({MAX_TOOL_HOPS}). "
                "Stopping to prevent loops."
            )
            yield ChatMessage("text", warn, {"finish_reason": "tool_limit_reached"})

    async def chat_once(
        self,
        conversation_id: str,
        user_msg: str,
        request_id: str,
    ) -> ChatEvent:
        """
        Non-streaming single-shot using Responses API.
        """
        await self._ready.wait()
        if not self.tool_mgr:
            raise RuntimeError("Tool manager not initialized")

        existing_response = await self.persist.get_existing_assistant_response(
            conversation_id, request_id
        )
        if existing_response:
            return existing_response

        should_continue = await self.persist.ensure_user_message_persisted(
            conversation_id, user_msg, request_id
        )
        if not should_continue:
            existing_response = await self.persist.get_existing_assistant_response(
                conversation_id, request_id
            )
            if existing_response:
                return existing_response

        conv = await self._build_conversation_history(conversation_id)

        (
            assistant_full_text,
            total_usage,
            model,
        ) = await self._generate_assistant_response(conv)

        ev = ChatEvent(
            conversation_id=conversation_id,
            type="assistant_message",
            role="assistant",
            content=assistant_full_text,
            usage=total_usage,
            provider="openai",
            model=model,
            extra={"user_request_id": request_id},
        )
        await self.repo.add_event(ev)
        return ev

    async def cleanup(self) -> None:
        if getattr(self, "http_client", None):
            with contextlib.suppress(Exception):
                await self.http_client.close()
        if not self.tool_mgr:
            return
        for client in self.tool_mgr.clients:
            with contextlib.suppress(Exception):
                await client.close()

    # --------- Internals ---------

    async def _build_conversation_history(
        self, conversation_id: str
    ) -> list[dict[str, Any]]:
        """
        Build a Chat-Completions-shaped history (system/user/assistant/tool),
        then convert to Responses input right before calling the API.
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
                tool_call_id = ev.extra.get("tool_call_id")
                if tool_call_id:
                    conv.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": ev.content or "",
                        }
                    )
        return conv

    def _messages_to_responses_input(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert Chat-style messages to Responses `input` items.
        Try simple string content format first, then fall back to array format if
        needed.
        """
        items: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "tool":
                items.append(
                    {
                        "type": "tool_result",
                        "tool_call_id": m.get("tool_call_id"),
                        "output": m.get("content") or "",
                    }
                )
                continue

            # Try simple string content format first
            text = m.get("content") or ""
            items.append(
                {
                    "role": role,
                    "content": text,  # Simple string format
                }
            )
        return items

    def _build_request(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        # Convert messages to Responses API input format
        input_items = self._messages_to_responses_input(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }

        # Add optional parameters - be conservative with what we include
        # Note: Responses API uses 'max_output_tokens', not 'max_tokens'
        if self.provider_cfg.get("max_output_tokens"):
            payload["max_output_tokens"] = self.provider_cfg["max_output_tokens"]
        elif self.provider_cfg.get("max_tokens"):
            # Map legacy 'max_tokens' to 'max_output_tokens' for compatibility
            payload["max_output_tokens"] = self.provider_cfg["max_tokens"]
        else:
            payload["max_output_tokens"] = 2048  # Default

        # Note: Some reasoning models (like o3) don't support temperature/top_p
        # Only add these parameters if they're explicitly supported
        model_name = self.model.lower()
        reasoning_models = ["o1", "o3", "o4"]  # Common reasoning models
        supports_sampling = not any(rm in model_name for rm in reasoning_models)

        if supports_sampling:
            # Only add temperature/top_p for non-reasoning models
            if "temperature" in self.provider_cfg:
                payload["temperature"] = self.provider_cfg["temperature"]
            if "top_p" in self.provider_cfg:
                payload["top_p"] = self.provider_cfg["top_p"]

        # Add MCP tools if available - use native MCP format for Responses API
        # Note: For now, we disable MCP tools for Responses API since they require
        # HTTP servers. We'll add HTTP server auto-start in a future update.
        # TODO: Implement automatic HTTP MCP server startup for full tool integration
        if False:  # Temporarily disabled - requires HTTP MCP servers
            # Convert our MCP tools to native MCP tool format for Responses API
            mcp_tool = {
                "type": "mcp",
                "server_url": "http://127.0.0.1:8000/mcp/",  # FastMCP HTTP server
                "server_label": "demo",
                "allowed_tools": [t["name"] for t in tools],
                "require_approval": "never",
            }
            payload["tools"] = [mcp_tool]

        return payload

    async def _call_responses_api(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """
        Non-streaming call to /v1/responses for reasoning models.
        Returns: {"message": {"content": str|None, "tool_calls": list|None},
                 "usage": {...}, "model": "..."}
        """
        try:
            payload = self._build_request(messages, tools)
            # DEBUG: Log the exact payload being sent
            logger.info(
                f"DEBUG: Sending payload to /responses: {json.dumps(payload, indent=2)}"
            )
            resp = await self.http_client.post_json("/responses", payload)
            if not isinstance(resp, httpx.Response):
                raise TransportError("Expected httpx.Response")
            data = resp.json()
            return self._parse_responses_final(data)
        except Exception as e:
            logger.error(f"Responses API call failed: {e}")
            raise UpstreamProtocolError(f"API call failed: {e}") from e

    def _parse_responses_final(self, data: dict[str, Any]) -> dict[str, Any]:
        """Parse OpenAI Responses API format."""
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        # data["output"] is a list of typed items: messages, tool calls, etc.
        for item in data.get("output") or []:
            if item.get("type") == "message" and item.get("role") == "assistant":
                for c in item.get("content", []):
                    ctype = c.get("type")
                    if ctype in ("output_text", "text"):
                        text_parts.append(c.get("text", ""))
                    elif ctype in ("tool_call", "function_call"):
                        args = c.get("arguments", {})
                        tool_calls.append(
                            {
                                "id": c.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": c.get("name", ""),
                                    "arguments": args
                                    if isinstance(args, str)
                                    else json.dumps(args),
                                },
                            }
                        )

        usage = data.get("usage", {})
        return {
            "message": {
                "content": "".join(text_parts) or None,
                "tool_calls": tool_calls or None,
            },
            "finish_reason": data.get("status"),
            "usage": usage,
            "model": data.get("model", self.model),
        }

    def _parse_chat_completion_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Parse standard OpenAI chat completion response format."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        return {
            "message": {
                "content": message.get("content"),
                "tool_calls": message.get("tool_calls"),
            },
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage", {}),
            "model": data.get("model", self.model),
        }

    async def _stream_responses_api(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncGenerator[dict[str, Any]]:
        """
        Stream from /v1/responses. Parse SSE format for reasoning models.
        """
        payload = self._build_request(messages, tools)
        payload["stream"] = True  # Enable streaming

        # DEBUG: Log the exact streaming payload being sent
        logger.info(
            f"DEBUG: Sending streaming payload: {json.dumps(payload, indent=2)}"
        )

        try:
            stream = await self.http_client.post_json(
                "/responses", payload, stream=True
            )
            if not isinstance(stream, httpx.Response):
                async for chunk in stream:
                    # chunk should already be decoded dicts
                    yield chunk
                return

            # If we got a Response, parse SSE lines
            async for line in stream.aiter_lines():  # type: ignore[attr-defined]
                if not line:
                    continue
                if line.startswith("data: "):
                    raw = line[6:]
                    if raw.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(raw)
                    except Exception:
                        continue
        except Exception as e:
            logger.error(f"Streaming API call failed: {e}")
            raise UpstreamProtocolError(f"Streaming API call failed: {e}") from e

    async def _process_streaming_with_tools(
        self, conv: list[dict[str, Any]], conversation_id: str, request_id: str
    ) -> AsyncGenerator[ChatMessage]:
        tools_payload = self.tool_mgr.get_mcp_tools() if self.tool_mgr else []
        full_assistant_content = ""

        assistant_msg: dict[str, Any] = {}
        async for chunk in self._stream_llm_response(conv, tools_payload):
            if isinstance(chunk, ChatMessage):
                yield chunk
                if chunk.type == "text":
                    full_assistant_content += chunk.content
            else:
                assistant_msg = chunk

        hops = 0
        while assistant_msg.get("tool_calls"):
            if hops >= MAX_TOOL_HOPS:
                warn = (
                    f"⚠️ Reached maximum tool call limit ({MAX_TOOL_HOPS}). "
                    "Stopping to prevent loops."
                )
                yield ChatMessage("text", warn, {"finish_reason": "tool_limit_reached"})
                break

            conv.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.get("content") or "",
                    "tool_calls": assistant_msg["tool_calls"],
                }
            )

            await self._execute_tool_calls(
                conv, assistant_msg["tool_calls"], conversation_id, request_id
            )

            assistant_msg = {}
            async for chunk in self._stream_llm_response(conv, tools_payload):
                if isinstance(chunk, ChatMessage):
                    yield chunk
                else:
                    assistant_msg = chunk
            hops += 1

    async def _stream_llm_response(
        self, conv: list[dict[str, Any]], tools_payload: list[dict[str, Any]]
    ) -> AsyncGenerator[ChatMessage | dict[str, Any]]:
        """
        Normalize Responses events to your existing ChatMessage + final dict.
        """
        message_buffer = ""
        current_tool_calls: list[dict[str, Any]] = []
        finish_reason: str | None = None

        try:
            async for frame in self._stream_responses_api(conv, tools_payload):
                # Accept both {"event": "...", "data": {...}} and flattened dicts
                event = frame.get("event") or frame.get("type")
                data = frame.get("data") or frame

                if event in ("response.output_text.delta", "response.text.delta"):
                    delta = data.get("delta") or data.get("text") or ""
                    if delta:
                        message_buffer += delta
                        yield ChatMessage("text", delta, {"type": "delta"})

                elif event in (
                    "response.function_call_arguments.delta",
                    "response.mcp_call_arguments.delta",
                ):
                    idx = data.get("index", 0)
                    while len(current_tool_calls) <= idx:
                        current_tool_calls.append(
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        )
                    tc = data.get("tool_call") or data.get("function_call") or {}
                    if "id" in tc:
                        current_tool_calls[idx]["id"] = tc["id"]
                    if "name" in tc:
                        current_tool_calls[idx]["function"]["name"] = tc["name"]
                    if "arguments_delta" in data:
                        prev = current_tool_calls[idx]["function"]["arguments"]
                        current_tool_calls[idx]["function"]["arguments"] = prev + (
                            data["arguments_delta"] or ""
                        )
                    elif "arguments" in tc:
                        args = tc["arguments"]
                        current_tool_calls[idx]["function"]["arguments"] = (
                            args if isinstance(args, str) else json.dumps(args)
                        )

                elif event in ("response.completed", "response.error"):
                    finish_reason = event

                # You can optionally handle reasoning/trace/metrics events here.

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield ChatMessage(
                "error",
                f"Streaming interrupted: {e!s}",
                {"type": "error", "recoverable": True},
            )
            return

        assistant_message = {
            "content": message_buffer or None,
            "tool_calls": current_tool_calls
            if current_tool_calls
            and any(c["function"]["name"] for c in current_tool_calls)
            else None,
            "finish_reason": finish_reason,
        }
        yield assistant_message

    async def _execute_tool_calls(
        self,
        conv: list[dict[str, Any]],
        calls: list[dict[str, Any]],
        conversation_id: str,
        request_id: str,
    ) -> None:
        assert self.tool_mgr is not None
        logger.info(f"Executing {len(calls)} tool calls (Responses)")

        seen_calls: set[tuple[str, str]] = set()

        for call in calls:
            tool_name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments: {e}")
                error_msg = f"Error: Invalid JSON in tool arguments: {e}"

                # Persist & add as tool_result
                tool_result_event = ChatEvent(
                    conversation_id=conversation_id,
                    type="tool_result",
                    role="tool",
                    content=error_msg,
                    extra={
                        "user_request_id": request_id,
                        "tool_call_id": call.get("id"),
                        "tool_name": tool_name,
                        "error": "json_parse_error",
                    },
                )
                await self.repo.add_event(tool_result_event)
                conv.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": error_msg,
                    }
                )
                continue

            call_key = (tool_name, json.dumps(args, sort_keys=True))
            if call_key in seen_calls:
                warn = f"⚠️ Cycle detected: tool '{tool_name}' called with same args"
                tool_result_event = ChatEvent(
                    conversation_id=conversation_id,
                    type="tool_result",
                    role="tool",
                    content=warn,
                    extra={
                        "user_request_id": request_id,
                        "tool_call_id": call.get("id"),
                        "tool_name": tool_name,
                        "error": "cycle_detected",
                    },
                )
                await self.repo.add_event(tool_result_event)
                conv.append(
                    {"role": "tool", "tool_call_id": call.get("id"), "content": warn}
                )
                continue

            seen_calls.add(call_key)

            # Persist tool call
            tool_call_event = ChatEvent(
                conversation_id=conversation_id,
                type="tool_call",
                role="assistant",
                tool_calls=[
                    ToolCall(id=call.get("id", ""), name=tool_name, arguments=args)
                ],
                extra={"user_request_id": request_id, "tool_call_id": call.get("id")},
            )
            await self.repo.add_event(tool_call_event)

            # Execute
            try:
                result = await self.tool_mgr.call_tool(tool_name, args)
                content = self._pluck_content(result)
            except Exception as e:
                logger.error(f"Tool execution failed for '{tool_name}': {e}")
                content = f"Error executing tool '{tool_name}': {e}"

            # Persist result
            tool_result_event = ChatEvent(
                conversation_id=conversation_id,
                type="tool_result",
                role="tool",
                content=content,
                extra={
                    "user_request_id": request_id,
                    "tool_call_id": call.get("id"),
                    "tool_name": tool_name,
                },
            )
            await self.repo.add_event(tool_result_event)

            # Append to conversation (will be converted to Responses tool_result on
            # send)
            conv.append(
                {"role": "tool", "tool_call_id": call.get("id"), "content": content}
            )

    def _pluck_content(self, res: types.CallToolResult) -> str:
        if not res.content:
            return "✓ done"
        if hasattr(res, "structuredContent") and res.structuredContent:
            try:
                return json.dumps(res.structuredContent, indent=2)
            except Exception as e:
                logger.warning(f"Failed to serialize structured content: {e}")
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

    def _convert_usage(self, api_usage: dict[str, Any] | None) -> Usage:
        if not api_usage:
            return Usage()
        return Usage(
            prompt_tokens=api_usage.get("prompt_tokens", 0),
            completion_tokens=api_usage.get("completion_tokens", 0),
            total_tokens=api_usage.get("total_tokens", 0),
            reasoning_tokens=api_usage.get("reasoning_tokens", 0),
        )

    async def _generate_assistant_response(
        self, conv: list[dict[str, Any]]
    ) -> tuple[str, Usage, str]:
        assert self.tool_mgr is not None
        tools_payload = self.tool_mgr.get_mcp_tools()

        assistant_full_text = ""
        total_usage = Usage()
        model = ""

        reply = await self._call_responses_api(conv, tools_payload)
        assistant_msg = reply["message"]

        if reply.get("usage"):
            call_usage = self._convert_usage(reply["usage"])
            total_usage.prompt_tokens += call_usage.prompt_tokens
            total_usage.completion_tokens += call_usage.completion_tokens
            total_usage.total_tokens += call_usage.total_tokens
            total_usage.reasoning_tokens += call_usage.reasoning_tokens

        model = reply.get("model", "")

        if assistant_msg.get("content"):
            assistant_full_text += assistant_msg["content"]

        hops = 0
        while assistant_msg.get("tool_calls"):
            if hops >= MAX_TOOL_HOPS:
                assistant_full_text += (
                    f"\n\n⚠️ Reached maximum tool call limit ({MAX_TOOL_HOPS}). "
                    "Stopping."
                )
                break

            conv.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.get("content") or "",
                    "tool_calls": assistant_msg["tool_calls"],
                }
            )

            for call in assistant_msg["tool_calls"]:
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = await self.tool_mgr.call_tool(call["function"]["name"], args)
                content = self._pluck_content(result)
                conv.append(
                    {"role": "tool", "tool_call_id": call.get("id"), "content": content}
                )

            reply = await self._call_responses_api(conv, tools_payload)
            assistant_msg = reply["message"]

            if reply.get("usage"):
                call_usage = self._convert_usage(reply["usage"])
                total_usage.prompt_tokens += call_usage.prompt_tokens
                total_usage.completion_tokens += call_usage.completion_tokens
                total_usage.total_tokens += call_usage.total_tokens
                total_usage.reasoning_tokens += call_usage.reasoning_tokens

            if assistant_msg.get("content"):
                assistant_full_text += assistant_msg["content"]

            hops += 1

        return assistant_full_text, total_usage, model
