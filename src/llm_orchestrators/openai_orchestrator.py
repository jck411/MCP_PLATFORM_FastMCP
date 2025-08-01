"""
OpenAI Orchestrator for MCP Platform

This module handles the business logic for chat sessions with integrated
OpenAI API compatibility:
- Conversation management with simple default prompts
- Tool orchestration
- MCP client coordination
- Direct OpenAI API interactions (no adapter layer)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import uuid
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

# Maximum number of tool call hops to prevent infinite recursion
MAX_TOOL_HOPS = 8


class OrchestratorError(Exception):
    """Base class for orchestrator errors."""


class InitializationError(OrchestratorError):
    """Raised when orchestrator is used before initialization."""


class TransportError(OrchestratorError):
    """Raised when HTTP/stream transport contracts are violated."""


class UpstreamProtocolError(OrchestratorError):
    """Raised when upstream API responses violate expected protocols."""


class ToolExecutionError(OrchestratorError):
    """Raised when tool execution fails."""


@dataclass(slots=True)
class ChatMessage:
    """
    Represents a chat message with metadata.
    """
    mtype: Literal["text", "error"]
    content: str
    meta: dict[str, Any]

    def __init__(
        self,
        mtype: Literal["text", "error"],
        content: str,
        meta: dict[str, Any] | None = None
    ):
        self.mtype = mtype
        self.content = content
        self.meta = meta if meta is not None else {}

    @property
    def type(self) -> str:
        """Compatibility property for existing code."""
        return self.mtype


class OpenAIOrchestrator:
    """
    Conversation orchestrator with integrated OpenAI API handling.
    1. Takes your message
    2. Figures out what tools might be needed
    3. Asks the AI to respond (and use tools if needed) via OpenAI API
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
        self.chat_conf = config.get("chat", {}).get("service", {})
        self.tool_mgr: SchemaManager | None = None
        self._init_lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self.mcp_prompt: MCPResourcePromptService | None = None
        self._system_prompt: str = ""

        # Establish provider fields first
        self.active_name = (llm_config.get("active") or "").lower()
        providers = llm_config.get("providers") or {}
        self.provider_cfg = providers.get(self.active_name, {})

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

        # Configure resilient HTTP client
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "MCP-Platform/1.0",
        }

        self.http_client = ResilientHttpClient(
            base_url=self.base_url,
            headers=headers,
            config=http_config
        )

    def _extract_api_key(self) -> str:
        """Extract API key from env or provider config; raise if absent."""
        provider_name = (self.llm_config.get("active") or "").upper()
        env_key = f"{provider_name}_API_KEY"
        return (
            os.getenv(env_key)
            or self.provider_cfg.get("api_key", "")
        )

    @property
    def model(self) -> str:
        """Get the model name from the current configuration."""
        return self.provider_cfg.get("model", "")

    async def initialize(self) -> None:
        """Initialize the orchestrator and all MCP clients."""
        async with self._init_lock:
            if self._ready.is_set():
                return

            try:
                # Connect to all clients and collect results
                connection_results = await asyncio.gather(
                    *(c.connect() for c in self.clients), return_exceptions=True
                )

                # Filter out only successfully connected clients
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

                # Use connected clients for tool management (empty list is acceptable)
                self.tool_mgr = SchemaManager(connected_clients)
                await self.tool_mgr.initialize()

                # Resource catalog & system prompt (MCP-only, provider-agnostic)
                self.mcp_prompt = MCPResourcePromptService(
                    self.tool_mgr, self.chat_conf
                )
                await self.mcp_prompt.update_resource_catalog_on_availability()
                self._system_prompt = await self.mcp_prompt.make_system_prompt()

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

                # Configurable system prompt logging
                if self.chat_conf.get("logging", {}).get("system_prompt", True):
                    logger.info("System prompt being used:\n%s", self._system_prompt)
                else:
                    logger.debug("System prompt logging disabled in configuration")

                self._ready.set()

            except Exception as e:
                logger.error(f"Failed to initialize OpenAI orchestrator: {e}")
                raise InitializationError(
                    f"Orchestrator initialization failed: {e}"
                ) from e

    def _build_openai_request(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Build OpenAI API request payload."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.provider_cfg.get("temperature", 0.1),
            "max_tokens": self.provider_cfg.get("max_tokens", 2048),
            "top_p": self.provider_cfg.get("top_p", 1.0),
        }

        if tools:
            # Convert MCP tools to OpenAI format
            openai_tools = []
            for tool in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["inputSchema"],
                    }
                })
            payload["tools"] = openai_tools

        # Add reasoning/thinking support for OpenRouter
        if self._supports_reasoning():
            reasoning_config = self._get_reasoning_config()
            if reasoning_config:
                payload["reasoning"] = reasoning_config

        return payload

    def _supports_reasoning(self) -> bool:
        """Check if current provider/model supports reasoning/thinking blocks."""
        # Only enable reasoning for OpenRouter or specific base URLs
        base_url_lower = self.base_url.lower()
        if "openrouter.ai" in base_url_lower:
            return True

        # Enable for specific models that support reasoning
        model_lower = self.model.lower()
        reasoning_models = ["o1", "o3", "o4", "deepseek", "claude", "qwen", "r1"]
        return any(rm in model_lower for rm in reasoning_models)

    def _get_reasoning_config(self) -> dict[str, Any] | None:
        """Get reasoning configuration from provider settings."""
        if not self._supports_reasoning():
            return None

        # Check if reasoning is explicitly configured in provider config
        reasoning_config = self.provider_cfg.get("reasoning", {})

        # If no explicit config, use default reasoning settings
        if not reasoning_config:
            # Use moderate reasoning by default for thinking
            reasoning_config = {"effort": "medium"}

        # Log that we're enabling reasoning
        if reasoning_config:
            logger.info(
                f"Enabling reasoning/thinking for model '{self.model}': "
                f"{reasoning_config}"
            )

        return reasoning_config

    def _log_reasoning(self, reasoning: str, context: str) -> None:
        """Log reasoning/thinking content if configured."""
        if not self.chat_conf.get("logging", {}).get("llm_replies", False):
            return

        # Truncate reasoning if configured
        logging_config = self.chat_conf.get("logging", {})
        truncate_length = logging_config.get("llm_reply_truncate_length", 500)
        if reasoning and len(reasoning) > truncate_length:
            reasoning = reasoning[:truncate_length] + "..."

        logger.info(f"🧠 Reasoning/Thinking ({context}): {reasoning}")

    def _parse_openai_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Parse OpenAI API response."""
        choice = data["choices"][0]
        message = choice["message"]

        # Log reasoning/thinking if present
        if message.get("reasoning"):
            self._log_reasoning(message["reasoning"], "Non-streaming response")

        return {
            "message": message,
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
            "model": data.get("model", self.model),
        }

    async def _call_openai_api(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Make direct OpenAI API call with retry logic."""
        try:
            payload = self._build_openai_request(messages, tools)
            response = await self.http_client.post_json("/chat/completions", payload)

            # For non-streaming, we expect an httpx.Response
            if isinstance(response, httpx.Response):
                return self._parse_openai_response(response.json())

            raise TransportError("Expected httpx.Response for non-streaming request")

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise UpstreamProtocolError(f"API call failed: {e}") from e

    async def _stream_openai_api(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncGenerator[dict[str, Any]]:
        """Stream responses from OpenAI API with retry logic."""
        try:
            payload = self._build_openai_request(messages, tools)
            response_gen = await self.http_client.post_json(
                "/chat/completions", payload, stream=True
            )

            # For streaming, we expect an AsyncGenerator
            if not isinstance(response_gen, httpx.Response):
                async for chunk in response_gen:
                    yield chunk
                return

            raise TransportError("Expected AsyncGenerator for streaming request")

        except Exception as e:
            logger.error(f"OpenAI streaming API call failed: {e}")
            raise UpstreamProtocolError(f"Streaming API call failed: {e}") from e

    async def _build_conversation_history(
        self, conversation_id: str
    ) -> list[dict[str, Any]]:
        """Build conversation history including tool calls and results."""
        events = await self.repo.get_events(conversation_id, self.max_history_messages)
        conv: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
        ]

        for ev in events:
            if ev.type in ("user_message", "assistant_message", "system_update"):
                conv.append({"role": ev.role, "content": ev.content})
            elif ev.type == "tool_call" and ev.tool_calls:
                # Add assistant message with tool calls
                tool_calls = []
                for tc in ev.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    })
                conv.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls
                })
            elif ev.type == "tool_result":
                # Add tool result message
                tool_call_id = ev.extra.get("tool_call_id")
                if tool_call_id:
                    conv.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": ev.content or ""
                    })

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
            raise RuntimeError("Tool manager not initialized")

        # Check for existing response to prevent double-processing
        existing_response = await self.persist.get_existing_assistant_response(
            conversation_id, request_id
        )
        if existing_response:
            logger.info(f"Returning cached response for request_id: {request_id}")
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

        # 2) Build conversation history from persistent store
        conv = await self._build_conversation_history(conversation_id)

        # 3) Generate streaming response with tool handling
        async for msg in self._process_streaming_with_tools(
            conv, conversation_id, request_id
        ):
            yield msg

    async def _process_streaming_with_tools(
        self, conv: list[dict[str, Any]], conversation_id: str, request_id: str
    ) -> AsyncGenerator[ChatMessage]:
        """Handle the streaming response generation and tool execution."""
        assert self.tool_mgr is not None

        tools_payload = self.tool_mgr.get_mcp_tools()
        full_assistant_content = ""

        # Initial response - stream and collect
        assistant_msg: dict[str, Any] = {}
        async for chunk in self._stream_llm_response(conv, tools_payload):
            if isinstance(chunk, ChatMessage):
                yield chunk
                # Collect content for final persistence
                if chunk.type == "text":
                    full_assistant_content += chunk.content
            else:
                assistant_msg = chunk

        # Log LLM reply if configured
        reply_data = {
            "message": assistant_msg,
            "usage": None,  # Usage not available in streaming mode
            "model": self.model
        }
        self._log_llm_reply(reply_data, "Streaming initial response")

        # Handle tool calls in a loop
        hops = 0
        while calls := assistant_msg.get("tool_calls"):
            if hops >= MAX_TOOL_HOPS:
                logger.warning(
                    f"Maximum tool hops ({MAX_TOOL_HOPS}) reached, stopping recursion"
                )
                error_msg = (
                    f"⚠️ Reached maximum tool call limit ({MAX_TOOL_HOPS}). "
                    "Stopping to prevent infinite recursion."
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

            # Execute tool calls
            await self._execute_tool_calls(conv, calls, conversation_id, request_id)

            # Get follow-up response - stream and collect
            assistant_msg = {}
            async for chunk in self._stream_llm_response(conv, tools_payload):
                if isinstance(chunk, ChatMessage):
                    yield chunk
                    # Collect content for final persistence
                    if chunk.type == "text":
                        full_assistant_content += chunk.content
                else:
                    assistant_msg = chunk

            # Log LLM reply if configured
            reply_data = {
                "message": assistant_msg,
                "usage": None,  # Usage not available in streaming mode
                "model": self.model
            }
            context = f"Streaming tool follow-up (hop {hops + 1})"
            self._log_llm_reply(reply_data, context)

            hops += 1

        # 4) Persist final assistant message
        await self.persist.persist_final_assistant_message(
            conversation_id, full_assistant_content, request_id
        )

    async def _stream_llm_response(
        self, conv: list[dict[str, Any]], tools_payload: list[dict[str, Any]]
    ) -> AsyncGenerator[ChatMessage | dict[str, Any]]:
        """Stream response from LLM and yield chunks to user."""
        message_buffer = ""
        current_tool_calls: list[dict[str, Any]] = []
        finish_reason: str | None = None
        reasoning_buffer = ""

        try:
            async for chunk in self._stream_openai_api(conv, tools_payload):
                if "choices" not in chunk or not chunk["choices"]:
                    continue

                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                # Handle content streaming
                if content := delta.get("content"):
                    message_buffer += content
                    yield ChatMessage("text", content, {"type": "delta"})

                # Handle reasoning streaming (OpenRouter/thinking models)
                if reasoning := delta.get("reasoning"):
                    reasoning_buffer += reasoning
                    # Don't yield reasoning content to user, just collect for logging

                # Handle tool calls streaming
                if tool_calls := delta.get("tool_calls"):
                    logger.info(f"Received tool call delta: {tool_calls}")
                    self._accumulate_tool_calls(current_tool_calls, tool_calls)

                # Handle finish reason
                if "finish_reason" in choice:
                    finish_reason = choice["finish_reason"]

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield ChatMessage(
                "error",
                f"Streaming interrupted: {e!s}",
                {"type": "error", "recoverable": True},
            )
            return

        # Log accumulated reasoning if present
        if reasoning_buffer:
            self._log_reasoning(reasoning_buffer, "Streaming response")

        # Yield complete assistant message as final item
        assistant_message = {
            "content": message_buffer or None,
            "tool_calls": current_tool_calls if current_tool_calls and any(
                call["function"]["name"] for call in current_tool_calls
            ) else None,
            "finish_reason": finish_reason
        }

        # Debug logging for tool calls
        if assistant_message.get("tool_calls"):
            logger.info(f"Tool calls detected: {len(assistant_message['tool_calls'])}")
            for i, call in enumerate(assistant_message['tool_calls']):
                func_name = call['function']['name']
                func_args = call['function']['arguments']
                logger.info(f"  Tool {i+1}: {func_name} with args: {func_args}")
        else:
            logger.info("No tool calls detected in response")
            logger.info(f"current_tool_calls buffer: {current_tool_calls}")
            logger.info(f"finish_reason: {finish_reason}")

        yield assistant_message

    def _accumulate_tool_calls(
        self, current_tool_calls: list[dict[str, Any]], tool_calls: list[dict[str, Any]]
    ) -> None:
        """Accumulate streaming tool calls into the current buffer."""
        for tool_call in tool_calls:
            # Ensure we have enough space in the buffer
            while len(current_tool_calls) <= tool_call.get("index", 0):
                current_tool_calls.append({
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""}
                })

            idx = tool_call.get("index", 0)
            if "id" in tool_call and tool_call["id"] is not None:
                current_tool_calls[idx]["id"] = tool_call["id"]
            if "function" in tool_call:
                func = tool_call["function"]
                if "name" in func:
                    current_tool_calls[idx]["function"]["name"] += func["name"]
                if "arguments" in func:
                    call_args = current_tool_calls[idx]["function"]["arguments"]
                    current_tool_calls[idx]["function"]["arguments"] = (
                        call_args + func["arguments"]
                    )

    async def _execute_tool_calls(
        self,
        conv: list[dict[str, Any]],
        calls: list[dict[str, Any]],
        conversation_id: str,
        request_id: str
    ) -> None:
        """Execute tool calls and add results to conversation and history."""
        assert self.tool_mgr is not None

        logger.info(f"Executing {len(calls)} tool calls")

        # Cycle detection to prevent infinite loops
        seen_calls: set[tuple[str, str]] = set()

        for call in calls:
            tool_name = call["function"]["name"]
            # Ensure tool call ID is valid (not None or empty)
            tool_call_id = call.get("id") or f"tool_call_{uuid.uuid4().hex[:8]}"

            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments: {e}")
                error_msg = f"Error: Invalid JSON in tool arguments: {e}"

                # Persist error as tool result
                tool_result_event = ChatEvent(
                    conversation_id=conversation_id,
                    type="tool_result",
                    role="tool",
                    content=error_msg,
                    extra={
                        "user_request_id": request_id,
                        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
                        "error": "json_parse_error"
                    }
                )
                await self.repo.add_event(tool_result_event)

                conv.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": error_msg,
                })
                continue

            # Check for cycles
            call_key = (tool_name, json.dumps(args, sort_keys=True))
            if call_key in seen_calls:
                logger.warning(f"Cycle detected for tool call: {tool_name}")
                error_msg = (
                    f"⚠️ Cycle detected: tool '{tool_name}' called with same arguments"
                )

                # Persist cycle detection as tool result
                tool_result_event = ChatEvent(
                    conversation_id=conversation_id,
                    type="tool_result",
                    role="tool",
                    content=error_msg,
                    extra={
                        "user_request_id": request_id,
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "error": "cycle_detected"
                    }
                )
                await self.repo.add_event(tool_result_event)

                conv.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": error_msg,
                })
                continue

            seen_calls.add(call_key)

            # 1. Persist the tool call event
            tool_call_event = ChatEvent(
                conversation_id=conversation_id,
                type="tool_call",
                role="assistant",  # Tool calls are initiated by assistant
                tool_calls=[ToolCall(
                    id=tool_call_id,
                    name=tool_name,
                    arguments=args
                )],
                extra={
                    "user_request_id": request_id,
                    "tool_call_id": tool_call_id
                }
            )
            await self.repo.add_event(tool_call_event)

            logger.info(f"Calling tool '{tool_name}' with args: {args}")

            # 2. Execute the tool with error handling
            try:
                result = await self.tool_mgr.call_tool(tool_name, args)
                content = self._pluck_content(result)
            except Exception as e:
                logger.error(f"Tool execution failed for '{tool_name}': {e}")
                content = f"Error executing tool '{tool_name}': {e}"

            # 3. Persist the tool result event
            tool_result_event = ChatEvent(
                conversation_id=conversation_id,
                type="tool_result",
                role="tool",
                content=content,
                extra={
                    "user_request_id": request_id,
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name
                }
            )
            await self.repo.add_event(tool_result_event)

            logger.info(f"Tool '{tool_name}' returned: {content[:100]}...")

            # 4. Add to conversation for immediate LLM context
            conv.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                }
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
            raise RuntimeError("Tool manager not initialized")

        # Check for existing response to prevent double-billing
        existing_response = await self.persist.get_existing_assistant_response(
            conversation_id, request_id
        )
        if existing_response:
            logger.info(
                f"Returning cached response for request_id: {request_id}"
            )
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
                    request_id
                )
                return existing_response

        # 2) build canonical history from repo
        events = await self.repo.get_events(conversation_id, self.max_history_messages)

        # 3) convert to OpenAI-style list for your LLM call
        conv: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
        ]
        for ev in events:
            if ev.type in ("user_message", "assistant_message", "system_update"):
                conv.append({"role": ev.role, "content": ev.content})

        # User message is already included in events since we persisted it above

        # 4) Generate assistant response with usage tracking
        (
            assistant_full_text,
            total_usage,
            model
        ) = await self._generate_assistant_response(conv)

        # 5) persist assistant message with usage and reference to user request
        assistant_ev = ChatEvent(
            conversation_id=conversation_id,
            type="assistant_message",
            role="assistant",
            content=assistant_full_text,
            usage=total_usage,
            provider="openai",  # or map from llm_client/config
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
        """Generate assistant response using tools if needed."""
        if not self.tool_mgr:
            raise RuntimeError("Tool manager not initialized")

        tools_payload = self.tool_mgr.get_mcp_tools()

        assistant_full_text = ""
        total_usage = Usage()
        model = ""

        reply = await self._call_openai_api(conv, tools_payload)
        assistant_msg = reply["message"]

        # Log LLM reply if configured
        self._log_llm_reply(reply, "Initial LLM response")

        # Track usage from this API call
        if reply.get("usage"):
            call_usage = self._convert_usage(reply["usage"])
            total_usage.prompt_tokens += call_usage.prompt_tokens
            total_usage.completion_tokens += call_usage.completion_tokens
            total_usage.total_tokens += call_usage.total_tokens

        # Store model from first API call
        model = reply.get("model", "")

        if txt := assistant_msg.get("content"):
            assistant_full_text += txt

        hops = 0
        while calls := assistant_msg.get("tool_calls"):
            if hops >= MAX_TOOL_HOPS:
                logger.warning(
                    f"Maximum tool hops ({MAX_TOOL_HOPS}) reached, stopping recursion"
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

            for call in calls:
                tool_name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"] or "{}")
                # Ensure tool call ID is valid (not None or empty)
                tool_call_id = call.get("id") or f"tool_call_{uuid.uuid4().hex[:8]}"

                result = await self.tool_mgr.call_tool(tool_name, args)
                content = self._pluck_content(result)

                conv.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,
                    }
                )

            reply = await self._call_openai_api(conv, tools_payload)
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

        message = reply.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])
        reasoning = message.get("reasoning", "")

        # Truncate content if configured
        logging_config = self.chat_conf.get("logging", {})
        truncate_length = logging_config.get("llm_reply_truncate_length", 500)
        if content and len(content) > truncate_length:
            content = content[:truncate_length] + "..."

        log_parts = [f"LLM Reply ({context}):"]

        # Log reasoning first if present (separate from main content)
        if reasoning:
            reasoning_truncated = reasoning
            if len(reasoning) > truncate_length:
                reasoning_truncated = reasoning[:truncate_length] + "..."
            log_parts.append(f"🧠 Reasoning: {reasoning_truncated}")

        if content:
            log_parts.append(f"Content: {content}")

        if tool_calls:
            log_parts.append(f"Tool calls: {len(tool_calls)}")
            for i, call in enumerate(tool_calls):
                func_name = call.get("function", {}).get("name", "unknown")
                log_parts.append(f"  - Tool {i+1}: {func_name}")

        usage = reply.get("usage", {})
        if usage:
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
            log_parts.append(
                f"Usage: {prompt_tokens}p + {completion_tokens}c = {total_tokens}t"
            )

        model = reply.get("model", "unknown")
        log_parts.append(f"Model: {model}")

        logger.info(" | ".join(log_parts))

    def _pluck_content(self, res: types.CallToolResult) -> str:
        """Extract content from a tool call result."""
        if not res.content:
            return "✓ done"

        # Handle structured content
        if hasattr(res, "structuredContent") and res.structuredContent:
            try:
                return json.dumps(res.structuredContent, indent=2)
            except Exception as e:
                logger.warning(f"Failed to serialize structured content: {e}")

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
        """Clean up resources by closing all connected MCP clients and HTTP client."""
        # Clean up HTTP client with error suppression
        if getattr(self, "http_client", None):
            with contextlib.suppress(Exception):
                await self.http_client.close()

        if not self.tool_mgr:
            logger.warning("Tool manager not initialized during cleanup")
            return

        # Get connected clients from tool manager and close with error handling
        for client in self.tool_mgr.clients:
            with contextlib.suppress(Exception):
                await client.close()
                logger.debug(f"Closed client: {client.name}")

        logger.info("OpenAI orchestrator cleanup completed")

    async def apply_prompt(self, name: str, args: dict[str, str]) -> list[dict]:
        """Apply a parameterized prompt and return conversation messages."""
        if not self.tool_mgr:
            raise RuntimeError("Tool manager not initialized")

        res = await self.tool_mgr.get_prompt(name, args)

        return [
            {"role": m.role, "content": m.content.text}
            for m in res.messages
            if isinstance(m.content, types.TextContent)
        ]
