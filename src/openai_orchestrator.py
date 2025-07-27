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
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import httpx
from mcp import types

from src.history.chat_store import ChatEvent, ChatRepository, ToolCall, Usage
from src.history.persistence import ConversationPersistenceService
from src.mcp_services.prompting import MCPResourcePromptService

if TYPE_CHECKING:
    from src.main import MCPClient
    from src.tool_schema_manager import ToolSchemaManager
else:
    from src.tool_schema_manager import ToolSchemaManager

logger = logging.getLogger(__name__)

# Maximum number of tool call hops to prevent infinite recursion
MAX_TOOL_HOPS = 8


class ChatMessage:
    """
    Represents a chat message with metadata.
    """

    def __init__(
        self, mtype: str, content: str, meta: dict[str, Any] | None = None
    ):
        self.type = mtype
        self.content = content
        self.meta = meta or {}
        self.metadata = self.meta


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
        ctx_window: int = 4000,
    ):
        self.clients = clients
        self.llm_config = llm_config
        self.config = config
        self.repo = repo
        self.persist = ConversationPersistenceService(repo)
        self.ctx_window = ctx_window
        self.chat_conf = config.get("chat", {}).get("service", {})
        self.tool_mgr: ToolSchemaManager | None = None
        self._init_lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self.mcp_prompt: MCPResourcePromptService | None = None
        self._system_prompt: str = ""

        # Establish provider fields first
        self.active_name = (llm_config.get("active") or "").lower()
        providers = llm_config.get("providers") or {}
        self.provider_cfg = providers.get(self.active_name, {})

        # HTTP configuration with defaults
        http_config = llm_config.get("http", {})
        self.max_retries = http_config.get("max_retries", 3)
        self.initial_retry_delay = http_config.get("initial_retry_delay", 1.0)
        self.max_retry_delay = http_config.get("max_retry_delay", 16.0)
        self.retry_multiplier = http_config.get("retry_multiplier", 2.0)
        self.retry_jitter = http_config.get("retry_jitter", 0.1)

        # Now you can safely read api key and base_url
        self.api_key = self._extract_api_key()
        self.base_url = (self.provider_cfg.get("base_url") or "").rstrip("/")

        # Validate required configuration
        if not self.api_key:
            raise RuntimeError(
                f"Missing API key: set {self.active_name.upper()}_API_KEY or "
                f"providers['{self.active_name}']['api_key']"
            )
        if not self.base_url:
            raise RuntimeError("Missing base_url in provider config")

        # Configure HTTP client with reliability features
        timeout = self.provider_cfg.get("timeout", http_config.get("timeout", 30.0))
        max_keepalive = http_config.get("max_keepalive_connections", 20)
        max_connections = http_config.get("max_connections", 100)
        keepalive_expiry = http_config.get("keepalive_expiry", 30.0)

        self.http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "MCP-Platform/1.0",
            },
            # Connection pooling for better performance
            limits=httpx.Limits(
                max_keepalive_connections=max_keepalive,
                max_connections=max_connections,
                keepalive_expiry=keepalive_expiry,
            ),
        )

    def _extract_api_key(self) -> str:
        """Extract API key from env or provider config; raise if absent."""
        provider_name = (self.llm_config.get("active") or "").upper()
        env_key = f"{provider_name}_API_KEY"
        return (
            os.getenv(env_key)
            or self.provider_cfg.get("api_key", "")
        )

    async def _exponential_backoff_delay(self, attempt: int) -> None:
        """Calculate and apply exponential backoff delay with jitter."""
        if attempt == 0:
            return

        # Calculate delay: base * multiplier^(attempt-1)
        delay = min(
            self.initial_retry_delay * (self.retry_multiplier ** (attempt - 1)),
            self.max_retry_delay
        )

        # Add jitter to prevent thundering herd
        task_hash = hash(asyncio.current_task()) % 100
        jitter = delay * self.retry_jitter * (2 * task_hash / 100 - 1)
        delay += jitter

        logger.debug(f"Retrying in {delay:.2f}s (attempt {attempt})")
        await asyncio.sleep(delay)

    def _should_retry(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry."""
        if isinstance(error, httpx.HTTPStatusError):
            # Retry on rate limiting and server errors
            return error.response.status_code in (429, 500, 502, 503, 504)
        # Retry on connection and timeout errors
        return isinstance(error, httpx.ConnectError | httpx.TimeoutException)

    async def _make_http_request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make HTTP request with exponential backoff retry logic."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                await self._exponential_backoff_delay(attempt)

                response = await self.http.request(method, url, **kwargs)
                response.raise_for_status()
                return response

            except Exception as e:
                last_error = e

                if attempt == self.max_retries or not self._should_retry(e):
                    break

                logger.warning(
                    f"HTTP request failed (attempt {attempt + 1}/"
                    f"{self.max_retries + 1}): {e}"
                )

        # Re-raise the last error
        assert last_error is not None
        raise last_error

    async def _unified_post_request(
        self,
        payload: dict[str, Any],
        stream: bool = False,
    ) -> httpx.Response | AsyncGenerator[dict[str, Any]]:
        """Unified POST helper for both streaming and non-streaming requests."""
        if stream:
            payload["stream"] = True

        try:
            if stream:
                return self._stream_response_generator(payload)
            return await self._make_http_request_with_retry(
                "POST", "/chat/completions", json=payload
            )
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise

    async def _stream_response_generator(
        self, payload: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any]]:
        """Generate streaming responses with retry logic."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                await self._exponential_backoff_delay(attempt)

                async with self.http.stream(
                    "POST", "/chat/completions", json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                return
                            yield json.loads(data)
                return  # Successful completion

            except Exception as e:
                last_error = e

                if attempt == self.max_retries or not self._should_retry(e):
                    break

                logger.warning(
                    f"Streaming request failed (attempt {attempt + 1}/"
                    f"{self.max_retries + 1}): {e}"
                )

        # Re-raise the last error
        assert last_error is not None
        raise last_error

    @property
    def model(self) -> str:
        """Get the model name from the current configuration."""
        return self.provider_cfg.get("model", "")

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
                        f"Client '{self.clients[i].name}' failed to connect: {result}"
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
            self.tool_mgr = ToolSchemaManager(connected_clients)
            await self.tool_mgr.initialize()

            # Resource catalog & system prompt (MCP-only, provider-agnostic)
            self.mcp_prompt = MCPResourcePromptService(self.tool_mgr, self.chat_conf)
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

        return payload

    def _parse_openai_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Parse OpenAI API response."""
        choice = data["choices"][0]
        return {
            "message": choice["message"],
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
            "model": data.get("model", self.model),
        }

    async def _call_openai_api(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Make direct OpenAI API call with retry logic."""
        payload = self._build_openai_request(messages, tools)
        response = await self._unified_post_request(payload, stream=False)
        assert isinstance(response, httpx.Response)
        return self._parse_openai_response(response.json())

    async def _stream_openai_api(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncGenerator[dict[str, Any]]:
        """Stream responses from OpenAI API with retry logic."""
        payload = self._build_openai_request(messages, tools)
        response_gen = await self._unified_post_request(payload, stream=True)
        assert not isinstance(response_gen, httpx.Response)
        async for chunk in response_gen:
            yield chunk

    async def _build_conversation_history(
        self, conversation_id: str
    ) -> list[dict[str, Any]]:
        """Build conversation history including tool calls and results."""
        events = await self.repo.last_n_tokens(conversation_id, self.ctx_window)
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

        async for chunk in self._stream_openai_api(conv, tools_payload):
            if "choices" not in chunk or not chunk["choices"]:
                continue

            choice = chunk["choices"][0]
            delta = choice.get("delta", {})

            # Handle content streaming
            if content := delta.get("content"):
                message_buffer += content
                yield ChatMessage("text", content, {"type": "delta"})

            # Handle tool calls streaming
            if tool_calls := delta.get("tool_calls"):
                logger.info(f"Received tool call delta: {tool_calls}")
                self._accumulate_tool_calls(current_tool_calls, tool_calls)

            # Handle finish reason
            if "finish_reason" in choice:
                finish_reason = choice["finish_reason"]

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
            if "id" in tool_call:
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

        for call in calls:
            tool_name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"] or "{}")

            # 1. Persist the tool call event
            tool_call_event = ChatEvent(
                conversation_id=conversation_id,
                type="tool_call",
                role="assistant",  # Tool calls are initiated by assistant
                tool_calls=[ToolCall(
                    id=call["id"],
                    name=tool_name,
                    arguments=args
                )],
                extra={
                    "user_request_id": request_id,
                    "tool_call_id": call["id"]
                }
            )
            tool_call_event.compute_and_cache_tokens()
            await self.repo.add_event(tool_call_event)

            logger.info(f"Calling tool '{tool_name}' with args: {args}")

            # 2. Execute the tool
            result = await self.tool_mgr.call_tool(tool_name, args)

            # Handle structured content
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
                    "tool_name": tool_name
                },
                raw=result  # Keep full result for debugging
            )
            tool_result_event.compute_and_cache_tokens()
            await self.repo.add_event(tool_result_event)

            logger.info(f"Tool '{tool_name}' returned: {content[:100]}...")

            # 4. Add to conversation for immediate LLM context
            conv.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
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
        events = await self.repo.last_n_tokens(conversation_id, self.ctx_window)

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
        # Ensure token count is computed
        assistant_ev.compute_and_cache_tokens()
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

                result = await self.tool_mgr.call_tool(tool_name, args)
                content = self._pluck_content(result)

                conv.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
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

        # Truncate content if configured
        logging_config = self.chat_conf.get("logging", {})
        truncate_length = logging_config.get("llm_reply_truncate_length", 500)
        if content and len(content) > truncate_length:
            content = content[:truncate_length] + "..."

        log_parts = [f"LLM Reply ({context}):"]

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
        if self.http:
            await self.http.aclose()

        if not self.tool_mgr:
            raise RuntimeError("Tool manager not initialized")

        # Get connected clients from tool manager
        for client in self.tool_mgr.clients:
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"Error closing client {client.name}: {e}")

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
