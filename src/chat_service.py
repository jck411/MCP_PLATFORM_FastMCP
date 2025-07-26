"""
Chat Service for MCP Platform

This module handles the business logic for chat sessions, including:
- Conversation management with simple default prompts
- Tool orchestration
- MCP client coordination
- LLM interactions
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from mcp import types

from src.history.chat_store import ChatEvent, ChatRepository, Usage

if TYPE_CHECKING:
    from src.main import LLMClient, MCPClient
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


class ChatService:
    """
    Conversation orchestrator - recommended pattern
    1. Takes your message
    2. Figures out what tools might be needed
    3. Asks the AI to respond (and use tools if needed)
    4. Sends you back the response
    """

    def __init__(
        self,
        clients: list[MCPClient],
        llm_client: LLMClient,
        config: dict[str, Any],
        repo: ChatRepository,
        ctx_window: int = 4000,
    ):
        self.clients = clients
        self.llm_client = llm_client
        self.config = config
        self.repo = repo
        self.ctx_window = ctx_window
        self.chat_conf = config.get("chat", {}).get("service", {})
        self.tool_mgr: ToolSchemaManager | None = None
        self._init_lock = asyncio.Lock()
        self._ready = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the chat service and all MCP clients."""
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

            # Update resource catalog to only include available resources
            await self._update_resource_catalog_on_availability()
            self._system_prompt = await self._make_system_prompt()

            logger.info(
                "ChatService ready: %d tools, %d resources, %d prompts",
                len(self.tool_mgr.get_openai_tools()),
                len(self._resource_catalog),
                len(self.tool_mgr.list_available_prompts()),
            )

            logger.info("Resource catalog: %s", self._resource_catalog)

            # Configurable system prompt logging
            if self.chat_conf.get("logging", {}).get("system_prompt", True):
                logger.info("System prompt being used:\n%s", self._system_prompt)
            else:
                logger.debug("System prompt logging disabled in configuration")

            self._ready.set()

    async def process_message(
        self,
        user_msg: str,
        history: list[dict[str, Any]],
    ) -> AsyncGenerator[ChatMessage]:
        """
        Process a user message and yield response chunks.
        REQUIRES streaming LLM client.
        """
        await self._ready.wait()

        if not self.tool_mgr:
            raise RuntimeError("Tool manager not initialized")

        # FAIL FAST: Ensure LLM client supports streaming
        if not hasattr(self.llm_client, 'get_streaming_response_with_tools'):
            raise RuntimeError(
                "LLM client does not support streaming. "
                "Use chat_once() for non-streaming responses."
            )

        conv = [
            {"role": "system", "content": self._system_prompt},
            *history,
            {"role": "user", "content": user_msg},
        ]

        tools_payload = self.tool_mgr.get_openai_tools()

        # Initial response - stream and collect
        assistant_msg: dict[str, Any] = {}
        async for chunk in self._stream_llm_response(conv, tools_payload):
            if isinstance(chunk, ChatMessage):
                yield chunk
            else:
                assistant_msg = chunk

        # Log LLM reply if configured
        reply_data = {
            "message": assistant_msg,
            "usage": None,  # Usage not available in streaming mode
            "model": self.llm_client.config.get("model", "")
        }
        self._log_llm_reply(reply_data, "Streaming initial response")

        # Handle tool calls in a loop
        hops = 0
        while calls := assistant_msg.get("tool_calls"):
            if hops >= MAX_TOOL_HOPS:
                logger.warning(
                    f"Maximum tool hops ({MAX_TOOL_HOPS}) reached, stopping recursion"
                )
                yield ChatMessage(
                    "text",
                    f"⚠️ Reached maximum tool call limit ({MAX_TOOL_HOPS}). "
                    "Stopping to prevent infinite recursion.",
                    {"finish_reason": "tool_limit_reached"},
                )
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
            await self._execute_tool_calls(conv, calls)

            # Get follow-up response - stream and collect
            assistant_msg = {}
            async for chunk in self._stream_llm_response(conv, tools_payload):
                if isinstance(chunk, ChatMessage):
                    yield chunk
                else:
                    assistant_msg = chunk

            # Log LLM reply if configured
            reply_data = {
                "message": assistant_msg,
                "usage": None,  # Usage not available in streaming mode
                "model": self.llm_client.config.get("model", "")
            }
            context = f"Streaming tool follow-up (hop {hops + 1})"
            self._log_llm_reply(reply_data, context)

            hops += 1

    async def _stream_llm_response(
        self, conv: list[dict[str, Any]], tools_payload: list[dict[str, Any]]
    ) -> AsyncGenerator[ChatMessage | dict[str, Any]]:
        """Stream response from LLM and yield chunks to user."""
        message_buffer = ""
        current_tool_calls: list[dict[str, Any]] = []
        finish_reason: str | None = None

        async for chunk in self.llm_client.get_streaming_response_with_tools(
            conv, tools_payload
        ):
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
                self._accumulate_tool_calls(current_tool_calls, tool_calls)

            # Handle finish reason
            if "finish_reason" in choice:
                finish_reason = choice["finish_reason"]

        # Yield complete assistant message as final item
        yield {
            "content": message_buffer or None,
            "tool_calls": current_tool_calls if current_tool_calls and any(
                call["function"]["name"] for call in current_tool_calls
            ) else None,
            "finish_reason": finish_reason
        }

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
        self, conv: list[dict[str, Any]], calls: list[dict[str, Any]]
    ) -> None:
        """Execute tool calls and add results to conversation."""
        assert self.tool_mgr is not None

        for call in calls:
            tool_name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"] or "{}")

            result = await self.tool_mgr.call_tool(tool_name, args)

            # Handle structured content
            content = self._pluck_content(result)

            conv.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": content,
                }
            )

    async def _get_existing_assistant_response(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None:
        """Get existing assistant response for a request_id if it exists."""
        existing_assistant_id = await self.repo.get_last_assistant_reply_id(
            conversation_id, request_id
        )
        if existing_assistant_id:
            events = await self.repo.get_events(conversation_id)
            for event in events:
                if event.id == existing_assistant_id:
                    return event
        return None

    async def chat_once(
        self,
        conversation_id: str,
        user_msg: str,
        request_id: str,
        model: str | None = None,
    ) -> ChatEvent:
        """Non-streaming: persist user msg first, call LLM once, persist assistant."""
        await self._ready.wait()

        if not self.tool_mgr:
            raise RuntimeError("Tool manager not initialized")

        # Check for existing response to prevent double-billing
        existing_response = await self._get_existing_assistant_response(
            conversation_id, request_id
        )
        if existing_response:
            logger.info(
                f"Returning cached response for request_id: {request_id}"
            )
            return existing_response

        # 1) persist user message FIRST with idempotency check
        user_ev = ChatEvent(
            conversation_id=conversation_id,
            type="user_message",
            role="user",
            content=user_msg,
            extra={"request_id": request_id},
        )
        # Ensure token count is computed
        user_ev.compute_and_cache_tokens()
        was_added = await self.repo.add_event(user_ev)

        if not was_added:
            # User message already exists, check for assistant response again
            existing_response = await self._get_existing_assistant_response(
                conversation_id, request_id
            )
            if existing_response:
                logger.info(
                    "Returning existing response for duplicate "
                    f"request_id: {request_id}"
                )
                return existing_response
            # No assistant response yet, continue with LLM call

        # 2) build canonical history from repo
        events = await self.repo.last_n_tokens(conversation_id, self.ctx_window)

        # 3) convert to OpenAI-style list for your LLM call
        conv: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
        ]
        for ev in events:
            if ev.type in ("user_message", "assistant_message", "system_update"):
                conv.append({"role": ev.role, "content": ev.content})

        conv.append({"role": "user", "content": user_msg})

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

        tools_payload = self.tool_mgr.get_openai_tools()

        assistant_full_text = ""
        total_usage = Usage()
        model = ""

        reply = await self.llm_client.get_response_with_tools(conv, tools_payload)
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

            reply = await self.llm_client.get_response_with_tools(conv, tools_payload)
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

    async def _make_system_prompt(self) -> str:
        """Build the system prompt with actual resource contents and prompts."""
        base = self.chat_conf["system_prompt"].rstrip()

        assert self.tool_mgr is not None

        # Only include resources that are actually available
        available_resources = await self._get_available_resources()
        if available_resources:
            base += "\n\n**Available Resources:**"
            for uri, content_info in available_resources.items():
                resource_info = self.tool_mgr.get_resource_info(uri)
                name = resource_info.resource.name if resource_info else uri

                base += f"\n\n**{name}** ({uri}):"

                for content in content_info:
                    if isinstance(content, types.TextResourceContents):
                        lines = content.text.strip().split('\n')
                        for line in lines:
                            base += f"\n{line}"
                    elif isinstance(content, types.BlobResourceContents):
                        base += f"\n[Binary content: {len(content.blob)} bytes]"
                    else:
                        base += f"\n[{type(content).__name__} available]"

        prompt_names = self.tool_mgr.list_available_prompts()
        if prompt_names:
            prompt_list = []
            for name in prompt_names:
                pinfo = self.tool_mgr.get_prompt_info(name)
                if pinfo:
                    desc = pinfo.prompt.description or "No description available"
                    prompt_list.append(f"• **{name}**: {desc}")

            prompts_text = "\n".join(prompt_list)
            base += (
                f"\n\n**Available Prompts** (use apply_prompt method):\n"
                f"{prompts_text}"
            )

        return base

    async def _get_available_resources(
        self,
    ) -> dict[str, list[types.TextResourceContents | types.BlobResourceContents]]:
        """
        Check resource availability and return only resources that can be read
        successfully.

        This implements graceful degradation by only including working resources
        in the system prompt, following best practices for resource management.
        """
        available_resources: dict[
            str, list[types.TextResourceContents | types.BlobResourceContents]
        ] = {}

        if not self._resource_catalog or not self.tool_mgr:
            return available_resources

        for uri in self._resource_catalog:
            try:
                resource_result = await self.tool_mgr.read_resource(uri)
                if resource_result.contents:
                    # Only include resources that have actual content
                    available_resources[uri] = resource_result.contents
                    logger.debug(f"Resource {uri} is available and loaded")
                else:
                    logger.debug(f"Resource {uri} has no content, skipping")
            except Exception as e:
                # Log the error but don't include in system prompt
                # This prevents the LLM from being told about broken resources
                logger.warning(
                    f"Resource {uri} is unavailable and will be excluded "
                    f"from system prompt: {e}"
                )
                continue

        if available_resources:
            logger.info(
                f"Including {len(available_resources)} available resources "
                f"in system prompt"
            )
        else:
            logger.info(
                "No resources are currently available - system prompt will not "
                "include resource section"
            )

        return available_resources

    async def _update_resource_catalog_on_availability(self) -> None:
        """
        Update the resource catalog to reflect current availability.

        This implements a circuit-breaker-like pattern where we periodically
        check if previously failed resources have become available again.
        """
        if not self.tool_mgr:
            return

        # Get all registered resources from the tool manager
        all_resource_uris = self.tool_mgr.list_available_resources()

        # Filter to only include resources that are actually available
        available_uris = []
        for uri in all_resource_uris:
            try:
                resource_result = await self.tool_mgr.read_resource(uri)
                if resource_result.contents:
                    available_uris.append(uri)
            except Exception:
                # Skip unavailable resources silently
                continue

        # Update the catalog to only include working resources
        self._resource_catalog = available_uris
        logger.debug(
            f"Updated resource catalog: {len(available_uris)} of "
            f"{len(all_resource_uris)} resources are available"
        )

    async def cleanup(self) -> None:
        """Clean up resources by closing all connected MCP clients."""
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
