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

if TYPE_CHECKING:
    from src.main import LLMClient, MCPClient
    from src.tool_schema_manager import ToolSchemaManager
else:
    from src.tool_schema_manager import ToolSchemaManager

logger = logging.getLogger(__name__)


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
    ):
        self.clients = clients
        self.llm_client = llm_client
        self.config = config
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
                raise RuntimeError("No MCP clients successfully connected")

            logger.info(
                f"Successfully connected to {len(connected_clients)} out of "
                f"{len(self.clients)} MCP clients"
            )

            # Use only connected clients for tool management
            self.tool_mgr = ToolSchemaManager(connected_clients)
            await self.tool_mgr.initialize()

            self._resource_catalog = sorted(
                self.tool_mgr.list_available_resources()
            )
            self._system_prompt = await self._make_system_prompt()

            logger.info(
                "ChatService ready: %d tools, %d resources, %d prompts",
                len(self.tool_mgr.get_openai_tools()),
                len(self._resource_catalog),
                len(self.tool_mgr.list_available_prompts()),
            )

            logger.info("Resource catalog: %s", self._resource_catalog)
            logger.info("System prompt being used:\n%s", self._system_prompt)

            self._ready.set()

    async def process_message(
        self,
        user_msg: str,
        history: list[dict[str, Any]],
    ) -> AsyncGenerator[ChatMessage]:
        """Process a user message and yield response chunks."""
        await self._ready.wait()

        if not self.tool_mgr:
            raise RuntimeError("Tool manager not initialized")

        conv = [
            {"role": "system", "content": self._system_prompt},
            *history,
            {"role": "user", "content": user_msg},
        ]

        tools_payload = self.tool_mgr.get_openai_tools()

        reply = await self.llm_client.get_response_with_tools(conv, tools_payload)
        assistant_msg = reply["message"]

        if txt := assistant_msg.get("content"):
            yield ChatMessage(
                "text", txt, {"finish_reason": assistant_msg.get("finish_reason")}
            )

        while calls := assistant_msg.get("tool_calls"):
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

                # Handle structured content
                content = self._pluck_content(result)

                conv.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": content,
                    }
                )

            assistant_msg = (
                await self.llm_client.get_response_with_tools(conv, tools_payload)
            )["message"]

            if txt := assistant_msg.get("content"):
                yield ChatMessage(
                    "text", txt, {"finish_reason": assistant_msg.get("finish_reason")}
                )

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

        if self._resource_catalog:
            base += "\n\n**Available Resources:**"
            for uri in self._resource_catalog:
                try:
                    resource_result = await self.tool_mgr.read_resource(uri)
                    if resource_result.contents:
                        resource_info = self.tool_mgr.get_resource_info(uri)
                        name = resource_info.resource.name if resource_info else uri

                        base += f"\n\n**{name}** ({uri}):"

                        for content in resource_result.contents:
                            if isinstance(content, types.TextResourceContents):
                                lines = content.text.strip().split('\n')
                                for line in lines:
                                    base += f"\n{line}"
                            elif isinstance(content, types.BlobResourceContents):
                                base += f"\n[Binary content: {len(content.blob)} bytes]"
                            else:
                                base += f"\n[{type(content).__name__} available]"
                except Exception as e:
                    logger.warning(f"Could not read resource {uri}: {e}")
                    base += f"\n\n**{uri}**: Resource temporarily unavailable"

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

    async def cleanup(self) -> None:
        """Clean up resources by closing all connected MCP clients."""
        # Only cleanup clients that are actually connected
        if hasattr(self, 'tool_mgr') and self.tool_mgr:
            # Get connected clients from tool manager
            for client in self.tool_mgr.clients:
                try:
                    await client.close()
                except Exception as e:
                    logger.warning(f"Error closing client {client.name}: {e}")
        else:
            # Fallback: try to close all clients if tool_mgr isn't available
            for client in self.clients:
                try:
                    if client.is_connected:
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
