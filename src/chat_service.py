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

import mcp.types as types

if TYPE_CHECKING:
    from src.main import LLMClient, MCPClient
    from src.tool_schema_manager import ToolSchemaManager

logger = logging.getLogger(__name__)


class ChatMessage:
    """Represents a chat message with metadata."""

    def __init__(
        self, mtype: str, content: str, meta: dict[str, Any] | None = None
    ):
        self.type = mtype
        self.content = content
        self.meta = meta or {}
        # Maintain backward compatibility with existing code
        self.metadata = self.meta


class ChatService:
    """Conversation orchestrator – recommended pattern (no virtual prompt tools)."""

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

    # ─────────────────────────── Public API ───────────────────────────────────

    async def initialize(self) -> None:
        """Initialize the chat service and all MCP clients."""
        async with self._init_lock:
            if self._ready.is_set():
                return

            # 1. connect clients (fail‑soft)
            await asyncio.gather(
                *(c.connect() for c in self.clients), return_exceptions=True
            )

            # 2. build managers / caches
            from src.tool_schema_manager import ToolSchemaManager  # late import

            self.tool_mgr = ToolSchemaManager(self.clients)
            await self.tool_mgr.initialize()

            # 3. prime immutable data
            self._resource_catalog = sorted(
                self.tool_mgr.get_resource_descriptions()
            )
            self._system_prompt = self._make_system_prompt()

            logger.info(
                "ChatService ready: %d tools, %d resources, %d prompts",
                len(self.tool_mgr.get_openai_tools()),
                len(self._resource_catalog),
                len(self.tool_mgr.list_available_prompts()),
            )
            self._ready.set()

    async def cleanup(self) -> None:
        """Clean up all resources."""
        for c in self.clients:
            await c.close()
        self._ready.clear()

    async def process_message(  # <- main entry
        self,
        user_msg: str,
        history: list[dict[str, Any]],
    ) -> AsyncGenerator[ChatMessage, None]:
        """Process a user message and yield response chunks."""
        await self._ready.wait()

        # Ensure tool_mgr is available
        if not self.tool_mgr:
            raise RuntimeError("Tool manager not initialized")

        # ---- 0 · prepare conversation ---------------------------------------
        conv = [
            {"role": "system", "content": self._system_prompt},
            *history,
            {"role": "user", "content": user_msg},
        ]

        tools_payload = self.tool_mgr.get_openai_tools()

        # ---- 1 · first LLM pass ---------------------------------------------
        reply = await self.llm_client.get_response_with_tools(conv, tools_payload)
        assistant_msg = reply["message"]

        # Emit text content if available
        if txt := assistant_msg.get("content"):
            yield ChatMessage(
                "text", txt, {"finish_reason": assistant_msg.get("finish_reason")}
            )

        # ---- 2 · handle tool calls loop -------------------------------------
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
                conv.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": self._pluck_text(result),
                    }
                )

            assistant_msg = (
                await self.llm_client.get_response_with_tools(conv, tools_payload)
            )["message"]

            # Emit text content if available
            if txt := assistant_msg.get("content"):
                yield ChatMessage(
                    "text", txt, {"finish_reason": assistant_msg.get("finish_reason")}
                )

    # ───────────────────────── Helper methods ────────────────────────────────

    def _make_system_prompt(self) -> str:
        """Static system prompt containing resource catalogue & house rules."""
        base = self.chat_conf["system_prompt"].rstrip()

        # Ensure tool_mgr is available
        if not self.tool_mgr:
            return base

        if self._resource_catalog:
            cat = "\n".join(f"• {uri}" for uri in self._resource_catalog)
            base += (
                "\n\nAvailable knowledge base resources **(read‑only)**\n"
                "Call the appropriate MCP tool if you need any of them:\n"
                f"{cat}"
            )

        # inject *static* prompts that require no arguments, once:
        for name in self.tool_mgr.list_available_prompts():
            pinfo = self.tool_mgr.get_prompt_info(name)
            if pinfo and not pinfo.prompt.arguments:  # no params → safe to inline
                try:
                    # Create new event loop for sync context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    fetched = loop.run_until_complete(
                        self.tool_mgr.get_prompt(name)
                    )
                    loop.close()

                    msgs = "\n".join(
                        m.content.text
                        for m in fetched.messages
                        if isinstance(m.content, types.TextContent)
                    )
                    base += f"\n\n### Reference prompt «{name}»\n{msgs}"
                except Exception as e:
                    logger.warning(f"Failed to inline prompt '{name}': {e}")
        return base

    @staticmethod
    def _pluck_text(res: types.CallToolResult) -> str:
        """Extract text content from a tool call result."""
        if not res.content:
            return "✓ done"
        out: list[str] = []
        for item in res.content:
            if isinstance(item, types.TextContent):
                out.append(item.text)
            else:
                out.append(f"[{type(item).__name__}]")
        return "\n".join(out)

    # ───────────────────────── Utility methods ──────────────────────────────

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
