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
    """
    Represents a chat message with metadata.

    PLAIN ENGLISH: This is just a container that holds a message (like "Hello!")
    along with extra info about it (like what type of message it is).
    """

    def __init__(
        self, mtype: str, content: str, meta: dict[str, Any] | None = None
    ):
        # The type of message (like "text", "error", "tool_execution")
        self.type = mtype
        # The actual message content (the words/text)
        self.content = content
        # Extra information about the message (like timestamps, etc.)
        self.meta = meta or {}
        # Keep old name for backwards compatibility with existing code
        self.metadata = self.meta


class ChatService:
    """
    Conversation orchestrator – recommended pattern
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
        # Store the connections to MCP servers (where tools live)
        self.clients = clients
        # Store the connection to the AI (like OpenAI)
        self.llm_client = llm_client
        # Store all the configuration settings
        self.config = config
        # Extract just the chat-related settings
        self.chat_conf = config.get("chat", {}).get("service", {})

        # This will manage all our tools/prompts/resources (starts as None)
        self.tool_mgr: ToolSchemaManager | None = None

        # These help ensure we only initialize once, even if called multiple times
        self._init_lock = asyncio.Lock()  # Prevents race conditions
        self._ready = asyncio.Event()     # Signals when we're ready to work

    # ─────────────────────────── Public API ───────────────────────────────────

    async def initialize(self) -> None:
        """
        Initialize the chat service and all MCP clients.

        1. Connects to all the MCP servers (where tools live)
        2. Discovers what tools/prompts/resources are available
        3. Builds a system prompt that tells the AI what it can do
        4. Gets everything ready for chatting
        """
        # Use a lock to prevent multiple initializations at once
        async with self._init_lock:
            # If we're already ready, don't do anything
            if self._ready.is_set():
                return

            # STEP 1: Connect to all MCP servers
            # We use "fail-soft" meaning if one server fails, others can still work
            await asyncio.gather(
                *(c.connect() for c in self.clients), return_exceptions=True
            )

            # STEP 2: Set up our tool manager
            # This discovers what tools/prompts/resources are available
            from src.tool_schema_manager import ToolSchemaManager  # late import

            self.tool_mgr = ToolSchemaManager(self.clients)
            await self.tool_mgr.initialize()  # This does the actual discovery

                        # STEP 3: Cache important information for fast access
            # Get list of all available resource URIs
            self._resource_catalog = sorted(
                self.tool_mgr.list_available_resources()
            )
            # Build the system prompt with actual resource contents
            self._system_prompt = await self._make_system_prompt()

            # Log what we found for debugging
            logger.info(
                "ChatService ready: %d tools, %d resources, %d prompts",
                len(self.tool_mgr.get_openai_tools()),
                len(self._resource_catalog),
                len(self.tool_mgr.list_available_prompts()),
            )

            # Debug: Log the resource catalog and system prompt
            logger.info("Resource catalog: %s", self._resource_catalog)
            logger.info("System prompt being used:\n%s", self._system_prompt)

            # Signal that we're ready to start chatting
            self._ready.set()

    async def cleanup(self) -> None:
        """
        Clean up all resources.

        This is like "shutting down" the chat system properly.
        It closes all connections to MCP servers so nothing is left hanging.
        """
        # Close connections to all MCP servers
        for c in self.clients:
            await c.close()
        # Mark ourselves as not ready anymore
        self._ready.clear()

    async def process_message(  # <- main entry
        self,
        user_msg: str,
        history: list[dict[str, Any]],
    ) -> AsyncGenerator[ChatMessage, None]:
        """
        Process a user message and yield response chunks.

        This is the main function that handles chatting!

        Here's what happens when you send a message:
        1. We prepare the conversation (your message + chat history +
           system instructions)
        2. We ask the AI for a response
        3. If the AI wants to use tools, we run those tools
        4. We get a final response and send it back to you

        We "yield" responses, meaning we can send you partial responses as they come in
        (like when you see text appearing word by word in ChatGPT).
        """
        # Wait until we're fully initialized before doing anything
        await self._ready.wait()

        # Make sure our tool manager is ready
        if not self.tool_mgr:
            raise RuntimeError("Tool manager not initialized")

        # ---- STEP 0: Prepare the conversation for the AI ----
        # This includes system instructions, chat history, and your new message
        conv = [
            {"role": "system", "content": self._system_prompt},  # Instructions for AI
            *history,                                           # Previous messages
            {"role": "user", "content": user_msg},             # Your new message
        ]

        # Get the list of tools the AI can use (in OpenAI format)
        tools_payload = self.tool_mgr.get_openai_tools()

        # ---- STEP 1: Ask the AI for its first response ----
        reply = await self.llm_client.get_response_with_tools(conv, tools_payload)
        assistant_msg = reply["message"]

        # If the AI said something, send it to you immediately
        if txt := assistant_msg.get("content"):
            yield ChatMessage(
                "text", txt, {"finish_reason": assistant_msg.get("finish_reason")}
            )

        # ---- STEP 2: Handle tool calls (if the AI wants to use tools) ----
        # The AI might want to use tools multiple times, so we loop
        while calls := assistant_msg.get("tool_calls"):
            # Add the AI's message (with tool calls) to the conversation
            conv.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.get("content") or "",
                    "tool_calls": calls,
                }
            )

            # Run each tool the AI requested
            for call in calls:
                tool_name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"] or "{}")

                # Actually run the tool and get results
                result = await self.tool_mgr.call_tool(tool_name, args)

                # Add the tool results to the conversation
                conv.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": self._pluck_text(result),
                    }
                )

            # Ask the AI to respond again, now that it has tool results
            assistant_msg = (
                await self.llm_client.get_response_with_tools(conv, tools_payload)
            )["message"]

            # Send the AI's final response to you
            if txt := assistant_msg.get("content"):
                yield ChatMessage(
                    "text", txt, {"finish_reason": assistant_msg.get("finish_reason")}
                )

    # ───────────────────────── Helper methods ────────────────────────────────

    async def _make_system_prompt(self) -> str:
        """
        Build the system prompt with actual resource contents and prompts.
        """
        # Start with the base personality/behavior instructions
        base = self.chat_conf["system_prompt"].rstrip()

        # Tool manager is guaranteed to exist at this point
        assert self.tool_mgr is not None

        # Add actual resource contents to the system prompt
        if self._resource_catalog:
            base += "\n\n**Available Resources:**"
            for uri in self._resource_catalog:
                try:
                    # Read the actual resource content
                    resource_result = await self.tool_mgr.read_resource(uri)
                    if resource_result.contents:
                        # Get resource info for better naming
                        resource_info = self.tool_mgr.get_resource_info(uri)
                        name = resource_info.resource.name if resource_info else uri

                        base += f"\n\n**{name}** ({uri}):"

                        # Add the actual resource contents
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

        # Add information about special prompts the AI can use
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

    @staticmethod
    def _pluck_text(res: types.CallToolResult) -> str:
        """
        Extract text content from a tool call result.

        PLAIN ENGLISH: When tools run, they return results in a complex format.
        This function just extracts the actual text content we care about,
        like pulling the meat out of a sandwich and ignoring the wrapper.
        """
        # If the tool didn't return anything, just say it's done
        if not res.content:
            return "✓ done"

        # Extract text from each piece of content
        out: list[str] = []
        for item in res.content:
            if isinstance(item, types.TextContent):
                # It's text, so grab the actual text
                out.append(item.text)
            else:
                # It's something else (like an image), so just describe what it is
                out.append(f"[{type(item).__name__}]")

        # Join all the text pieces together
        return "\n".join(out)

    # ───────────────────────── Utility methods ──────────────────────────────

    async def apply_prompt(self, name: str, args: dict[str, str]) -> list[dict]:
        """
        Apply a parameterized prompt and return conversation messages.

        This is a helper function that can use special
        "prompt templates".
        """
        # Make sure our tool manager is ready
        if not self.tool_mgr:
            raise RuntimeError("Tool manager not initialized")

        # Get the prompt with the provided arguments filled in
        res = await self.tool_mgr.get_prompt(name, args)

        # Convert the prompt messages to a format we can use in conversations
        return [
            {"role": m.role, "content": m.content.text}
            for m in res.messages
            if isinstance(m.content, types.TextContent)
        ]
