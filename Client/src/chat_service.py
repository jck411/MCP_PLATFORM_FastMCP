"""
Chat Service for MCP Platform

This module handles the business logic for chat sessions, including:
- Conversation management
- Tool orchestration
- MCP client coordination
- LLM interactions
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

import mcp.types as types
from mcp import McpError

if TYPE_CHECKING:
    from src.main import LLMClient, MCPClient

from src.prompt_manager import PromptManager
from src.tool_schema_manager import ToolSchemaManager

logger = logging.getLogger(__name__)


class ChatMessage:
    """Represents a chat message with metadata."""

    def __init__(
        self, message_type: str, content: Any, metadata: Dict[str, Any] = None
    ):
        self.type = message_type
        self.content = content
        self.metadata = metadata or {}


class ChatService:
    """
    Main chat service that handles conversation logic and MCP coordination.

    This service is responsible for:
    - Managing conversation state
    - Coordinating between MCP clients and LLM
    - Executing tools and processing results
    - Building system messages
    """

    def __init__(
        self,
        clients: list["MCPClient"],
        llm_client: "LLMClient",
        config: dict[str, Any],
    ):
        self.clients = clients
        self.llm_client = llm_client
        self.config = config
        self.tool_schema_manager = ToolSchemaManager(clients)
        self.prompt_manager = PromptManager(clients, self.tool_schema_manager)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the chat service and all MCP clients."""
        if self._initialized:
            return

        # Connect all MCP clients with health monitoring
        connection_tasks = []
        for client in self.clients:
            connection_tasks.append(self._connect_with_health_check(client))

        # Wait for all connections to complete
        await asyncio.gather(*connection_tasks)

        # Initialize tool schema manager
        await self.tool_schema_manager.initialize()

        # Log initialization
        metadata = self.tool_schema_manager.export_schema_metadata()
        logger.info(
            f"Chat service initialized with {metadata['tool_count']} tools "
            f"from {metadata['client_count']} clients"
        )

        self._initialized = True

    async def _connect_with_health_check(self, client: "MCPClient") -> None:
        """Connect a client and perform health check."""
        try:
            await client.connect()

            # Perform health check
            if await client.ping():
                logger.info(f"Client '{client.name}' health check passed")
            else:
                logger.warning(f"Client '{client.name}' health check failed")

        except Exception as e:
            logger.error(f"Failed to connect client '{client.name}': {e}")
            # Don't raise - allow service to start with partial client connectivity

    async def cleanup(self) -> None:
        """Clean up all resources."""
        for client in self.clients:
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"Error closing client '{client.name}': {e}")

        self._initialized = False

    async def process_message(
        self, user_message: str, conversation_history: list[dict[str, Any]]
    ) -> AsyncGenerator[ChatMessage, None]:
        """
        Process a user message and yield response chunks.

        This method handles the complete conversation flow:
        1. Start with default system prompt
        2. Get LLM response
        3. If tool is called:
           - Execute tool
           - Switch to tool-specific prompt if available
           - Get final response
        4. Next message starts fresh with default prompt
        """
        if not self._initialized:
            raise RuntimeError("Chat service not initialized")

        # Start with default system prompt
        messages = [
            {
                "role": "system",
                "content": await self.prompt_manager.get_system_message(),
            }
        ]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        # Get available tools
        tools = self.tool_schema_manager.get_openai_tools()

        try:
            # Get LLM response
            llm_response = await self.llm_client.get_response_with_tools(
                messages, tools if tools else None
            )

            # Handle structured response format
            message = llm_response.get("message", {})
            tool_calls = message.get("tool_calls", [])

            # Yield text content if available
            if message.get("content"):
                yield ChatMessage(
                    "text",
                    message["content"],
                    {"finish_reason": llm_response.get("finish_reason")},
                )

            # Process tool calls if any
            if tool_calls:
                # Add assistant message with tool calls to history
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.get("content") or "",
                        "tool_calls": tool_calls,
                    }
                )

                async for tool_message in self._process_tool_calls(
                    tool_calls, messages
                ):
                    yield tool_message

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            yield ChatMessage(
                "error",
                f"An error occurred while processing your message: {str(e)}",
                {"error_type": type(e).__name__},
            )

    async def _process_tool_calls(
        self, tool_calls: list[dict[str, Any]], messages: list[dict[str, Any]]
    ) -> AsyncGenerator[ChatMessage, None]:
        """Process tool calls and yield results."""

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            # Yield tool execution notification
            logger.info(f"Yielding tool execution message for tool: {tool_name}")
            yield ChatMessage(
                "tool_execution",
                f"🔧 Executing tool: {tool_name}",
                {"tool_name": tool_name, "tool_args": tool_args},
            )

            # Execute the tool
            tool_result = await self._execute_tool_call(tool_name, tool_args)
            logger.info(f"Tool result for {tool_name}: {tool_result[:200]}...")

            # Add tool result to message history
            messages.append(
                {
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call["id"],
                }
            )

        # After all tools are executed, check if first tool has a specific prompt
        first_tool = tool_calls[0]["function"]["name"]
        tool_prompt = await self.prompt_manager.get_system_message(first_tool)

        # Update system message if tool has specific prompt
        messages[0] = {"role": "system", "content": tool_prompt}

        # Get final response from LLM with updated prompt
        try:
            tools = self.tool_schema_manager.get_openai_tools()
            final_response = await self.llm_client.get_response_with_tools(
                messages, tools if tools else None
            )

            # Handle the final response
            final_message = final_response.get("message", {})
            if final_message.get("content"):
                yield ChatMessage(
                    "text",
                    final_message["content"],
                    {"finish_reason": final_response.get("finish_reason")},
                )

        except Exception as e:
            logger.error(f"Error getting final response after tool calls: {e}")
            yield ChatMessage(
                "error",
                f"Error getting final response: {str(e)}",
                {"error_type": type(e).__name__},
            )

    async def _execute_tool_call(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> str:
        """Execute a tool call and return the result."""
        try:
            logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")
            result = await self.tool_schema_manager.call_tool(tool_name, tool_args)
            logger.info(f"Raw tool result: {result}")
            extracted_content = self._extract_content_from_result(result)
            logger.info(f"Extracted content: {extracted_content[:200]}...")
            return extracted_content
        except McpError as e:
            error_msg = f"MCP error executing tool '{tool_name}': {e.error.message}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def _get_system_message(self) -> str:
        """Get system message from prompt manager."""
        return await self.prompt_manager.get_system_message()

    def _get_active_tool_from_history(
        self, conversation_history: list[dict[str, Any]]
    ) -> Optional[str]:
        """
        Determine the active tool from conversation history.

        This looks at the most recent assistant message with tool calls
        to determine which tool is currently being used.
        """
        # Look through history in reverse to find most recent tool usage
        for message in reversed(conversation_history):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                # Get the tool name from the first tool call
                tool_calls = message["tool_calls"]
                if tool_calls and len(tool_calls) > 0:
                    return tool_calls[0]["function"]["name"]
        return None

    def _extract_content_from_result(self, result: types.CallToolResult) -> str:
        """Extract text content from a tool call result."""
        if not result.content:
            return "Tool executed successfully (no content returned)"

        content_parts = []
        for content_item in result.content:
            if hasattr(content_item, "text"):
                content_parts.append(content_item.text)
            elif hasattr(content_item, "data"):
                # Handle binary data
                content_parts.append(f"[Binary data: {len(content_item.data)} bytes]")
            else:
                content_parts.append(str(content_item))

        return "\n".join(content_parts) if content_parts else "No readable content"

    def _extract_content_from_prompt_message(
        self, content: types.PromptMessage | types.TextContent | types.ImageContent
    ) -> str:
        """Extract text content from a prompt message."""
        if hasattr(content, "text"):
            return content.text
        elif hasattr(content, "content"):
            # Handle nested content structures
            if isinstance(content.content, list):
                text_parts = []
                for item in content.content:
                    if hasattr(item, "text"):
                        text_parts.append(item.text)
                return "\n".join(text_parts)
            elif hasattr(content.content, "text"):
                return content.content.text
        return str(content)
