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
from mcp import McpError

if TYPE_CHECKING:
    from src.main import LLMClient, MCPClient

from src.tool_schema_manager import ToolSchemaManager

logger = logging.getLogger(__name__)


class ChatMessage:
    """Represents a chat message with metadata."""

    def __init__(
        self, message_type: str, content: Any, metadata: dict[str, Any] | None = None
    ):
        self.type = message_type
        self.content = content
        self.metadata = metadata or {}


class ChatService:
    """
    Main chat service that handles conversation logic and MCP coordination.

    This service is responsible for:
    - Managing conversation state with a built-in default system prompt
    - Coordinating between MCP clients and LLM
    - Executing tools and processing results
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
        
        # Extract chat service specific config
        self.chat_config = config.get("chat", {}).get("service", {})
        
        self.tool_schema_manager = ToolSchemaManager(clients)
        self._initialized = False

    def _build_system_prompt(self) -> str:
        """Build the system prompt using configuration."""
        return self.chat_config["system_prompt"]
 
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

    async def _connect_with_health_check(self, client: MCPClient) -> None:
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
        1. Use default system prompt
        2. Get LLM response
        3. If tool is called, execute tool and get final response
        """
        if not self._initialized:
            raise RuntimeError("Chat service not initialized")

        # Use default system prompt
        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(),
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

            # Check if tool notifications are enabled
            tool_notifications = self.chat_config["tool_notifications"]
            if tool_notifications["enabled"]:
                # Get configurable notification format
                icon = tool_notifications["icon"]
                format_template = tool_notifications["format"]
                
                notification_text = format_template.format(
                    icon=icon,
                    tool_name=tool_name,
                    tool_args=(
                        tool_args if tool_notifications["show_args"] else ""
                    )
                )
                
                # Yield tool execution notification
                if self.chat_config["logging"]["tool_execution"]:
                    logger.info(
                        f"Yielding tool execution message for tool: {tool_name}"
                    )
                
                yield ChatMessage(
                    "tool_execution",
                    notification_text,
                    {"tool_name": tool_name, "tool_args": tool_args},
                )

            # Execute the tool
            tool_result = await self._execute_tool_call(tool_name, tool_args)
            
            # Log tool result if logging is enabled
            if self.chat_config["logging"]["tool_execution"]:
                truncate_length = self.chat_config["logging"]["result_truncate_length"]
                logger.info(
                    f"Tool result for {tool_name}: {tool_result[:truncate_length]}..."
                )

            # Add tool result to message history
            messages.append(
                {
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call["id"],
                }
            )

        # Get final response from LLM (keeping the same system prompt)
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
            if self.chat_config["logging"]["tool_execution"]:
                logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")
            
            result = await self.tool_schema_manager.call_tool(tool_name, tool_args)
            
            if self.chat_config["logging"]["tool_execution"]:
                logger.info(f"Raw tool result: {result}")
            
            extracted_content = self._extract_content_from_result(result)
            
            if self.chat_config["logging"]["tool_execution"]:
                truncate_length = self.chat_config["logging"]["result_truncate_length"]
                logger.info(
                    f"Extracted content: {extracted_content[:truncate_length]}..."
                )
            
            return extracted_content
        except McpError as e:
            error_msg = f"MCP error executing tool '{tool_name}': {e.error.message}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _get_system_message(self) -> str:
        """Get system message directly."""
        return self._build_system_prompt()

    def _extract_content_from_result(self, result: types.CallToolResult) -> str:
        """Extract text content from a tool call result."""
        if not result.content:
            return "Tool executed successfully (no content returned)"

        content_parts = []
        for content_item in result.content:
            # Check the actual type and extract accordingly
            if isinstance(content_item, types.TextContent):
                content_parts.append(content_item.text)
            elif isinstance(content_item, types.ImageContent):
                content_parts.append(f"[Image: {content_item.mimeType}]")
            elif isinstance(content_item, types.AudioContent):
                content_parts.append(f"[Audio: {content_item.mimeType}]")
            else:
                # Handle other content types safely
                data = getattr(content_item, "data", None)
                if data is not None:
                    try:
                        content_parts.append(f"[Binary data: {len(data)} bytes]")
                    except (TypeError, AttributeError):
                        content_parts.append(f"[{type(content_item).__name__}]")
                else:
                    content_parts.append(str(content_item))

        return "\n".join(content_parts) if content_parts else "No readable content"
