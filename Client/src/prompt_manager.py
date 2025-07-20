"""
Prompt Manager for MCP Platform

This module handles the management of system prompts, including:
- Default system prompts
- Tool-specific prompts
- Prompt extraction and formatting
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import mcp.types as types

if TYPE_CHECKING:
    from src.main import MCPClient
    from src.tool_schema_manager import ToolSchemaManager

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages system prompts and tool-specific prompts.

    This class is responsible for:
    - Managing default system prompts
    - Fetching tool-specific prompts
    - Extracting and formatting prompt content
    """

    def __init__(
        self, clients: List["MCPClient"], tool_schema_manager: "ToolSchemaManager"
    ) -> None:
        self.clients = clients
        self.tool_schema_manager = tool_schema_manager
        # Cache for tool-prompt mappings
        self._tool_prompts: Dict[str, str] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize prompt mappings."""
        if self._initialized:
            return

        # Map tools to their specific prompts
        for client in self.clients:
            try:
                prompts = await client.list_prompts()
                tools = await client.list_tools()

                # Create mapping of tool names to their prompts
                for prompt in prompts:
                    # Extract tool name from prompt name
                    if prompt.name.endswith("_tool_prompt"):
                        tool_name = prompt.name.replace("_tool_prompt", "")
                        # Verify this tool exists
                        if any(tool.name == tool_name for tool in tools):
                            prompt_result = await client.get_prompt(prompt.name, {})
                            has_messages = (
                                hasattr(prompt_result, "messages")
                                and prompt_result.messages
                            )
                            if (
                                has_messages
                                and prompt_result.messages[0].role == "assistant"
                            ):
                                content = self._extract_content_from_prompt_message(
                                    prompt_result.messages[0].content
                                )
                                self._tool_prompts[tool_name] = content
                                logger.info(f"Loaded prompt for tool: {tool_name}")
            except Exception as e:
                logger.warning(f"Could not initialize prompts from {client.name}: {e}")

        self._initialized = True
        logger.info(
            f"Initialized prompt manager with {len(self._tool_prompts)} "
            "tool-specific prompts"
        )

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt with available tools."""
        tool_descriptions = self.tool_schema_manager.get_tool_descriptions()
        tools_text = (
            "\n".join(tool_descriptions) if tool_descriptions else "No tools available"
        )

        return (
            f"You are a helpful assistant with access to the following tools:\n\n"
            f"{tools_text}\n\n"
            f"When a user asks you to perform a task that requires using one of "
            f"these tools, you should use the appropriate tool. For tasks that "
            f"don't require tools, provide direct assistance.\n\n"
            f"For tool-based tasks:\n"
            f"1. Choose the most appropriate tool\n"
            f"2. Use the tool with proper parameters\n"
            f"3. Present results clearly\n"
            f"4. Provide context and insights\n\n"
            f"For non-tool tasks:\n"
            f"1. Provide direct answers\n"
            f"2. Be clear and concise\n"
            f"3. Ask for clarification if needed"
        )

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

    async def get_system_message(self, tool_name: Optional[str] = None) -> str:
        """
        Get the appropriate system message based on tool usage.

        Args:
            tool_name: The name of the tool being used, if any.
                   If provided and a tool-specific prompt exists, returns that prompt.
                   If None or no tool-specific prompt exists, returns default prompt.
        """
        if not self._initialized:
            await self.initialize()

        # If we have a tool name and a specific prompt for it, use that
        if tool_name and tool_name in self._tool_prompts:
            return self._tool_prompts[tool_name]

        # Otherwise use the default prompt which already includes tool descriptions
        return self._get_default_system_prompt()
