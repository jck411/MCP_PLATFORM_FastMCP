"""
Tool Schema Manager for MCP Platform

This module provides utilities for managing tool schemas, parameter validation,
and conversion between MCP and OpenAI formats using the official MCP SDK.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import mcp.types as types
from mcp import McpError
from pydantic import ValidationError

if TYPE_CHECKING:
    from src.main import MCPClient

logger = logging.getLogger(__name__)


class ToolSchemaManager:
    """
    Manages tool schemas and provides SDK-native conversion utilities.

    This class leverages the MCP SDK's native Pydantic models to:
    - Extract schemas using built-in methods
    - Validate parameters using Pydantic validation
    - Preserve metadata during conversion
    - Provide proper error handling
    """

    def __init__(self, clients: List["MCPClient"]) -> None:
        """Initialize the tool schema manager with MCP clients."""
        self.clients = clients
        self._tool_registry: Dict[str, "ToolInfo"] = {}
        self._openai_tools: List[Dict[str, Any]] = []

    async def initialize(self) -> None:
        """Initialize tool registry by collecting tools from all clients."""
        self._tool_registry.clear()
        self._openai_tools.clear()

        for client in self.clients:
            await self._register_client_tools(client)

        logger.info(f"Initialized tool registry with {len(self._tool_registry)} tools")

    async def _register_client_tools(self, client: "MCPClient") -> None:
        """Register tools from a specific MCP client."""
        try:
            tools = await client.list_tools()
            for tool in tools:
                # Handle tool name conflicts
                tool_name = tool.name
                if tool_name in self._tool_registry:
                    logger.warning(f"Tool name conflict: '{tool_name}' already exists")
                    tool_name = f"{client.name}_{tool_name}"

                # Convert to OpenAI format
                openai_schema = self._convert_to_openai_schema(tool)

                # Store tool info
                tool_info = ToolInfo(tool, client, openai_schema)
                self._tool_registry[tool_name] = tool_info
                self._openai_tools.append(openai_schema)

            logger.info(f"Registered {len(tools)} tools from client '{client.name}'")
        except Exception as e:
            logger.error(f"Error registering tools from client '{client.name}': {e}")

    def _convert_to_openai_schema(self, tool: types.Tool) -> Dict[str, Any]:
        """
        Convert MCP Tool to OpenAI format using SDK's native schema methods.

        This method uses the Tool's built-in Pydantic methods to extract schema
        data, preserving metadata and ensuring consistency.
        """
        # Use Pydantic's model_dump to get complete tool data as dict
        tool_data = tool.model_dump(exclude_none=True)

        # Build OpenAI schema with preserved metadata
        openai_schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "No description provided",
                "parameters": tool.inputSchema
                or {"type": "object", "properties": {}, "required": []},
            },
        }

        # Preserve additional metadata if available
        if tool.title:
            openai_schema["function"]["title"] = tool.title

        if tool.annotations:
            openai_schema["function"]["annotations"] = tool.annotations.model_dump()

        if tool.outputSchema:
            openai_schema["function"]["output_schema"] = tool.outputSchema

        # Add MCP-specific metadata
        openai_schema["function"]["_mcp_metadata"] = {
            "original_tool": tool_data,
            "sdk_version": "1.11.0+",
            # Store JSON representation for consumers that expect it
            "json_schema": tool.model_dump_json(),
        }

        return openai_schema

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI format."""
        return self._openai_tools.copy()

    def get_tool_info(self, tool_name: str) -> Optional["ToolInfo"]:
        """Get detailed information about a specific tool."""
        return self._tool_registry.get(tool_name)

    def validate_tool_parameters(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate tool parameters using the SDK's native validation.

        This method uses the tool's inputSchema to validate parameters,
        leveraging Pydantic's validation mechanisms.
        """
        tool_info = self._tool_registry.get(tool_name)
        if not tool_info:
            raise McpError(
                error=types.ErrorData(
                    code=types.ErrorCode.INVALID_PARAMS,
                    message=f"Tool '{tool_name}' not found",
                )
            )

        try:
            # Use the tool's inputSchema for validation
            schema = tool_info.tool.inputSchema

            # Create a temporary Pydantic model for validation
            from pydantic import create_model

            # Convert JSON schema to Pydantic field definitions
            field_definitions = self._schema_to_pydantic_fields(schema)

            # Create dynamic model for validation
            ValidationModel = create_model(f"{tool_name}Params", **field_definitions)

            # Validate parameters
            validated_params = ValidationModel(**parameters)
            return validated_params.model_dump()

        except ValidationError as e:
            error_details = []
            for error in e.errors():
                error_details.append(
                    f"{error['loc'][0] if error['loc'] else 'root'}: " f"{error['msg']}"
                )

            raise McpError(
                error=types.ErrorData(
                    code=types.ErrorCode.INVALID_PARAMS,
                    message=(
                        f"Parameter validation failed for tool '{tool_name}': "
                        f"{'; '.join(error_details)}"
                    ),
                )
            )
        except Exception as e:
            logger.error(f"Error validating parameters for tool '{tool_name}': {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.ErrorCode.INTERNAL_ERROR,
                    message=f"Parameter validation error: {str(e)}",
                )
            )

    def _schema_to_pydantic_fields(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON schema to Pydantic field definitions."""
        from typing import Union

        from pydantic import Field

        fields = {}
        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])

        for field_name, field_schema in properties.items():
            field_type = self._json_type_to_python_type(
                field_schema.get("type", "string")
            )
            is_required = field_name in required_fields

            field_description = field_schema.get("description", "")

            if is_required:
                fields[field_name] = (field_type, Field(description=field_description))
            else:
                fields[field_name] = (
                    Union[field_type, None],
                    Field(default=None, description=field_description),
                )

        return fields

    def _json_type_to_python_type(self, json_type: str) -> type:
        """Convert JSON schema type to Python type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        return type_mapping.get(json_type, str)

    async def call_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> types.CallToolResult:
        """
        Call a tool with validated parameters.

        This method provides a complete tool execution flow with:
        - Parameter validation
        - Proper error handling
        - Metadata preservation
        """
        tool_info = self._tool_registry.get(tool_name)
        if not tool_info:
            raise McpError(
                error=types.ErrorData(
                    code=types.ErrorCode.INVALID_PARAMS,
                    message=f"Tool '{tool_name}' not found",
                )
            )

        # Validate parameters
        validated_params = self.validate_tool_parameters(tool_name, parameters)

        # Execute tool via appropriate client
        try:
            return await tool_info.client.call_tool(
                tool_info.tool.name, validated_params
            )
        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}': {e}")
            raise

    def get_tool_descriptions(self) -> List[str]:
        """Get human-readable descriptions of all available tools."""
        descriptions = []
        for tool_name, tool_info in self._tool_registry.items():
            tool = tool_info.tool
            desc = f"- {tool_name}: {tool.description or 'No description'}"
            descriptions.append(desc)
        return descriptions

    def export_schema_metadata(self) -> Dict[str, Any]:
        """Export metadata about the tool registry."""
        return {
            "tool_count": len(self._tool_registry),
            "client_count": len(self.clients),
            "tools": list(self._tool_registry.keys()),
        }


class ToolInfo:
    """Information about a registered tool."""

    def __init__(
        self, tool: types.Tool, client: "MCPClient", openai_schema: Dict[str, Any]
    ):
        self.tool = tool  # Original MCP Tool object
        self.client = client  # Associated MCP client
        self.openai_schema = openai_schema  # Converted OpenAI schema
