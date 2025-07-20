"""
Tool Schema Manager for MCP Platform

This module provides utilities for managing tool schemas, parameter validation,
and conversion between MCP and OpenAI formats using the official MCP SDK.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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

    def __init__(self, clients: list[MCPClient]) -> None:
        """Initialize the tool schema manager with MCP clients."""
        self.clients = clients
        self._tool_registry: dict[str, ToolInfo] = {}
        self._prompt_registry: dict[str, PromptInfo] = {}
        self._resource_registry: dict[str, ResourceInfo] = {}
        self._openai_tools: list[dict[str, Any]] = []

    async def initialize(self) -> None:
        """Initialize registries by collecting tools, prompts, and resources."""
        self._tool_registry.clear()
        self._prompt_registry.clear()
        self._resource_registry.clear()
        self._openai_tools.clear()

        for client in self.clients:
            await self._register_client_tools(client)
            await self._register_client_prompts(client)
            await self._register_client_resources(client)

        logger.info(
            f"Initialized registries with {len(self._tool_registry)} tools, "
            f"{len(self._prompt_registry)} prompts, "
            f"{len(self._resource_registry)} resources"
        )

    async def _register_client_tools(self, client: MCPClient) -> None:
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

    async def _register_client_prompts(self, client: MCPClient) -> None:
        """Register prompts from a specific MCP client."""
        try:
            prompts = await client.list_prompts()
            for prompt in prompts:
                # Handle prompt name conflicts
                prompt_name = prompt.name
                if prompt_name in self._prompt_registry:
                    logger.warning(
                        f"Prompt name conflict: '{prompt_name}' already exists"
                    )
                    prompt_name = f"{client.name}_{prompt_name}"

                # Store prompt info
                prompt_info = PromptInfo(prompt, client)
                self._prompt_registry[prompt_name] = prompt_info

            logger.info(
                f"Registered {len(prompts)} prompts from client '{client.name}'"
            )
        except Exception as e:
            logger.error(f"Error registering prompts from client '{client.name}': {e}")

    async def _register_client_resources(self, client: MCPClient) -> None:
        """Register resources from a specific MCP client."""
        try:
            resources = await client.list_resources()
            for resource in resources:
                # Handle resource URI conflicts
                resource_uri = str(resource.uri)
                if resource_uri in self._resource_registry:
                    logger.warning(
                        f"Resource URI conflict: '{resource_uri}' already exists"
                    )
                    # For resources, we don't rename - we use the full URI as key
                    resource_uri = f"{client.name}::{resource_uri}"

                # Store resource info
                resource_info = ResourceInfo(resource, client)
                self._resource_registry[resource_uri] = resource_info

            logger.info(
                f"Registered {len(resources)} resources from client '{client.name}'"
            )
        except Exception as e:
            logger.error(
                f"Error registering resources from client '{client.name}': {e}"
            )

    def _convert_to_openai_schema(self, tool: types.Tool) -> dict[str, Any]:
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

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Get all tools in OpenAI format."""
        return self._openai_tools.copy()

    def get_tool_info(self, tool_name: str) -> ToolInfo | None:
        """Get detailed information about a specific tool."""
        return self._tool_registry.get(tool_name)

    def get_prompt_info(self, prompt_name: str) -> PromptInfo | None:
        """Get detailed information about a specific prompt."""
        return self._prompt_registry.get(prompt_name)

    def get_resource_info(self, resource_uri: str) -> ResourceInfo | None:
        """Get detailed information about a specific resource."""
        return self._resource_registry.get(resource_uri)

    def list_available_prompts(self) -> list[str]:
        """Get list of all available prompt names."""
        return list(self._prompt_registry.keys())

    def list_available_resources(self) -> list[str]:
        """Get list of all available resource URIs."""
        return list(self._resource_registry.keys())

    def validate_tool_parameters(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate tool parameters using the SDK's native validation.

        This method uses the tool's inputSchema to validate parameters,
        leveraging Pydantic's validation mechanisms.
        """
        tool_info = self._tool_registry.get(tool_name)
        if not tool_info:
            raise McpError(
                error=types.ErrorData(
                    code=types.INVALID_PARAMS,
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
                    code=types.INVALID_PARAMS,
                    message=(
                        f"Parameter validation failed for tool '{tool_name}': "
                        f"{'; '.join(error_details)}"
                    ),
                )
            ) from e
        except Exception as e:
            logger.error(f"Error validating parameters for tool '{tool_name}': {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Parameter validation error: {str(e)}",
                )
            ) from e

    def _schema_to_pydantic_fields(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Convert JSON schema to Pydantic field definitions."""
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
                    field_type | None,
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
        self, tool_name: str, parameters: dict[str, Any]
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
                    code=types.INVALID_PARAMS,
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

    async def get_prompt(
        self, prompt_name: str, arguments: dict[str, Any] | None = None
    ) -> types.GetPromptResult:
        """
        Get a prompt with validated arguments.

        This method provides a complete prompt execution flow with:
        - Prompt lookup
        - Argument validation
        - Proper error handling
        """
        prompt_info = self._prompt_registry.get(prompt_name)
        if not prompt_info:
            raise McpError(
                error=types.ErrorData(
                    code=types.INVALID_PARAMS,
                    message=f"Prompt '{prompt_name}' not found",
                )
            )

        # Execute prompt via appropriate client
        try:
            return await prompt_info.client.get_prompt(
                prompt_info.prompt.name, arguments
            )
        except Exception as e:
            logger.error(f"Error getting prompt '{prompt_name}': {e}")
            raise

    async def read_resource(self, resource_uri: str) -> types.ReadResourceResult:
        """
        Read a resource by URI.

        This method provides a complete resource reading flow with:
        - Resource lookup
        - URI validation
        - Proper error handling
        """
        resource_info = self._resource_registry.get(resource_uri)
        if not resource_info:
            raise McpError(
                error=types.ErrorData(
                    code=types.INVALID_PARAMS,
                    message=f"Resource '{resource_uri}' not found",
                )
            )

        # Read resource via appropriate client
        try:
            return await resource_info.client.read_resource(resource_uri)
        except Exception as e:
            logger.error(f"Error reading resource '{resource_uri}': {e}")
            raise

    def get_tool_descriptions(self) -> list[str]:
        """Get human-readable descriptions of all available tools."""
        descriptions = []
        for tool_name, tool_info in self._tool_registry.items():
            tool = tool_info.tool
            desc = f"- {tool_name}: {tool.description or 'No description'}"
            descriptions.append(desc)
        return descriptions

    def get_prompt_descriptions(self) -> list[str]:
        """Get human-readable descriptions of all available prompts."""
        descriptions = []
        for prompt_name, prompt_info in self._prompt_registry.items():
            prompt = prompt_info.prompt
            desc = f"- {prompt_name}: {prompt.description or 'No description'}"
            descriptions.append(desc)
        return descriptions

    def get_resource_descriptions(self) -> list[str]:
        """Get human-readable descriptions of all available resources."""
        descriptions = []
        for resource_uri, resource_info in self._resource_registry.items():
            resource = resource_info.resource
            name = resource.name or resource_uri
            desc = f"- {name}: {resource.description or 'No description'}"
            descriptions.append(desc)
        return descriptions

    def export_schema_metadata(self) -> dict[str, Any]:
        """Export metadata about all registries."""
        return {
            "tool_count": len(self._tool_registry),
            "prompt_count": len(self._prompt_registry),
            "resource_count": len(self._resource_registry),
            "client_count": len(self.clients),
            "tools": list(self._tool_registry.keys()),
            "prompts": list(self._prompt_registry.keys()),
            "resources": list(self._resource_registry.keys()),
        }


class ToolInfo:
    """Information about a registered tool."""

    def __init__(
        self, tool: types.Tool, client: MCPClient, openai_schema: dict[str, Any]
    ):
        self.tool = tool  # Original MCP Tool object
        self.client = client  # Associated MCP client
        self.openai_schema = openai_schema  # Converted OpenAI schema


class PromptInfo:
    """Information about a registered prompt."""

    def __init__(self, prompt: types.Prompt, client: MCPClient):
        self.prompt = prompt  # Original MCP Prompt object
        self.client = client  # Associated MCP client


class ResourceInfo:
    """Information about a registered resource."""

    def __init__(self, resource: types.Resource, client: MCPClient):
        self.resource = resource  # Original MCP Resource object
        self.client = client  # Associated MCP client
