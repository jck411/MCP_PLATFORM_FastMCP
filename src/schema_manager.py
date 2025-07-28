"""
Schema Manager for MCP Platform

This module provides utilities for managing MCP schemas including tools, prompts,
and resources. It handles parameter validation, registry management, and provides
a unified interface for interacting with MCP entities using the official MCP SDK.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mcp import McpError, types
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic.json_schema import JsonSchemaValue

if TYPE_CHECKING:
    from src.main import MCPClient

logger = logging.getLogger(__name__)


class SchemaManager:
    """
    Manages MCP schemas and provides SDK-native conversion utilities.
    """

    def __init__(self, clients: list[MCPClient]) -> None:
        """Initialize the schema manager with MCP clients."""
        self.clients = clients
        self._tool_registry: dict[str, ToolInfo] = {}
        self._prompt_registry: dict[str, PromptInfo] = {}
        self._resource_registry: dict[str, ResourceInfo] = {}
        self._schema_cache: dict[str, type[BaseModel]] = {}

    async def initialize(self) -> None:
        """Initialize registries by collecting tools, prompts, and resources."""
        self._tool_registry.clear()
        self._prompt_registry.clear()
        self._resource_registry.clear()
        self._schema_cache.clear()

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
        # Skip clients that are not connected
        if not client.is_connected:
            logger.warning(
                f"Skipping tool registration for disconnected client '{client.name}'"
            )
            return

        try:
            tools = await client.list_tools()
            for tool in tools:
                tool_name = tool.name
                if tool_name in self._tool_registry:
                    logger.warning(f"Tool name conflict: '{tool_name}' already exists")
                    tool_name = f"{client.name}_{tool_name}"

                validation_model = self._create_validation_model(tool)

                if validation_model:
                    self._schema_cache[tool_name] = validation_model

                tool_info = ToolInfo(tool, client)
                self._tool_registry[tool_name] = tool_info

            logger.info(f"Registered {len(tools)} tools from client '{client.name}'")
        except Exception as e:
            logger.error(f"Error registering tools from client '{client.name}': {e}")
            raise

    async def _register_client_prompts(self, client: MCPClient) -> None:
        """Register prompts from a specific MCP client."""
        # Skip clients that are not connected
        if not client.is_connected:
            logger.warning(
                f"Skipping prompt registration for disconnected client '{client.name}'"
            )
            return

        try:
            prompts = await client.list_prompts()
            for prompt in prompts:
                prompt_name = prompt.name
                if prompt_name in self._prompt_registry:
                    logger.warning(
                        f"Prompt name conflict: '{prompt_name}' already exists"
                    )
                    prompt_name = f"{client.name}_{prompt_name}"

                prompt_info = PromptInfo(prompt, client)
                self._prompt_registry[prompt_name] = prompt_info

            logger.info(
                f"Registered {len(prompts)} prompts from client '{client.name}'"
            )
        except Exception as e:
            logger.error(f"Error registering prompts from client '{client.name}': {e}")
            raise

    async def _register_client_resources(self, client: MCPClient) -> None:
        """Register resources from a specific MCP client."""
        # Skip clients that are not connected
        if not client.is_connected:
            logger.warning(
                f"Skipping resource registration for disconnected "
                f"client '{client.name}'"
            )
            return

        try:
            resources = await client.list_resources()
            for resource in resources:
                resource_uri = str(resource.uri)
                if resource_uri in self._resource_registry:
                    logger.warning(
                        f"Resource URI conflict: '{resource_uri}' already exists"
                    )
                    resource_uri = f"{client.name}::{resource_uri}"

                resource_info = ResourceInfo(resource, client)
                self._resource_registry[resource_uri] = resource_info

            logger.info(
                f"Registered {len(resources)} resources from client '{client.name}'"
            )
        except Exception as e:
            logger.error(
                f"Error registering resources from client '{client.name}': {e}"
            )
            raise

    def _create_validation_model(self, tool: types.Tool) -> type[BaseModel] | None:
        """Create a Pydantic model for input validation."""
        if not tool.inputSchema:
            return None

        field_definitions = self._schema_to_pydantic_fields(tool.inputSchema)
        model_name = f"{tool.name}Params"
        return create_model(model_name, **field_definitions)

    def _schema_to_pydantic_fields(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Convert JSON schema to Pydantic field definitions."""

        def _process_schema(
            field_schema: JsonSchemaValue,
        ) -> tuple[type, dict[str, Any]]:
            if "type" not in field_schema:
                raise ValueError("Schema field must have a type")

            field_type = self._json_type_to_python_type(field_schema["type"])
            field_args = {
                "description": field_schema.get("description", ""),
                "title": field_schema.get("title"),
            }

            if "enum" in field_schema:
                field_args["enum"] = field_schema["enum"]

            if "format" in field_schema:
                field_args["format"] = field_schema["format"]

            if field_schema["type"] == "array" and "items" in field_schema:
                item_type, _ = _process_schema(field_schema["items"])
                field_type = list[item_type]

            if field_schema["type"] == "object" and "properties" in field_schema:
                nested_fields = self._schema_to_pydantic_fields(field_schema)
                field_type = create_model("NestedModel", **nested_fields)

            return field_type, field_args

        fields = {}
        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])

        for field_name, field_schema in properties.items():
            field_type, field_args = _process_schema(field_schema)
            is_required = field_name in required_fields

            if is_required:
                fields[field_name] = (field_type, Field(**field_args))
            else:
                fields[field_name] = (
                    field_type | None,
                    Field(default=None, **field_args),
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
        if json_type not in type_mapping:
            raise ValueError(f"Unsupported JSON schema type: {json_type}")
        return type_mapping[json_type]

    async def validate_tool_parameters(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate tool parameters."""
        tool_info = self._tool_registry.get(tool_name)
        if not tool_info:
            raise McpError(
                error=types.ErrorData(
                    code=types.INVALID_PARAMS,
                    message=f"Tool '{tool_name}' not found",
                )
            )

        validation_model = self._schema_cache.get(tool_name)
        if not validation_model:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"No validation model found for tool '{tool_name}'",
                )
            )

        try:
            validated = validation_model(**parameters)
            return validated.model_dump()
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                error_details.append(
                    f"{error['loc'][0] if error['loc'] else 'root'}: {error['msg']}"
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

    def get_mcp_tools(self) -> list[dict[str, Any]]:
        """Get all tools in native MCP format (no conversion)."""
        mcp_tools = []
        for tool_info in self._tool_registry.values():
            mcp_tools.append({
                "name": tool_info.tool.name,
                "description": tool_info.tool.description or "No description provided",
                "inputSchema": tool_info.tool.inputSchema,
            })
        return mcp_tools

    def get_tool_info(self, tool_name: str) -> ToolInfo:
        """Get detailed information about a specific tool."""
        tool_info = self._tool_registry.get(tool_name)
        if not tool_info:
            raise McpError(
                error=types.ErrorData(
                    code=types.INVALID_PARAMS,
                    message=f"Tool '{tool_name}' not found",
                )
            )
        return tool_info

    def get_prompt_info(self, prompt_name: str) -> PromptInfo:
        """Get detailed information about a specific prompt."""
        prompt_info = self._prompt_registry.get(prompt_name)
        if not prompt_info:
            raise McpError(
                error=types.ErrorData(
                    code=types.INVALID_PARAMS,
                    message=f"Prompt '{prompt_name}' not found",
                )
            )
        return prompt_info

    def get_resource_info(self, resource_uri: str) -> ResourceInfo:
        """Get detailed information about a specific resource."""
        resource_info = self._resource_registry.get(resource_uri)
        if not resource_info:
            raise McpError(
                error=types.ErrorData(
                    code=types.INVALID_PARAMS,
                    message=f"Resource '{resource_uri}' not found",
                )
            )
        return resource_info

    def list_available_prompts(self) -> list[str]:
        """Get list of all available prompt names."""
        return list(self._prompt_registry.keys())

    def list_available_resources(self) -> list[str]:
        """Get list of all available resource URIs."""
        return list(self._resource_registry.keys())

    async def call_tool(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> types.CallToolResult:
        """Call a tool with validated parameters."""
        tool_info = self.get_tool_info(tool_name)
        validated_params = await self.validate_tool_parameters(tool_name, parameters)
        return await tool_info.client.call_tool(tool_info.tool.name, validated_params)

    async def get_prompt(
        self, prompt_name: str, arguments: dict[str, Any] | None = None
    ) -> types.GetPromptResult:
        """Get a prompt with validated arguments."""
        prompt_info = self.get_prompt_info(prompt_name)
        return await prompt_info.client.get_prompt(prompt_info.prompt.name, arguments)

    async def read_resource(self, resource_uri: str) -> types.ReadResourceResult:
        """Read a resource by URI."""
        resource_info = self.get_resource_info(resource_uri)
        return await resource_info.client.read_resource(resource_uri)


class ToolInfo:
    """Information about a registered tool."""

    def __init__(self, tool: types.Tool, client: MCPClient):
        self.tool = tool
        self.client = client


class PromptInfo:
    """Information about a registered prompt."""

    def __init__(self, prompt: types.Prompt, client: MCPClient):
        self.prompt = prompt
        self.client = client


class ResourceInfo:
    """Information about a registered resource."""

    def __init__(self, resource: types.Resource, client: MCPClient):
        self.resource = resource
        self.client = client
