"""
Main module for MCP client implementation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, McpError, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

from src.config import Configuration
from src.history.chat_store import JsonlRepo
from src.websocket_server import run_websocket_server

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MCPClient:
    """Official MCP Client implementation following SDK patterns."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self._reconnect_attempts: int = 0
        self._max_reconnect_attempts: int = 5
        self._reconnect_delay: float = 1.0
        self._is_connected: bool = False
        self.client_version = "0.1.0"

    def _resolve_command(self) -> str | None:
        """Resolve command to executable path with generic handling."""
        command = self.config.get("command")
        if not command:
            return None

        if os.path.isabs(command):
            return command if os.path.exists(command) else None

        resolved = shutil.which(command)

        if not resolved and command == "npx" and sys.platform == "win32":
            node_path = shutil.which("node")
            if node_path:
                logging.warning("Using node instead of npx on Windows")
                return node_path

        return resolved

    async def connect(self) -> None:
        """Connect to MCP server using official transport patterns."""
        while self._reconnect_attempts < self._max_reconnect_attempts:
            try:
                await self._attempt_connection()
                self._is_connected = True
                self._reconnect_attempts = 0
                return
            except Exception as e:
                self._reconnect_attempts += 1
                self._is_connected = False

                if self._reconnect_attempts >= self._max_reconnect_attempts:
                    logging.error(
                        f"Failed to connect to {self.name} after "
                        f"{self._max_reconnect_attempts} attempts: {e}"
                    )
                    raise

                logging.warning(
                    f"Connection attempt {self._reconnect_attempts} failed for "
                    f"{self.name}: {e}. Retrying in {self._reconnect_delay}s..."
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30.0)

    async def _attempt_connection(self) -> None:
        """Attempt a single connection to the MCP server."""
        command = self._resolve_command()

        if not command:
            raise ValueError(
                f"Command '{self.config.get('command')}' not found in PATH"
            )

        server_params = StdioServerParameters(
            command=command,
            args=self.config.get("args", []),
            env={**os.environ, **self.config.get("env", {})}
            if self.config.get("env")
            else None,
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = stdio_transport

        client_info = types.Implementation(name=self.name, version=self.client_version)

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream, client_info=client_info)
        )

        await asyncio.wait_for(self.session.initialize(), timeout=30.0)

        logging.info(f"MCP client '{self.name}' connected successfully")

    async def ping(self) -> bool:
        """Send ping to verify connection is alive."""
        if not self.session or not self._is_connected:
            return False

        try:
            await self.session.list_tools()
            return True
        except Exception as e:
            logging.warning(f"Ping failed for {self.name}: {e}")
            self._is_connected = False
            return False

    async def list_tools(self) -> list[types.Tool]:
        """List available tools using official SDK patterns."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            result = await self.session.list_tools()
            return result.tools
        except McpError as e:
            logging.error(
                f"MCP error listing tools from {self.name}: {e.error.message}"
            )
            raise
        except Exception as e:
            logging.error(f"Error listing tools from {self.name}: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Failed to list tools: {e!s}",
                )
            ) from e

    async def list_prompts(self) -> list[types.Prompt]:
        """List available prompts using official SDK patterns."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            result = await self.session.list_prompts()
            return result.prompts
        except McpError as e:
            logging.error(
                f"MCP error listing prompts from {self.name}: {e.error.message}"
            )
            raise
        except Exception as e:
            logging.error(f"Error listing prompts from {self.name}: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Failed to list prompts: {e!s}",
                )
            ) from e

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> types.GetPromptResult:
        """Get a prompt by name using official SDK patterns."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            return await self.session.get_prompt(name, arguments)
        except McpError as e:
            logging.error(
                f"MCP error getting prompt '{name}' from {self.name}: {e.error.message}"
            )
            raise
        except Exception as e:
            logging.error(f"Error getting prompt '{name}' from {self.name}: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Failed to get prompt '{name}': {e!s}",
                )
            ) from e

    async def list_resources(self) -> list[types.Resource]:
        """List available resources using official SDK patterns."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            result = await self.session.list_resources()
            return result.resources
        except McpError as e:
            logging.error(
                f"MCP error listing resources from {self.name}: {e.error.message}"
            )
            raise
        except Exception as e:
            logging.error(f"Error listing resources from {self.name}: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Failed to list resources: {e!s}",
                )
            ) from e

    async def read_resource(self, uri: str) -> types.ReadResourceResult:
        """Read a resource by URI using official SDK patterns."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            resource_uri = AnyUrl(uri)
            return await self.session.read_resource(resource_uri)
        except McpError as e:
            logging.error(
                f"MCP error reading resource '{uri}' from {self.name}: "
                f"{e.error.message}"
            )
            raise
        except Exception as e:
            logging.error(f"Error reading resource '{uri}' from {self.name}: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Failed to read resource '{uri}': {e!s}",
                )
            ) from e

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> types.CallToolResult:
        """Call a tool using official SDK patterns."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            logging.info(f"Calling tool '{name}' on client '{self.name}'")

            result = await self.session.call_tool(name, arguments)
            logging.info(f"Tool '{name}' executed successfully")
            return result
        except McpError as e:
            logging.error(f"MCP error calling tool '{name}': {e.error.message}")
            raise
        except Exception as e:
            logging.error(f"Error calling tool '{name}': {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Tool call failed: {e!s}",
                )
            ) from e

    async def get_tool_schemas(self) -> list[str]:
        """Get tool schemas as JSON strings using SDK's native methods."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            tools = await self.list_tools()
            return [tool.model_dump_json() for tool in tools]
        except Exception as e:
            logging.error(f"Error getting tool schemas from {self.name}: {e}")
            raise

    async def close(self) -> None:
        """Close the client connection and clean up resources."""
        async with self._cleanup_lock:
            try:
                self._is_connected = False
                await self.exit_stack.aclose()
                self.session = None
                logging.info(f"MCP client '{self.name}' disconnected")
            except Exception as e:
                logging.error(f"Error during cleanup of client {self.name}: {e}")

    @property
    def is_connected(self) -> bool:
        """Check if client is currently connected."""
        return self._is_connected


async def main() -> None:
    """Main entry point - WebSocket interface only."""
    config = Configuration()

    config_path = os.path.join(os.path.dirname(__file__), "servers_config.json")
    servers_config = config.load_config(config_path)

    clients = []
    for name, server_config in servers_config["mcpServers"].items():
        # Only create clients for enabled servers
        if server_config.get("enabled", False):
            clients.append(MCPClient(name, server_config))
        else:
            logging.info(f"Skipping disabled server: {name}")

    llm_config = config.get_full_llm_config()

    # Create repository for chat history
    repo = JsonlRepo("events.jsonl")

    await run_websocket_server(clients, llm_config, config.get_config_dict(), repo)


if __name__ == "__main__":
    asyncio.run(main())
