from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
from contextlib import AsyncExitStack
from typing import Any

import httpx
import mcp.types as types
from mcp import ClientSession, McpError, StdioServerParameters
from mcp.client.stdio import stdio_client
from src.config import Configuration

# Configure logging
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

        # Set our application's version identifier for MCP server identification
        self.client_version = "0.1.0"

    async def connect(self) -> None:
        """Connect to MCP server using official transport patterns."""
        while self._reconnect_attempts < self._max_reconnect_attempts:
            try:
                await self._attempt_connection()
                self._reconnect_attempts = 0  # Reset on successful connection
                self._is_connected = True
                return
            except Exception as e:
                self._reconnect_attempts += 1
                if self._reconnect_attempts >= self._max_reconnect_attempts:
                    logging.error(
                        f"Failed to connect MCP client {self.name} after "
                        f"{self._max_reconnect_attempts} attempts: {e}"
                    )
                    raise

                # Exponential backoff with jitter
                delay = min(
                    self._reconnect_delay * (2**self._reconnect_attempts), 60.0
                )
                logging.warning(
                    f"Connection attempt {self._reconnect_attempts} failed for "
                    f"{self.name}, retrying in {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)

    async def _attempt_connection(self) -> None:
        """Attempt a single connection to the MCP server."""
        # Validate and resolve command with Windows npx handling
        command = self.config.get("command")
        if command == "npx":
            if sys.platform == "win32":
                # On Windows, npx can leak stdio pipes. Warn and use node directly.
                logging.warning(
                    "Using npx on Windows may leak stdio pipes. "
                    "Consider using node directly."
                )
                node_path = shutil.which("node")
                # Try to use node directly if available, otherwise use npx
                command = node_path or shutil.which("npx")
            else:
                command = shutil.which("npx")
        else:
            command = shutil.which(command) if command else None

        if not command:
            raise ValueError("The command must be a valid string and cannot be None.")

        # Create server parameters following official patterns
        server_params = StdioServerParameters(
            command=command,
            args=self.config.get("args", []),
            env={**os.environ, **self.config.get("env", {})}
            if self.config.get("env")
            else None,
        )

        # Use official stdio_client helper
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = stdio_transport

        # Create client info for session with dynamic version
        client_info = types.Implementation(name=self.name, version=self.client_version)

        # Create session with proper transport handling
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream, client_info=client_info)
        )

        # Initialize session with explicit protocol version and timeout
        try:
            # Try to use InitializationOptions for SDK 1.12+ compatibility
            init_options = types.InitializationOptions(
                protocolVersion=types.LATEST_PROTOCOL_VERSION
            )
            await asyncio.wait_for(self.session.initialize(init_options), timeout=30.0)
        except (TypeError, AttributeError):
            # Fallback for older SDK versions
            await asyncio.wait_for(self.session.initialize(), timeout=30.0)

        # Send initial ping to verify connection
        try:
            await self.session.ping()
        except AttributeError:
            # Fallback for older SDK versions
            await self.session.send_ping()

        logging.info(f"MCP client '{self.name}' connected successfully")

    async def ping(self) -> bool:
        """Send ping to verify connection is alive."""
        if not self.session or not self._is_connected:
            return False

        try:
            try:
                await self.session.ping()
            except AttributeError:
                # Fallback for older SDK versions
                await self.session.send_ping()
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
                    code=types.ErrorCode.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            result = await self.session.list_tools()
            return result.tools
        except McpError as e:
            # Re-raise MCP errors with proper error data
            logging.error(
                f"MCP error listing tools from {self.name}: {e.error.message}"
            )
            raise
        except Exception as e:
            # Convert other exceptions to MCP errors
            logging.error(f"Error listing tools from {self.name}: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to list tools: {str(e)}",
                )
            ) from e

    async def list_prompts(self) -> list[types.Prompt]:
        """List available prompts using official SDK patterns."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.ErrorCode.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            result = await self.session.list_prompts()
            return result.prompts
        except McpError as e:
            # Re-raise MCP errors with proper error data
            logging.error(
                f"MCP error listing prompts from {self.name}: {e.error.message}"
            )
            raise
        except Exception as e:
            # Convert other exceptions to MCP errors
            logging.error(f"Error listing prompts from {self.name}: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to list prompts: {str(e)}",
                )
            )

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> types.GetPromptResult:
        """Get a prompt by name using official SDK patterns."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.ErrorCode.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            return await self.session.get_prompt(name, arguments)
        except McpError as e:
            # Re-raise MCP errors with proper error data
            logging.error(
                f"MCP error getting prompt '{name}' from {self.name}: {e.error.message}"
            )
            raise
        except Exception as e:
            # Convert other exceptions to MCP errors
            logging.error(f"Error getting prompt '{name}' from {self.name}: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to get prompt '{name}': {str(e)}",
                )
            )

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> types.CallToolResult:
        """Call a tool using official SDK patterns."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.ErrorCode.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            # Log tool execution without exposing sensitive arguments
            logging.info(f"Calling tool '{name}' on client '{self.name}'")

            result = await self.session.call_tool(name, arguments)
            logging.info(f"Tool '{name}' executed successfully")
            return result
        except McpError as e:
            # Re-raise MCP errors with proper error data
            logging.error(f"MCP error calling tool '{name}': {e.error.message}")
            raise
        except Exception as e:
            # Convert other exceptions to MCP errors
            logging.error(f"Error calling tool '{name}': {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.ErrorCode.INTERNAL_ERROR,
                    message=f"Tool call failed: {str(e)}",
                )
            )

    async def get_tool_schemas(self) -> list[str]:
        """Get tool schemas as JSON strings using SDK's native methods."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.ErrorCode.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            tools = await self.list_tools()
            # Use model_dump_json() to get JSON strings as expected by consumers
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


class LLMClient:
    """HTTP client for LLM API requests with structured tool call support."""

    def __init__(self, config: dict[str, Any], api_key: str) -> None:
        self.config: dict[str, Any] = config
        self.api_key: str = api_key
        self.client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=config["base_url"],
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    async def get_response_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Get response from LLM API with structured tool calls support."""
        try:
            payload = {
                "model": self.config["model"],
                "messages": messages,
                "temperature": self.config.get("temperature", 0.7),
                "max_tokens": self.config.get("max_tokens", 4096),
                "top_p": self.config.get("top_p", 1.0),
            }

            if tools:
                payload["tools"] = tools

            response = await self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            result = response.json()

            # Return the complete choice object to preserve tool_calls and other
            # metadata
            choice = result["choices"][0]
            return {
                "message": choice["message"],
                "finish_reason": choice.get("finish_reason"),
                "index": choice.get("index", 0),
            }
        except httpx.HTTPError as e:
            logging.error(f"HTTP error: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.ErrorCode.INTERNAL_ERROR, message=f"HTTP error: {str(e)}"
                )
            )
        except KeyError as e:
            logging.error(f"Unexpected response format: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.ErrorCode.PARSE_ERROR,
                    message=f"Unexpected response format: {str(e)}",
                )
            )
        except Exception as e:
            logging.error(f"LLM API error: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.ErrorCode.INTERNAL_ERROR,
                    message=f"LLM API error: {str(e)}",
                )
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


async def main() -> None:
    """Main entry point - WebSocket interface only."""
    config = Configuration()

    # Load server configurations
    config_path = os.path.join(os.path.dirname(__file__), "servers_config.json")
    servers_config = config.load_config(config_path)

    # Create MCP clients
    clients = []
    for name, server_config in servers_config["mcpServers"].items():
        clients.append(MCPClient(name, server_config))

    # Create LLM client
    llm_config = config.get_llm_config()
    api_key = config.llm_api_key

    # Use async context manager to ensure proper cleanup
    async with LLMClient(llm_config, api_key) as llm_client:
        # Start WebSocket server
        from src.websocket_server import run_websocket_server

        await run_websocket_server(clients, llm_client, config.get_config_dict())


if __name__ == "__main__":
    asyncio.run(main())
