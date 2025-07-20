# MCP Platform Copilot Instructions

This workspace contains an MCP (Model Context Protocol) chatbot platform that connects to various MCP servers and uses LLM APIs for responses.

## Development Environment & Tools

### Package Management
- **ALWAYS** use `uv` for dependency management
- **ALWAYS** use `uv run` to execute Python commands
- Never use `pip` or other package managers

### Code Quality Standards
- **REQUIRED**: Use Pydantic for all data models and validation
- **REQUIRED**: Include comprehensive type hints on all functions, methods, and variables
- **REQUIRED**: Follow fail-fast principle - no fallbacks or silent error handling
- **REQUIRED**: When adding new code, remove all legacy/deprecated code immediately

### Testing Philosophy
- Create test code only when explicitly requested
- **ALWAYS** delete test code immediately after verification
- Never commit temporary test files

## Project Architecture

### Core Structure
```
src/
├── main.py              # Main MCP client application
├── chat_service.py      # Chat orchestration and session management
├── config.py           # Configuration management
├── tool_schema_manager.py # Tool schema validation and conversion
├── websocket_server.py  # WebSocket communication layer
└── servers_config.json  # MCP server configurations

Servers/
└── (ready for new FastMCP servers)
```

### Key Components

#### Configuration Management (`config.py`)
- Manages environment variables and server configurations
- Uses Pydantic for validation
- Loads from `.env` and configuration files

#### MCP Client Integration (`main.py`)
- Implements MCP client connections
- Handles server lifecycle management
- Manages tool discovery and execution

#### Tool Schema Management (`tool_schema_manager.py`)
- Converts MCP tool schemas to OpenAI format
- Validates tool parameters using Pydantic
- Manages tools, prompts, and resources across multiple servers
- Provides error handling with proper MCP error codes

#### Chat Service (`chat_service.py`)
- Orchestrates conversation flow
- Integrates with LLM APIs (Groq)
- Manages chat sessions and context

#### WebSocket Server (`websocket_server.py`)
- Provides real-time communication interface
- Handles client connections and message routing

## MCP SDK Integration

### Reference Documentation
- Primary: https://modelcontextprotocol.io/llms-full.txt
- Use official MCP SDK types and patterns

### Error Handling
- Use `mcp.types.INVALID_PARAMS`, `mcp.types.INTERNAL_ERROR` (not `ErrorCode.X`)
- Always chain exceptions with `from e` or `from None`
- Fail fast - no silent error recovery

### Type Usage
- Import types from `mcp.types`
- Use modern union syntax: `str | None` instead of `Union[str, None]`
- Leverage Pydantic models for validation

## Code Style Guidelines

### Imports
```python
from __future__ import annotations  # Always first
import standard_library
import third_party
import mcp.types as types
from local_modules import items
```

### Error Handling Pattern
```python
try:
    # operation
except SpecificError as e:
    logger.error(f"Specific error: {e}")
    raise McpError(
        error=types.ErrorData(
            code=types.INVALID_PARAMS,
            message="Clear error message",
        )
    ) from e
```

### Function Signatures
```python
async def function_name(
    param1: str,
    param2: int | None = None,
) -> ReturnType:
    """Clear docstring describing purpose."""
```

## Environment Setup
- Uses `pyproject.toml` for project configuration
- Environment variables in `.env`
- All dependencies managed through `uv`


