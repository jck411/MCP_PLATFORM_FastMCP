# MCP Platform

A Model Context Protocol (MCP) client with dual interface support (terminal and WebSocket) that connects to MCP servers for tool execution.

## Features

- **Pure MCP SDK Implementation**: Uses official MCP SDK types and patterns exclusively
- **Clean Architecture**: No legacy code, built with official MCP Client patterns
- **Official Tool Schema**: Direct use of `types.Tool`, `types.Prompt`, and `types.CallToolResult`
- **Multi-LLM Support**: OpenAI, Groq, Anthropic, Azure, OpenRouter via config
- **Dual Interface**: Terminal chat or WebSocket server for frontend integration

## Available Tools

The platform comes with two built-in tools:

### 1. Website Fetcher (`fetch`)
- Fetches and analyzes web content
- Handles different content types:
  - **HTML**: Extracts title, description, and main content
  - **JSON**: Returns parsed JSON data
  - **Other**: Returns content type and metadata
- Example:
  ```json
  {
    "url": "https://example.com",
    "title": "Example Domain",
    "description": "Page description",
    "content_preview": ["Main content paragraphs..."],
    "content_type": "html"
  }
  ```

### 2. Calculator (`calculate`)
- Performs basic mathematical calculations
- Supports operations: +, -, *, /, %, ^ (power)
- Handles parentheses for complex expressions
- Example:
  ```json
  // Input
  {"expression": "2 * (3 + 4) ^ 2"}

  // Output
  {
    "expression": "2 * (3 + 4) ^ 2",
    "result": "98"
  }
  ```

## Tool-Specific Prompts

Tools can have specialized prompts that activate when the tool is used:

1. **Default System Prompt**:
   - Used for initial interactions
   - Lists all available tools
   - Guides tool selection

2. **Fetch Tool Prompt**:
   - Activates when fetching websites
   - Provides specialized guidance for:
     - URL validation
     - Content type handling
     - Result presentation

3. **Per-Message Flow**:
   - Each message starts with default prompt
   - Tool usage triggers specialized prompts
   - Next message starts fresh

## Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Configure API key:**
   ```bash
   echo "LLM_API_KEY=your_key_here" > .env
   ```

3. **Run:**
   ```bash
   uv run python -m src.main
   ```

## Configuration

Edit `Client/src/config.yaml`:

```yaml
chat:
  interface: "websocket"  # or "terminal"
  websocket:
    host: "localhost"
    port: 8000

llm:
  active: "openai"  # openai, groq, anthropic, azure, openrouter
  providers:
    openai:
      model: "gpt-4o"
      temperature: 0.7
```

## Pure MCP SDK Implementation

The platform uses official MCP SDK exclusively:
- **Official MCP Client**: Direct use of `MCPClient` following official SDK patterns
- **Official Types**: Direct use of `types.Tool`, `types.Prompt`, `types.CallToolResult`
- **Official Transport**: Uses `stdio_client` helpers and proper transport handling
- **No Custom Wrappers**: Tools and prompts use SDK types without modification
- **SDK-Native Patterns**: All interactions follow official MCP SDK patterns

## Architecture

The platform follows clean MCP SDK patterns:

- **MCPClient**: Official MCP client implementation using proper transport handling
- **ChatSession**: Orchestrates user/LLM/MCP client interactions
- **LLMClient**: Provider-agnostic HTTP client for LLM APIs
- **WebSocket Server**: Real-time interface for web frontends

## Dual Message System

The platform handles hybrid LLM responses following MCP SDK patterns:

```
User: "Tell me a joke and fetch https://example.com"

Message 1: "Here's a joke: Why don't skeletons fight? They don't have the guts!"
Message 2: [Tool execution result with fetched content]
```

- **Text-only responses**: Single message
- **Tool-only responses**: Single message (tool result)
- **Hybrid responses**: Two messages (text first, then tool result)

## Dynamic Prompt System

The platform implements a dynamic prompt system that adapts based on tool usage:

1. **Default System Prompt**:
   - Used for initial message processing
   - Contains descriptions of all available tools
   - Guides tool selection and general interaction

2. **Tool-Specific Prompts**:
   - Stored as "assistant" role messages in MCP servers
   - Named with pattern: `{tool_name}_tool_prompt`
   - Only used after their corresponding tool is called
   - Falls back to default prompt if no tool-specific prompt exists

3. **Per-Message Prompt Flow**:
   ```
   User Message
   ├── Start with default prompt
   ├── LLM decides if tool needed
   ├── If tool called:
   │   ├── Execute tool
   │   ├── Get tool results
   │   └── Use tool's prompt (if exists) for final response
   └── Next message starts fresh with default prompt
   ```

This system ensures:
- Clean separation between tool selection and tool-specific behavior
- Proper fallback to default prompt when needed
- Fresh prompt context for each message
- Compliance with MCP SDK's prompt role requirements

## Clean Implementation

This implementation uses only official MCP SDK patterns:
- ✅ **Official MCPClient**: Uses proper MCP client patterns
- ✅ **Official Transport**: Uses `stdio_client` helpers
- ✅ **Official Types**: Direct use of `types.Tool`, `types.Prompt`, `types.CallToolResult`
- ✅ **No Legacy Code**: Clean implementation without backward compatibility layers
- ✅ **Type Safety**: Full type hints with SDK types

## WebSocket API

```
Connection: ws://localhost:8000/ws/chat

Send: {"action": "chat", "payload": {"text": "message"}, "request_id": "uuid"}
Receive: {"request_id": "uuid", "status": "chunk", "chunk": {"data": "response"}}
```

## Architecture

- **Single UV Environment**: All packages in one `.venv` with editable installs
- **MCP SDK Compliance**: Follows official patterns for client implementation
- **Workspace Structure**: Root manages dependencies, packages are editable
- **Hybrid Response Extraction**: Robust parsing of text + JSON tool calls
