# MCP Platform - How It Works

## Architecture Overview

The MCP Platform is a flexible chatbot system that connects to MCP (Model Context Protocol) servers and provides both terminal and WebSocket interfaces for interaction. The platform uses a consolidated environment structure for maximum development efficiency and reproducibility.

## Environment Structure

### Consolidated Virtual Environment
The platform uses a single consolidated virtual environment at the project root:

```
/home/jack/REPOS/MCP_PLATFROM/
├── .venv/                    # Single consolidated environment
├── Client/                   # Client package (no separate .venv)
│   ├── src/                  # Source code
│   └── pyproject.toml        # Client package definition
├── Servers/simple-tool/      # Server package
│   ├── mcp_simple_tool/      # Server source code
│   └── pyproject.toml        # Server package definition
├── pyproject.toml            # Root project with all dependencies
└── uv.lock                   # Single lock file
```

### Package Installation
All packages are installed in editable mode in the consolidated environment:

```toml
# Root pyproject.toml
[tool.uv.sources]
mcp-platform-client = { path = "Client", editable = true }
mcp-simple-tool = { path = "Servers/simple-tool", editable = true }
```

This allows:
- **Single environment management**: `uv sync` installs everything
- **Editable development**: Changes reflect immediately without reinstall
- **Consistent dependencies**: All packages use the same versions
- **Simplified imports**: Standard Python module imports work everywhere

### Import Structure
The consolidated environment uses absolute imports from installed packages:

```python
# Client modules (from Client/src/)
from src.main import MCPClient, LLMClient, ChatSession
from src.config import Configuration
from src.websocket_server import run_websocket_server

# Server modules (from Servers/simple-tool/)
import mcp_simple_tool
# Or via command line: python -m mcp_simple_tool
```

## Configuration System

### YAML-Based Configuration
Main application settings are loaded from `config.yaml`:

```yaml
# Chat Interface Configuration
chat:
  interface: "websocket"  # or "terminal"
  websocket:
    host: "localhost"
    port: 8000
    endpoint: "/ws/chat"
    allow_origins: ["*"]
    allow_credentials: true
    max_message_size: 16777216
    ping_interval: 20
    ping_timeout: 10

# LLM Provider Configuration (OpenAI-compatible APIs)
llm:
  active: "groq"  # openai, groq, azure, anthropic, openrouter, etc.
  providers:
    openai:
      base_url: "https://api.openai.com/v1"
      model: "gpt-4o"
      temperature: 0.7
      max_tokens: 4096
      top_p: 1.0

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(levelname)s - %(message)s"

# MCP Server Configuration
mcp:
  config_file: "servers_config.json"
```

### Environment Variables
API keys and secrets are stored in `.env`:
```env
LLM_API_KEY=your_api_key_here
```

## Core Components

### 1. Configuration Class
- Loads YAML configuration and environment variables
- Provides methods to access specific config sections
- Separates configuration (YAML) from secrets (.env)

### 2. MCPClient Class
- Official MCP client implementation following SDK patterns
- Handles MCP server connections using proper transport patterns
- Uses `stdio_client` helpers for transport management
- Maintains async connection lifecycle with proper cleanup

### 3. MCP SDK Types
- Uses official MCP SDK types directly (types.Tool, types.Prompt, etc.)
- No custom wrappers or formatting layers
- Full compatibility with MCP specification

### 4. LLMClient Class
- Provider-agnostic HTTP client using `httpx`
- Supports any OpenAI-compatible API
- Configurable via YAML (base_url, model, parameters)

### 5. ChatSession Class
- Orchestrates user/LLM/MCP client interactions
- Manages conversation history
- Handles tool execution flow

## Interface System

### Terminal Interface
```python
interface = config.chat_interface  # "terminal"
chat_session = ChatSession(clients, llm_client)
await chat_session.start()
```
- Interactive command-line chat
- Direct user input/output
- Type `quit` or `exit` to stop

### WebSocket Interface
```python
interface = config.chat_interface  # "websocket"
await run_websocket_server(clients, llm_client, config.get_config_dict())
```
- WebSocket server on localhost:8000
- RESTful health endpoints (`/health`, `/`)
- Real-time bidirectional communication
- CORS-enabled for web frontend integration

## LLM Provider Support

The system supports any OpenAI-compatible API by configuring the `base_url`:

- **OpenAI**: `https://api.openai.com/v1`
- **Groq**: `https://api.groq.com/openai/v1`
- **Azure OpenAI**: `https://your-resource.openai.azure.com/openai/deployments/your-deployment`
- **OpenRouter**: `https://openrouter.ai/api/v1`
- **Anthropic**: `https://api.anthropic.com/v1`
- **Local/Ollama**: `http://localhost:11434/v1`

## Tool Execution Flow

1. User sends message
2. LLM analyzes available tools and user intent
3. If tool needed, LLM responds with JSON:
   ```json
   {
     "tool": "tool-name",
     "arguments": {"param": "value"}
   }
   ```
4. System executes tool via appropriate MCP client
5. Tool result is sent back to LLM
6. LLM generates natural language response

## Running the Application

### Environment Setup
```bash
# One-time setup
cd /home/jack/REPOS/MCP_PLATFROM
uv sync --all-extras  # Installs all dependencies and packages
```

### Running the Client
```bash
# From the root directory (consolidated environment)
cd /home/jack/REPOS/MCP_PLATFROM
uv run python -m src.main
```

### Running MCP Servers
```bash
# Simple tool server (standalone testing)
uv run python -m mcp_simple_tool --help
```

### Server Configuration
The `servers_config.json` works seamlessly with the consolidated environment:

```json
{
  "mcpServers": {
    "simple-tool": {
      "command": "python",
      "args": ["-m", "mcp_simple_tool"],
      "env": {}
    }
  }
}
```

**Note**: The server configuration uses `python` directly because it runs as a subprocess from the client, which inherits the same environment when started with `uv run`.

Interface is controlled by `chat.interface` in `config.yaml`:
- `"terminal"`: Command-line interface
- `"websocket"`: Web server interface

## Key Features

- ✅ **Official MCP Client**: Uses official SDK patterns with proper transport handling
- ✅ **Clean Architecture**: No legacy code, pure SDK implementation
- ✅ **Consolidated Environment**: Single virtual environment for all components
- ✅ **Editable Packages**: Changes reflect immediately without reinstall
- ✅ **Reproducible Setup**: `uv sync` recreates the exact environment
- ✅ **Provider Agnostic**: Works with any OpenAI-compatible API
- ✅ **Dual Interface**: Terminal and WebSocket modes
- ✅ **YAML Configuration**: Easy to modify settings
- ✅ **Secure Secrets**: API keys in .env file
- ✅ **MCP Integration**: Connects to multiple MCP servers
- ✅ **Tool Execution**: Automatic tool discovery and execution
- ✅ **Async Architecture**: Efficient connection handling
