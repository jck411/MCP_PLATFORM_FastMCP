# MCP Platform Client

Production-ready Model Context Protocol (MCP) client with full support for **tools**, **prompts**, and **resources**.

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Set API key
export GROQ_API_KEY="your_key_here"

# Run the platform
./run.sh
```

## 📡 WebSocket API

**Connect:** `ws://localhost:8000/ws/chat`

**Send:**
```json
{
  "action": "chat",
  "request_id": "unique-request-id",  // REQUIRED - prevents double-billing
  "payload": {
    "text": "Hello",
    "model": "gpt-4",        // Optional - uses config default if not specified
    "streaming": true        // Optional - overrides config default
  }
}
```

**Receive:**
```json
{
  "request_id": "unique-request-id",
  "status": "chunk",  // "processing" | "chunk" | "complete" | "error"
  "chunk": {
    "type": "text",
    "data": "Response content...",
    "metadata": {}
  }
}
```

## ⚙️ Configuration

### Environment Variables
```bash
# LLM Provider API Keys (choose one)
export GROQ_API_KEY="your_groq_key"          # Recommended for speed
export OPENAI_API_KEY="your_openai_key"      # For GPT models
export ANTHROPIC_API_KEY="your_anthropic_key" # For Claude models

# Optional: Custom configuration paths
export MCP_CONFIG_PATH="/path/to/custom/config.yaml"
export MCP_SERVERS_CONFIG_PATH="/path/to/custom/servers_config.json"
```

### LLM Provider (`src/config.yaml`)
```yaml
llm:
  active: "groq"  # or "openai", "anthropic"
  providers:
    groq:
      api_key: "env:GROQ_API_KEY"
      model: "llama-3.1-70b-versatile"

chat:
  service:
    streaming:
      enabled: true  # REQUIRED: true for streaming, false for complete responses
```

### Add MCP Servers (`src/servers_config.json`)
```json
{
  "mcpServers": {
    "my_server": {
      "enabled": true,
      "command": "uv",
      "args": ["run", "python", "Servers/my_server.py"],
      "cwd": "/absolute/path/to/project"
    }
  },
  "settings": {
    "defaultTimeout": 30,
    "maxRetries": 3,
    "autoReconnect": true
  }
}
```

## 🛠️ Commands

```bash
# Start server
./run.sh
uv run python src/main.py
uv run mcp-platform

# Update dependencies
uv sync --upgrade

# Test server file
uv run python Servers/demo_server.py

# Debug with tokens
DEBUG_TOKENS=1 uv run python src/main.py

# Monitor token usage
cat events.jsonl | jq '.usage'
```

## 🔄 Streaming Modes

### Configuration Required
Set streaming mode in `src/config.yaml`:
```yaml
chat:
  service:
    streaming:
      enabled: true   # true = real-time chunks, false = complete responses
```

### Per-Message Override
```json
{
  "payload": {
    "text": "Your message",
    "streaming": false    // Override config default
  }
}
```

**FAIL FAST**: Platform fails immediately if streaming is not configured and not specified per-message. No silent fallbacks.

## 📁 Key Files

- `src/servers_config.json` - Enable/disable MCP servers
- `src/config.yaml` - LLM provider and streaming settings
- `Servers/` - Your MCP server implementations
- `events.jsonl` - Chat history and token usage

## ✅ Features

- **Full MCP Protocol**: Tools, prompts, resources
- **Multi-Server**: Connect multiple MCP servers simultaneously
- **Real-time**: WebSocket communication with streaming support
- **Token Tracking**: Accurate cost monitoring with tiktoken
- **Type Safety**: Pydantic validation throughout
- **Conflict Resolution**: Automatic handling of name/URI conflicts
- **Schema Conversion**: MCP to OpenAI format for LLM integration
- **Fail Fast**: Strict validation with clear error messages

## 🎯 Common Tasks

```bash
# Change LLM provider (edit src/config.yaml)
llm:
  active: "openai"  # or "groq", "anthropic"

# Enable/disable server (edit src/servers_config.json)
"demo": { "enabled": false }

# Debug connection issues
uv run python Servers/demo_server.py
uv sync --upgrade
```

---

**Requirements:** Python 3.13+, `request_id` required in all WebSocket messages, streaming configuration required.

