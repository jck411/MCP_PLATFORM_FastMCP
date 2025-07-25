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
  "request_id": "unique-id",
  "payload": {"text": "Hello"}
}
```

## ⚙️ Configuration

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
  }
}
```

### LLM Provider (`src/config.yaml`)
```yaml
llm:
  active: "groq"  # or "openai", "anthropic"
```

### Environment Variables
```bash
export GROQ_API_KEY="your_groq_key"
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
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
```

## 📁 Key Files

- `src/servers_config.json` - Enable/disable MCP servers
- `src/config.yaml` - LLM provider settings
- `Servers/` - Your MCP server implementations
- `events.jsonl` - Chat history and token usage

## ✅ Features

- **Full MCP Protocol**: Tools, prompts, resources
- **Multi-Server**: Connect multiple MCP servers
- **Real-time**: WebSocket communication
- **Token Tracking**: Accurate cost monitoring
- **Type Safety**: Pydantic validation throughout

---

**Requirements:** Python 3.13+, `request_id` required in all WebSocket messages.

