# OpenAI Responses API Integration

This MCP Platform now supports **OpenAI's Responses API** for reasoning models like `o3`, `o1-preview`, and `o4-mini`. The Responses API is specifically designed for advanced reasoning models that can think step-by-step through complex problems.

## ✅ What's Working

- **✅ Full Integration**: Responses API integrated into the MCP Platform
- **✅ Streaming Support**: Real-time token streaming from reasoning models
- **✅ Model Support**: o3, o1-preview, o4-mini, and other reasoning models
- **✅ Configuration**: Easy model switching via `config.yaml`
- **✅ Error Handling**: Robust error handling and retry logic
- **✅ Production Ready**: HTTP resilience and proper async handling

## 🚧 Current Limitations

- **🚧 No MCP Tools**: Tools are temporarily disabled (requires HTTP MCP servers)
- **🚧 No Function Calling**: Reasoning models work best without tools anyway
- **🚧 No Temperature Control**: Reasoning models don't support temperature

## 🚀 Quick Start

### 1. Set Your API Key
```bash
# Add to .env file
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Configure Model
In `src/config.yaml`, set:
```yaml
llm:
  active: "openai_responses"  # Enable Responses API

  providers:
    openai_responses:
      model: "o3"  # or o1-preview, o4-mini, etc.
      max_tokens: 2048
```

### 3. Run the Platform
```bash
uv run src/main.py
```

### 4. Connect via WebSocket
The platform runs on `ws://localhost:8000/ws/chat` by default.

## 🎯 Best Use Cases

The Responses API excels at:

- **🧠 Complex Reasoning**: Multi-step logical problems
- **🔬 Analysis Tasks**: Data analysis, research synthesis  
- **📝 Writing**: Long-form content with structured thinking
- **🎓 Education**: Detailed explanations and tutoring
- **🔍 Problem Solving**: Breaking down complex problems

## 📋 Supported Models

| Model | Description | Best For |
|-------|-------------|----------|
| `o3` | Latest reasoning model | Complex analysis, research |
| `o1-preview` | Advanced reasoning | Multi-step problems |
| `o4-mini` | Efficient reasoning | Faster responses |

## 🔧 Configuration Options

### Model Settings
```yaml
openai_responses:
  model: "o3"
  max_tokens: 2048    # Max output tokens
  # Note: temperature not supported by reasoning models
```

### WebSocket Settings
```yaml
chat:
  websocket:
    host: "localhost"
    port: 8000
    endpoint: "/ws/chat"
```

## 🔄 HTTP MCP Integration (Future)

Currently, MCP tools are disabled because the Responses API requires **HTTP MCP servers**, while our platform uses **stdio MCP servers**.

### Why HTTP Required?

```
┌─────────────┐     HTTP     ┌─────────────┐
│ OpenAI API  │ ←────────→  │ MCP Server  │
│ (remote)    │             │ (HTTP)      │
└─────────────┘             └─────────────┘
```

The OpenAI API runs on remote servers and needs to connect to your MCP server over HTTP.

### 🛠️ Future HTTP MCP Integration

To enable MCP tools with the Responses API, you'll need:

#### Option 1: Dual Transport Servers
```python
# Run MCP server in both modes
if __name__ == "__main__":
    if "--http" in sys.argv:
        mcp.run(transport="streamable-http")  # For OpenAI
    else:
        mcp.run()  # stdio for our platform
```

#### Option 2: HTTP-Only Architecture
```python
# Convert platform to use HTTP MCP servers
mcp_client = Client("http://localhost:8000/mcp/")
```

#### Option 3: FastMCP Proxy
```python
# Bridge stdio to HTTP
proxy = FastMCP.as_proxy(
    ProxyClient("demo_server.py"),
    name="HTTP Bridge"
)
proxy.run(transport="http", port=8001)
```

### Example HTTP MCP Server Setup

1. **Create HTTP version of your MCP server:**
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo")

@mcp.tool()
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression"""
    return eval(expression)  # Use safely in production!

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8001)
```

2. **Update orchestrator to use HTTP MCP:**
```python
# In openai_responses_orchestrator.py
if tools:
    mcp_tool = {
        "type": "mcp",
        "server_url": "http://localhost:8001/mcp/",
        "server_label": "demo", 
        "allowed_tools": [t["name"] for t in tools],
        "require_approval": "never"
    }
    payload["tools"] = [mcp_tool]
```

3. **Start both servers:**
```bash
# Terminal 1: HTTP MCP Server
python Servers/demo_server.py --http

# Terminal 2: Main Platform  
uv run src/main.py
```

## 🔍 Debugging

### Enable Debug Logging
The orchestrator includes debug logging to inspect payloads:

```python
logger.info(f"DEBUG: Sending payload: {json.dumps(payload, indent=2)}")
```

### Common Issues

1. **400 Bad Request**: Check for unsupported parameters (temperature, tools)
2. **Port Conflicts**: Ensure HTTP MCP server and WebSocket server use different ports
3. **API Key**: Verify `OPENAI_API_KEY` is set correctly

## 📚 Architecture

```
┌─────────────┐    WebSocket    ┌──────────────┐
│   Client    │ ←─────────────→ │   Platform   │
└─────────────┘                 └──────────────┘
                                        │
                                   HTTP POST
                                        ▼
┌─────────────┐                ┌──────────────┐
│   OpenAI    │ ←─────────────  │  Responses   │
│ Responses   │    (future)     │ Orchestrator │
│     API     │ ←─────────────→ │              │
└─────────────┘    MCP Tools    └──────────────┘
                                        │
                                    stdio
                                        ▼
                                ┌──────────────┐
                                │ MCP Servers  │
                                │   (stdio)    │
                                └──────────────┘
```

## 📈 Performance Notes

- **Reasoning Models**: Generate more tokens than regular models
- **Streaming**: Essential for user experience with long responses  
- **Token Usage**: Monitor usage as reasoning models can be token-heavy
- **Caching**: Consider implementing response caching for repeated queries

## 🎉 Success Story

The integration successfully supports:
- ✅ Real-time streaming responses
- ✅ Complex reasoning queries (quantum physics explanations, etc.)
- ✅ Robust error handling and retries
- ✅ Clean WebSocket interface
- ✅ Production-ready architecture

The platform is now ready for advanced reasoning tasks with OpenAI's most capable models!
