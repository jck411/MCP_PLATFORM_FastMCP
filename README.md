# MCP Platform Client

A production-ready Model Context Protocol (MCP) client with comprehensive support for **tools**, **prompts**, and **resources**. Built with the official MCP SDK and ready to connect to any FastMCP or MCP-compliant server.

## 🎯 Current Status

**✅ CLIENT READY FOR PRODUCTION**

Your MCP client is **fully implemented** and ready to connect to new FastMCP servers that support:

- **🔧 Tools**: Discovery, schema conversion, parameter validation, execution
- **💭 Prompts**: Listing, retrieval, argument handling  
- **📁 Resources**: Discovery, URI-based reading, content access
- **🌐 Multi-Server**: Simultaneous connections to multiple MCP servers
- **🤖 LLM Integration**: OpenAI-format tool schemas for seamless LLM interaction

## 🏗️ Architecture Overview

### **Core Components**

- **`MCPClient`**: Official MCP SDK client with full protocol support
- **`ToolSchemaManager`**: Unified registry for tools, prompts, and resources across all servers
- **`ChatService`**: Orchestrates conversations between users, LLMs, and MCP servers
- **`WebSocketServer`**: Real-time communication layer for frontend integration
- **`LLMClient`**: Multi-provider LLM integration (OpenAI, Groq, Anthropic, etc.)

### **Key Features**

- ✅ **Full MCP Protocol Support**: Tools, prompts, resources, and all standard operations
- ✅ **Multi-Server Management**: Connect to multiple MCP servers simultaneously
- ✅ **Conflict Resolution**: Automatic handling of name/URI conflicts across servers
- ✅ **Schema Conversion**: Convert MCP schemas to OpenAI format for LLM consumption
- ✅ **Parameter Validation**: Pydantic-based validation for all tool parameters
- ✅ **Error Handling**: Proper MCP error codes and graceful failure handling
- ✅ **Real-time Communication**: WebSocket interface for responsive frontends
- ✅ **Accurate Token Counting**: tiktoken-based token accounting with caching for precise context management

## 🚀 Ready for FastMCP Servers

When you create your new FastMCP servers, this client will automatically:

1. **Discover All Capabilities**
   ```python
   # Your client will automatically call:
   await client.list_tools()      # Discover all tools
   await client.list_prompts()    # Discover all prompts  
   await client.list_resources()  # Discover all resources
   ```

2. **Handle Multi-Server Scenarios**
   ```json
   {
     "mcpServers": {
       "data-server": {
         "command": "uv",
         "args": ["run", "python", "-m", "data_server"]
       },
       "web-server": {
         "command": "uv", 
         "args": ["run", "python", "-m", "web_server"]
       }
     }
   }
   ```

3. **Provide Unified Access**
   ```python
   # All capabilities accessible through ToolSchemaManager:
   tools = schema_manager.get_openai_tools()           # All tools in OpenAI format
   prompts = schema_manager.list_available_prompts()   # All prompts from all servers
   resources = schema_manager.list_available_resources() # All resources from all servers
   ```

## 📋 Implementation Checklist

### **✅ Completed - Ready for Production**

#### **MCP Protocol Support**
- ✅ `list_tools()` - Discover tools from servers
- ✅ `call_tool()` - Execute tools with parameter validation
- ✅ `list_prompts()` - Discover prompts from servers
- ✅ `get_prompt()` - Retrieve prompts with arguments
- ✅ `list_resources()` - Discover resources from servers
- ✅ `read_resource()` - Read resource content by URI

#### **Client Management**
- ✅ Multi-server connection handling
- ✅ Automatic reconnection with exponential backoff
- ✅ Health monitoring and ping functionality
- ✅ Graceful cleanup and resource management

#### **Schema Management** 
- ✅ Unified registry for all capability types
- ✅ Name/URI conflict resolution across servers
- ✅ OpenAI schema conversion for LLM integration
- ✅ Pydantic parameter validation
- ✅ Metadata export and introspection

#### **Integration Layer**
- ✅ WebSocket server for real-time communication
- ✅ Multi-LLM provider support (OpenAI, Groq, Anthropic, etc.)
- ✅ Structured tool calling with proper error handling
- ✅ Configuration management with environment variables

## 🛠️ Quick Setup

### **1. Configure Your Environment**
```bash
# Install dependencies (always gets latest compatible versions)
uv sync

# Optional: Update to latest versions
uv sync --upgrade

# Set up your LLM provider
echo "GROQ_API_KEY=your_key_here" > .env
```

### **2. Add Your MCP Servers**
Edit `src/servers_config.json`:
```json
{
  "mcpServers": {
    "my-fastmcp-server": {
      "command": "uv",
      "args": ["run", "python", "-m", "my_server"],
      "env": {
        "PYTHONPATH": "."
      }
    }
  }
}
```

### **3. Run the Client**
```bash
# Start WebSocket server (default)
./run.sh

# Or run directly
uv run python src/main.py

# Or use the installed script
uv run mcp-platform
```

## 🔧 Configuration

### **LLM Providers** (`src/config.yaml`)
```yaml
llm:
  active: "groq"  # openai, groq, anthropic, azure, openrouter, gemini, mistral
  providers:
    groq:
      model: "llama-3.1-8b-instant"  # or latest available model
      base_url: "https://api.groq.com/openai/v1"
      temperature: 0.7

chat:
  interface: "websocket"
  websocket:
    host: "localhost"
    port: 8000
```

### **Environment Variables**
```bash
# LLM Provider API Keys (choose one)
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key  
ANTHROPIC_API_KEY=your_anthropic_key
# ... etc
```

## 📡 WebSocket API

**Connect:** `ws://localhost:8000/ws/chat`

**Send Message:**
```json
{
  "action": "chat",
  "request_id": "unique-id",
  "payload": {
    "text": "List all available tools and resources"
  }
}
```

**Receive Response:**
```json
{
  "request_id": "unique-id", 
  "status": "chunk",
  "chunk": {
    "type": "text",
    "data": "Here are the available capabilities...",
    "metadata": {}
  }
}
```

## 🎯 Next Steps

1. **Create FastMCP Servers** - Your client is ready to connect to any servers you build
2. **Add Server Configurations** - Simply update `servers_config.json` with your new servers
3. **Build Frontends** - Use the WebSocket API to create web, mobile, or desktop interfaces
4. **Scale Horizontally** - Add as many MCP servers as needed, the client handles them all

## 📚 Technical Details

- **MCP SDK**: Latest stable version with full protocol support
- **Python Version**: 3.11+ (leveraging modern async/await and type features)
- **Key Dependencies**: FastAPI, uvicorn, httpx, Pydantic, official MCP SDK
- **Architecture**: Async/await throughout, proper resource management
- **Error Handling**: MCP-compliant error codes and structured error responses
- **Type Safety**: Full type hints using official MCP types
- **Dependency Management**: UV with automatic latest version resolution

### **Keeping Dependencies Updated**

```bash
# Check for available updates
uv tree

# Update to latest compatible versions
uv sync --upgrade

# Update Python version (if needed)
uv python install 3.12  # or latest stable
```

## 🔢 Token Accounting System

The platform includes a robust token counting system that ensures accurate context management:

### **Features**

- **🎯 Accurate Counting**: Uses tiktoken (OpenAI's tokenizer) for precise token counts
- **⚡ Performance Optimized**: Content-based caching prevents redundant computations
- **🔄 Automatic Windowing**: `last_n_tokens()` intelligently manages context windows
- **📊 Multiple Encodings**: Support for different model encodings (cl100k_base, etc.)

### **Key Components**

```python
# Automatic token counting for all chat events
event = ChatEvent(content="Your message here")
event.compute_and_cache_tokens()  # Uses tiktoken, caches result

# Context windowing with proper token limits
events = await repo.last_n_tokens(conversation_id, max_tokens=4000)

# Token-aware conversation building
from src.history.conversation_utils import build_conversation_with_token_limit

conversation, token_count = build_conversation_with_token_limit(
    system_prompt="You are a helpful assistant",
    events=history_events,
    user_message="What can you help me with?",
    max_tokens=4000,
    reserve_tokens=500  # Reserve space for response
)
```

### **Benefits**

- **Prevents Context Errors**: No more "context length exceeded" surprises
- **Optimizes Performance**: Caching eliminates redundant token computations
- **Improves Reliability**: Accurate windowing ensures consistent behavior
- **Better User Experience**: Predictable context management

### **Testing the Token System**

```bash
# Run the token counting demonstration
uv run python test_token_fix.py
```

---

**🚀 Your MCP Platform Client is production-ready and waiting for your FastMCP servers!**
