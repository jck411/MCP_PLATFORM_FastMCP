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

# Set up your LLM provider API key
export GROQ_API_KEY="your_groq_api_key_here"
# OR export OPENAI_API_KEY="your_openai_key_here"
```

### **2. Add Your MCP Servers**

Edit `src/servers_config.json`:
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

### **Configuration Hierarchy**

The platform uses a simple hierarchy for configuration:

1. **Environment Variables** - API keys and secrets
2. **`src/servers_config.json`** - Server management (enable/disable servers)
3. **`src/config.yaml`** - System settings (LLM provider, prompts)
4. **MCP Server Files** - Tool and feature configuration

### **Environment Variables**
```bash
# LLM Provider API Keys (choose one)
export GROQ_API_KEY="your_groq_key"          # Recommended for speed
export OPENAI_API_KEY="your_openai_key"      # For GPT models
export ANTHROPIC_API_KEY="your_claude_key"   # For Claude models

# Optional: Custom configuration paths
export MCP_CONFIG_PATH="/path/to/custom/config.yaml"
export MCP_SERVERS_CONFIG_PATH="/path/to/custom/servers_config.json"
```

### **LLM Providers** (`src/config.yaml`)
```yaml
llm:
  active: "groq"  # or "openai", "anthropic"
  
  providers:
    groq:
      api_key: "env:GROQ_API_KEY"
      model: "llama-3.1-70b-versatile"
      base_url: "https://api.groq.com/openai/v1"
    
    openai:
      api_key: "env:OPENAI_API_KEY" 
      model: "gpt-4"
      base_url: "https://api.openai.com/v1"

chat:
  service:
    system_prompt: "You are a helpful assistant with access to MCP tools and resources."
```

### **Server Management** (`src/servers_config.json`)

**Enable/Disable Servers:**
```json
{
  "mcpServers": {
    "demo": {
      "enabled": true,          // ✅ Server enabled
      "command": "uv",
      "args": ["run", "python", "Servers/demo_server.py"],
      "cwd": "/absolute/path/to/project"
    },
    "experimental": {
      "enabled": false,         // ❌ Server disabled
      "command": "uv",
      "args": ["run", "python", "Servers/experimental_server.py"],
      "cwd": "/absolute/path/to/project"
    }
  },
  "settings": {
    "defaultTimeout": 30,
    "maxRetries": 3,
    "retryDelay": 1.0,
    "logLevel": "INFO",
    "autoReconnect": true
  }
}
```

**Add New Server:**
```json
{
  "mcpServers": {
    "my_custom_server": {
      "enabled": true,
      "command": "uv",
      "args": ["run", "python", "Servers/my_custom_server.py"],
      "cwd": "/absolute/path/to/your/project"
    }
  }
}
```

### **Server Tool Configuration**

Configure tools within each server file:
```python
# In Servers/my_server.py

# Tool configuration at the top of the file
TOOL_CONFIG = {
    "basic_tool": True,
    "advanced_tool": False,
    "experimental_tool": True,
}

# Conditional tool registration based on config
if TOOL_CONFIG["basic_tool"]:
    @mcp.tool()
    def basic_tool():
        """This tool is enabled by config"""
        pass

if TOOL_CONFIG["advanced_tool"]:
    @mcp.tool() 
    def advanced_tool():
        """This tool is disabled by config"""
        pass

if TOOL_CONFIG["experimental_tool"]:
    @mcp.tool()
    def experimental_tool():
        """This tool is enabled by config"""
        pass
```

## 📡 WebSocket API

**Connect:** `ws://localhost:8000/ws/chat`

### **Required Message Format**
```json
{
  "action": "chat",
  "request_id": "unique-request-id",  // REQUIRED - prevents double-billing
  "payload": {
    "text": "List all available tools and resources",
    "model": "gpt-4"  // Optional - uses default from config if not specified
  }
}
```

### **Response Format**
```json
{
  "request_id": "unique-request-id", 
  "status": "chunk",
  "chunk": {
    "type": "text",
    "data": "Here are the available capabilities...",
    "metadata": {}
  }
}
```

### **Status Types**
- `"processing"` - Request received, processing started
- `"chunk"` - Partial response data
- `"complete"` - Request fully processed
- `"error"` - Error occurred during processing

## 🎯 Common Tasks

### **Change LLM Provider**
```yaml
# In src/config.yaml
llm:
  active: "groq"  # Switch to Groq
  # active: "openai"  # Or switch to OpenAI
  # active: "anthropic"  # Or switch to Anthropic
```

### **Enable/Disable Server**
```json
// In src/servers_config.json
"demo": { 
  "enabled": false  // Disable server
}
```

### **Debug Connection Issues**
```bash
# Check server logs
uv run python src/main.py

# Test specific server
uv run python Servers/demo_server.py

# Verify dependencies
uv sync --upgrade
```

### **Monitor Token Usage**
```bash
# Run with token debugging
DEBUG_TOKENS=1 uv run python src/main.py

# Check conversation history
cat events.jsonl | jq '.usage'
```

## 📚 Technical Details

- **MCP SDK**: Latest stable version with full protocol support
- **Python Version**: 3.13+ (leveraging modern async/await and type features)
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
uv python install 3.13  # or latest stable
```

## 🔢 Token Accounting System

### **Features**
- **Accurate Counting**: Uses tiktoken (OpenAI's tokenizer) for precise token counts
- **Caching**: Avoids re-computation with intelligent content hashing
- **Usage Tracking**: Records prompt/completion tokens for cost monitoring
- **Context Management**: Token-aware conversation trimming and optimization

### **Key Components**
```python
# Automatic token counting for all chat events
event.compute_and_cache_tokens()

# Context windowing with proper token limits  
events = await repo.last_n_tokens(conversation_id, max_tokens=4000)

# Token-aware conversation building
optimized = optimize_conversation_for_tokens(
    events, target_tokens=3000, preserve_recent=5
)
```

### **Benefits**
- **Cost Control**: Accurate billing and usage monitoring
- **Performance**: Smart context management prevents token overflow
- **Debugging**: Clear visibility into token consumption patterns
- **Optimization**: Efficient conversation trimming preserves important context

### **Testing the Token System**
```bash
# Run the token counting demonstration
uv run python -c "from src.history.token_counter import *; print(get_token_cache_stats())"
```

---

**🎉 Your MCP Platform is ready for production!** Connect your FastMCP servers and start building powerful AI applications.
