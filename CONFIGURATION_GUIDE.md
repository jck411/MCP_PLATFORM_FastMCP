# MCP Platform Configuration Guide

## 🎯 Configuration Overview

The platform uses a simple hierarchy for configuration:

1. **Environment Variables** - API keys and secrets
2. **`servers_config.json`** - Server management (enable/disable servers)
3. **`config.yaml`** - System settings (LLM provider, prompts)
4. **MCP Server Files** - Tool and feature configuration

## 📝 Quick Setup

### 1. Environment Variables
```bash
export LLM_API_KEY=your_key
export GROQ_API_KEY=your_groq_key
```

### 2. Server Management (`src/servers_config.json`)
```json
{
  "mcpServers": {
    "demo": {
      "enabled": true,
      "command": "uv",
      "args": ["run", "python", "Servers/demo_server.py"],
      "cwd": "/path/to/project"
    }
  }
}
```

### 3. System Settings (`src/config.yaml`)
```yaml
llm:
  active: "openai"
chat:
  service:
    system_prompt: "You are a helpful assistant"
```

## �️ Common Tasks

### Enable/Disable Server
```json
// In servers_config.json
"demo": { "enabled": false }
```

### Add New Server
```json
// In servers_config.json
"my_server": {
  "enabled": true,
  "command": "uv",
  "args": ["run", "python", "Servers/my_server.py"],
  "cwd": "/path/to/project"
}
```

### Change LLM Provider
```yaml
# In config.yaml
llm:
  active: "groq"  # or openai, anthropic, etc.
```

### Configure Server Tools
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

## ✅ Best Practices

- **Server Management**: Use `servers_config.json` to enable/disable servers
- **Tool Configuration**: Configure tool availability directly in each server file with config dictionaries
- **Secrets**: Keep API keys in environment variables only
- **Version Control**: Track configuration files and server tool configs
