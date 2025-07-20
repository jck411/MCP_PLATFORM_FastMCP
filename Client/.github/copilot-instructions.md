<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# MCP Chatbot Project Instructions

This is an MCP (Model Context Protocol) chatbot project that connects to various MCP servers and uses an LLM API for responses.

## Project Structure
- `main.py`: Main chatbot application with MCP client implementation
- `servers_config.json`: Configuration for MCP servers
- `.env`: Environment variables (API keys)
- `pyproject.toml`: UV project configuration

## Key Components
- **Configuration**: Manages environment variables and server configs
- **Server**: Handles MCP server connections and tool execution
- **Tool**: Represents available tools from MCP servers
- **LLMClient**: Communicates with Groq API for LLM responses
- **ChatSession**: Orchestrates the conversation flow

## Development Notes
- Use `uv` for dependency management
- The chatbot supports tool execution via JSON responses from the LLM
- Requires API keys for Groq LLM and optional Brave Search
- Uses asyncio for handling multiple server connections

You can find more info and examples at https://modelcontextprotocol.io/llms-full.txt
