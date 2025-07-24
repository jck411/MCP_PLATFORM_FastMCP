from mcp.server.fastmcp import FastMCP

# Create server instance with only required name
mcp = FastMCP("Demo Prompt Server")

# Define a minimal prompt with only required fields
@mcp.prompt()
def minimal_demo() -> str:
    """A demonstration of a minimal MCP prompt with only required fields."""
    # Return a simple string (converts to user message)
    return "This is a minimal demo prompt response!"

# Define another minimal prompt
@mcp.prompt()
def greeting_prompt() -> str:
    """A simple greeting prompt for demonstration."""
    return "Hello! This is a greeting from the demo prompt server."


if __name__ == "__main__":
    mcp.run()
