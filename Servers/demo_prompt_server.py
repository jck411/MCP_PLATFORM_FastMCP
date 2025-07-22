from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base as p

# Create server instance with only required name
mcp = FastMCP(
    name="Demo Prompt Server"
)

# Define a minimal prompt with only required fields
@mcp.prompt(
    name="minimal_demo",
    title="Minimal Demo Prompt",
    description="A demonstration of a minimal MCP prompt with only required fields."
)
def minimal_demo() -> str:
    # Return a simple string (converts to user message)
    return "This is a minimal demo prompt response!"

# Define another minimal prompt to show list[Message] return type
@mcp.prompt(
    name="minimal_demo_messages",
    title="Minimal Demo with Messages",
    description="A demonstration of returning explicit messages in a minimal prompt."
)
def minimal_demo_messages() -> list[p.Message]:
    # Return explicit messages
    return [
        p.SystemMessage("This is a minimal system message."),
        p.AssistantMessage("This is a minimal assistant response!")
    ] 