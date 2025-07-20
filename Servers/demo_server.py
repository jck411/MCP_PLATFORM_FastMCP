"""
FastMCP Desktop Example

A simple example that exposes the desktop directory as a resource,
demonstrates tools, and includes example prompts.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Create server
mcp = FastMCP("Demo")


@mcp.resource("dir://desktop")
def desktop() -> list[str]:
    """List the files in the user's desktop"""
    desktop = Path("/home/jack/Documents/MCP.resources")
    return [str(f) for f in desktop.iterdir()]


@mcp.tool()
def sum(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def list_resources() -> str:
    """List all available resources and their descriptions"""
    desktop = Path("/home/jack/Documents/MCP.resources")
    files = [str(f) for f in desktop.iterdir()]
    return f"Available resource files: {', '.join(files)}"


@mcp.tool()
def read_file(filename: str) -> str:
    """Read the contents of a file from the MCP resources directory"""
    desktop = Path("/home/jack/Documents/MCP.resources")
    file_path = desktop / filename
    if file_path.exists():
        return file_path.read_text()
    return f"File {filename} not found in resources directory"


@mcp.prompt()
def analyze_file(filename: str) -> str:
    """Analyze a file from the MCP resources directory"""
    return (
        f"Please analyze the file '{filename}' for key insights "
        f"and summarize its content."
    )


@mcp.prompt()
def explain_code(code: str, language: str = "python") -> str:
    """Explain how a piece of code works"""
    return (
        f"Please explain how this {language} code works:\n\n"
        f"```{language}\n{code}\n```"
    )


@mcp.prompt()
def summarize_conversation() -> str:
    """Create a summary of the current conversation"""
    return (
        "Please provide a concise summary of our conversation so far, "
        "highlighting the main topics and any important conclusions."
    )


if __name__ == "__main__":
    mcp.run()
