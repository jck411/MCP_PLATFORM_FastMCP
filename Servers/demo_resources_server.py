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


@mcp.resource(
    "resource://desktop-files",
    name="DesktopListing",
    description="List of files under ~/Documents/MCP.resources",
    mime_type="text/plain",
)
def desktop() -> str:
    """List the files in the MCP resources directory"""
    desktop = Path("/home/jack/Documents/MCP.resources")
    files = [f.name for f in desktop.iterdir() if f.is_file()]
    return "\n".join(f"- {file}" for file in sorted(files))




if __name__ == "__main__":
    mcp.run()