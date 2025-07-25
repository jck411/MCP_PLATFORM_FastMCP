"""
FastMCP Desktop Example

A simple example that exposes the desktop directory as a resource,
demonstrates tools, and includes example prompts.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Tool configuration - easily toggle tools on/off
TOOL_CONFIG = {
    "math_tools": True,
    "advanced_math": True,
    "conversation_prompts": False,
    "desktop_resources": False,
}

# Create server
mcp = FastMCP("Demo")


# File resources
if TOOL_CONFIG["desktop_resources"]:
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


# Basic math tools
if TOOL_CONFIG["math_tools"]:
    @mcp.tool()
    def sum(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    @mcp.tool()
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b

    @mcp.tool()
    def subtract(a: int, b: int) -> int:
        """Subtract second number from first"""
        return a - b


# Advanced math tools (disabled by default)
if TOOL_CONFIG["advanced_math"]:
    @mcp.tool()
    def power(a: int, b: int) -> int:
        """Raise first number to the power of second"""
        return a ** b

    @mcp.tool()
    def divide(a: float, b: float) -> float:
        """Divide first number by second"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b


# Conversation prompts
if TOOL_CONFIG["conversation_prompts"]:
    @mcp.prompt()
    def summarize_conversation() -> str:
        """Create a summary of the current conversation"""
        return (
            "Please provide a concise summary of our conversation so far, "
            "highlighting the main topics and any important conclusions."
        )

    @mcp.prompt()
    def generate_questions() -> str:
        """Generate follow-up questions about the conversation"""
        return (
            "Based on our conversation, please generate 3-5 thoughtful "
            "follow-up questions that could help continue or deepen "
            "the discussion."
        )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Log which features are enabled
    enabled_features = [k for k, v in TOOL_CONFIG.items() if v]
    logging.info(f"Demo Server starting with features: {', '.join(enabled_features)}")
    mcp.run()
