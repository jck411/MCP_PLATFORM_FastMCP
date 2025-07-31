"""
FastMCP Desktop Example

A simple example that exposes the desktop directory as a resource,
demonstrates tools, and includes example prompts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# Tool configuration - easily toggle tools on/off
TOOL_CONFIG = {
    "math_tools": True,
    "advanced_math": True,
    "conversation_prompts": False,
    "desktop_resources": True,
    "openrouter_tools": True,
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


# OpenRouter tools
if TOOL_CONFIG["openrouter_tools"]:
    @mcp.tool()
    async def get_openrouter_models() -> dict[str, Any]:
        """
        Fetch all available models from OpenRouter API.

        Returns a dictionary containing:
        - data: List of all available models with details like id, name,
          description, pricing, etc.
        - total_count: Total number of available models
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("https://openrouter.ai/api/v1/models")
                response.raise_for_status()

                models_data = response.json()

                if "data" in models_data:
                    return {
                        "data": models_data["data"],
                        "total_count": len(models_data["data"]),
                        "status": "success"
                    }

                return {
                    "error": "Unexpected response format from OpenRouter API",
                    "status": "error"
                }

        except httpx.HTTPError as e:
            return {
                "error": f"HTTP error occurred: {e!s}",
                "status": "error"
            }
        except Exception as e:
            return {
                "error": f"An error occurred: {e!s}",
                "status": "error"
            }

    @mcp.tool()
    async def search_openrouter_models(
        query: str,
        limit: int = 10
    ) -> dict[str, Any]:
        """
        Search OpenRouter models by name or description.

        Args:
            query: Search term to look for in model names and descriptions
            limit: Maximum number of results to return (default: 10)

        Returns:
            Dictionary containing matching models and search metadata
        """
        try:
            # First get all models
            models_response = await get_openrouter_models()

            if models_response.get("status") == "error":
                return models_response

            models = models_response["data"]
            query_lower = query.lower()

            # Search in model names and descriptions
            matching_models = []
            for model in models:
                name_match = query_lower in model.get("name", "").lower()
                desc_match = query_lower in model.get("description", "").lower()
                id_match = query_lower in model.get("id", "").lower()

                if name_match or desc_match or id_match:
                    matching_models.append(model)

                if len(matching_models) >= limit:
                    break

            return {
                "data": matching_models,
                "query": query,
                "total_matches": len(matching_models),
                "limit": limit,
                "status": "success"
            }

        except Exception as e:
            return {
                "error": f"Search error: {e!s}",
                "status": "error"
            }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Log which features are enabled
    enabled_features = [k for k, v in TOOL_CONFIG.items() if v]
    logging.info(f"Demo Server starting with features: {', '.join(enabled_features)}")
    mcp.run()
