import json
import logging

import httpx
import mcp.types as types
from bs4 import BeautifulSoup, Tag
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create server instance
app = Server("simple-tool")

# Example data
EXAMPLE_DATA = {
    "message": "Hello from Simple Tool Server!",
    "status": "active",
    "tools": ["fetch", "calculate"],  # Added calculate
}


@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    """List available prompts"""
    return [
        types.Prompt(
            name="fetch_tool_prompt",  # Changed name to be tool-specific
            title="Fetch Website Tool Prompt",
            description="A specialized prompt for the fetch website tool",
            arguments=[
                types.PromptArgument(
                    name="context",
                    description=(
                        "Optional context about what kind of website content is needed"
                    ),
                    required=False,
                ),
            ],
        )
    ]


@app.get_prompt()
async def get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Get a specific prompt"""
    if name != "fetch_tool_prompt":
        raise ValueError(f"Unknown prompt: {name}")

    context = arguments.get("context", "") if arguments else ""
    context_text = f" Context: {context}" if context else ""

    prompt_text = (
        "You are a specialized website content assistant that helps users fetch and "
        f"analyze web content.{context_text}\n\n"
        "The 'fetch' tool can retrieve content from websites and:\n"
        "- Handles HTML pages by extracting:\n"
        "  * Page title\n"
        "  * Meta description\n"
        "  * First 5 substantial paragraphs of content\n"
        "- Processes JSON endpoints automatically\n"
        "- Provides header and size information for other content types\n\n"
        "Best practices for using the fetch tool:\n"
        "1. Always ensure URLs include the protocol (http:// or https://)\n"
        "2. Be prepared to handle common errors:\n"
        "   - Invalid URLs\n"
        "   - Connection timeouts\n"
        "   - Non-200 status codes\n"
        "3. Consider content type expectations:\n"
        "   - Use HTML URLs for webpage content\n"
        "   - Use API endpoints for JSON data\n\n"
        "Your role is to:\n"
        "1. Validate and format URLs before fetching\n"
        "2. Interpret the fetched content based on its type\n"
        "3. Present the results in a structured, readable format\n"
        "4. Highlight key information from the extracted content\n"
        "5. Suggest follow-up actions based on the content type and user's needs"
    )

    return types.GetPromptResult(
        messages=[
            types.PromptMessage(
                role="assistant",  # Changed from "system" to "assistant"
                content=types.TextContent(type="text", text=prompt_text),
            )
        ],
        description=("A specialized prompt for fetching and analyzing website content"),
    )


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="fetch",
            description="Fetch the content of a website",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"}
                },
                "required": ["url"],
            },
        ),
        types.Tool(
            name="calculate",
            description=(
                "Basic math calculations (+, -, *, /, %, ^). " "Example: '2 + 3 * 4'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "Math expression to evaluate. "
                            "Supports: numbers and basic operators"
                        ),
                    }
                },
                "required": ["expression"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.ContentBlock]:
    """Handle all tool calls"""
    if name == "fetch":
        if "url" not in arguments:
            raise ValueError("Missing required argument: url")

        url = arguments["url"]
        logger.info(f"Fetching URL: {url}")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()

                # Get content type
                content_type = response.headers.get("content-type", "")

                # Return appropriate content based on type
                if "text/html" in content_type:
                    # Parse HTML with BeautifulSoup
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Extract useful information
                    title = soup.title.string if soup.title else "No title found"
                    description = ""
                    meta_desc = soup.find("meta", attrs={"name": "description"})
                    if meta_desc and isinstance(meta_desc, Tag):
                        description = meta_desc.get("content", "")

                    # Get main content (adjust selectors based on the site)
                    main_content = []
                    for p in soup.find_all("p")[:5]:  # First 5 paragraphs
                        text = p.get_text().strip()
                        if text and len(text) > 50:  # Only substantial paragraphs
                            main_content.append(text)

                    # Build structured result
                    result = {
                        "url": url,
                        "title": title,
                        "description": description,
                        "content_preview": main_content,
                        "content_type": "html",
                    }

                    return [
                        types.TextContent(
                            type="text",
                            text=json.dumps(result, indent=2),
                        )
                    ]
                elif "application/json" in content_type:
                    # Parse JSON response
                    json_data = response.json()
                    return [
                        types.TextContent(
                            type="text",
                            text=json.dumps(
                                {"url": url, "content_type": "json", "data": json_data},
                                indent=2,
                            ),
                        )
                    ]
                else:
                    # For other content types, return a summary
                    return [
                        types.TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "url": url,
                                    "content_type": content_type,
                                    "content_length": len(response.content),
                                    "headers": dict(response.headers),
                                },
                                indent=2,
                            ),
                        )
                    ]
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"error": str(e), "url": url}, indent=2),
                )
            ]

    elif name == "calculate":
        if "expression" not in arguments:
            raise ValueError("Missing required argument: expression")

        expression = arguments["expression"].strip()
        logger.info(f"Calculating expression: {expression}")

        try:
            # Replace ^ with ** for Python power operator
            expression = expression.replace("^", "**")

            # Evaluate the expression safely
            allowed_chars = set("0123456789.+-*/%() ")
            expr_without_power = expression.replace("**", "")
            if not all(c in allowed_chars for c in expr_without_power):
                raise ValueError("Expression contains invalid characters")

            result = eval(expression, {"__builtins__": {}}, {})

            # Format result nicely - handle both integers and decimals
            if isinstance(result, int) or result.is_integer():
                formatted_result = str(int(result))
            else:
                formatted_result = f"{result:.6f}".rstrip("0").rstrip(".")

            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "expression": arguments["expression"],
                            "result": formatted_result,
                        },
                        indent=2,
                    ),
                )
            ]
        except Exception as e:
            error_msg = f"Error evaluating expression: {str(e)}"
            logger.error(error_msg)
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": error_msg,
                            "expression": expression,
                        },
                        indent=2,
                    ),
                )
            ]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """Main entry point"""
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Simple Tool Server started")
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
