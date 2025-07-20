#!/usr/bin/env python3
"""
Direct test of prompt functionality
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chat_service import ChatService
from config import Configuration
from main import MCPClient

async def test_prompts_directly():
    """Test prompt functionality directly"""
    print("Testing prompt functionality...")
    
    # Load config
    config = Configuration()
    
    # Create MCP client with proper config format
    client_config = {
        "command": "uv",
        "args": ["run", "Servers/demo_server.py"],
        "env": {}
    }
    client = MCPClient("demo", client_config)
    
    # Create chat service
    chat_service = ChatService([client], None, config.get_config_dict())
    
    try:
        # Initialize
        await chat_service.initialize()
        
        # Check what tools are available
        tools = chat_service.tool_schema_manager.get_openai_tools()
        resource_tools = chat_service._get_resource_tools()
        prompt_tools = chat_service._get_prompt_tools()
        
        print(f"Regular tools: {len(tools)}")
        print(f"Resource tools: {len(resource_tools)}")
        print(f"Prompt tools: {len(prompt_tools)}")
        
        print("\nAvailable prompt tools:")
        for tool in prompt_tools:
            func = tool['function']
            print(f"  - {func['name']}: {func['description']}")
            
        # Test prompt registry
        prompt_names = chat_service.tool_schema_manager.list_available_prompts()
        print(f"\nRegistered prompts: {prompt_names}")
        
    finally:
        await chat_service.cleanup()

if __name__ == "__main__":
    asyncio.run(test_prompts_directly())
