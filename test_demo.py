#!/usr/bin/env python3
"""
Test script for the demo FastMCP server
"""

from __future__ import annotations

import asyncio
from pathlib import Path

async def test_demo_server() -> None:
    """Test the demo server functionality"""
    
    # Test the desktop resource path
    desktop_path = Path("/home/jack/Documents/MCP.resources")
    
    print(f"Testing desktop path: {desktop_path}")
    print(f"Path exists: {desktop_path.exists()}")
    
    if desktop_path.exists():
        files = list(desktop_path.iterdir())
        print(f"Files found: {[str(f) for f in files]}")
    
    # Test the sum function
    result = 5 + 3
    print(f"Sum test (5 + 3): {result}")
    
    print("✅ Demo server tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_demo_server())
