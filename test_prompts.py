#!/usr/bin/env python3
"""
Test script to verify prompt functionality
"""

import asyncio
import json
import websockets

async def test_prompts():
    """Test the prompt functionality via WebSocket"""
    try:
        uri = "ws://localhost:8080"
        async with websockets.connect(uri) as websocket:
            # Send a test message
            message = {
                "type": "message", 
                "content": "What tools and prompts do you have available? List them out."
            }
            await websocket.send(json.dumps(message))
            
            # Wait for responses
            async for response in websocket:
                try:
                    data = json.loads(response)
                    print(f"Response: {data}")
                    if data.get('type') == 'text' or data.get('type') == 'error':
                        break
                except json.JSONDecodeError:
                    print(f"Raw response: {response}")
                    break
                    
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    asyncio.run(test_prompts())
