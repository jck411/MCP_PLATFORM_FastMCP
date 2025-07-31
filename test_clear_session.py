#!/usr/bin/env python3
"""
Test script to verify clear_session functionality.
This script will be deleted after testing.
"""

import asyncio
import json
import websockets
import uuid

async def test_clear_session():
    """Test the clear_session WebSocket action."""
    uri = "ws://localhost:8000/ws/chat"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket server")
            
            # Test 1: Send a clear_session message
            request_id = str(uuid.uuid4())
            clear_message = {
                "action": "clear_session",
                "request_id": request_id
            }
            
            print(f"Sending clear_session message: {clear_message}")
            await websocket.send(json.dumps(clear_message))
            
            # Wait for response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"Received response: {response_data}")
            
            # Verify response structure
            if response_data.get("request_id") == request_id:
                if response_data.get("status") == "complete":
                    print("✅ Clear session test PASSED")
                    chunk = response_data.get("chunk", {})
                    if chunk.get("type") == "session_cleared":
                        print("✅ Session cleared successfully")
                        new_conv_id = chunk.get("metadata", {}).get("new_conversation_id")
                        if new_conv_id:
                            print(f"✅ New conversation ID: {new_conv_id}")
                        else:
                            print("❌ Missing new conversation ID")
                    else:
                        print(f"❌ Unexpected chunk type: {chunk.get('type')}")
                else:
                    print(f"❌ Unexpected status: {response_data.get('status')}")
            else:
                print(f"❌ Request ID mismatch: expected {request_id}, got {response_data.get('request_id')}")
                
    except websockets.exceptions.ConnectionRefused:
        print("❌ Could not connect to WebSocket server. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(test_clear_session())
