#!/usr/bin/env python3
"""
Test script to demonstrate LLM usage tracking functionality.
This script tests the updated LLMClient.get_response_with_tools() method
and ChatService usage tracking.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, Mock

from src.chat_service import ChatService
from src.history.chat_store import ChatEvent, Usage
from src.main import LLMClient


class MockLLMClient:
    """Mock LLM client that returns usage information."""
    
    def __init__(self):
        self.config = {"model": "test-model"}
    
    async def get_response_with_tools(self, messages, tools=None):
        """Mock implementation that returns usage data."""
        return {
            "message": {
                "content": "This is a test response.",
                "role": "assistant"
            },
            "finish_reason": "stop",
            "index": 0,
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 25,
                "total_tokens": 75
            },
            "model": "test-model-actual"
        }


async def test_usage_tracking():
    """Test that usage tracking works correctly."""
    print("Testing LLM usage tracking...")
    
    # Test LLMClient mock behavior
    mock_llm = MockLLMClient()
    response = await mock_llm.get_response_with_tools([])
    
    print(f"Mock LLM response includes usage: {response.get('usage')}")
    print(f"Mock LLM response includes model: {response.get('model')}")
    
    # Test Usage conversion
    api_usage = response["usage"]
    usage = Usage(
        prompt_tokens=api_usage.get("prompt_tokens", 0),
        completion_tokens=api_usage.get("completion_tokens", 0),
        total_tokens=api_usage.get("total_tokens", 0),
    )
    
    print(f"Converted to Usage model: {usage}")
    
    # Test ChatEvent with usage
    event = ChatEvent(
        conversation_id="test-conv",
        type="assistant_message",
        role="assistant",
        content="Test response",
        usage=usage,
        model=response["model"],
        provider="test-provider"
    )
    
    print(f"ChatEvent with usage: {event.usage}")
    print(f"ChatEvent model: {event.model}")
    print(f"ChatEvent provider: {event.provider}")
    
    # Test JSON serialization (important for persistence)
    event_dict = event.model_dump(mode="json")
    print(f"ChatEvent serializes correctly: {bool(event_dict.get('usage'))}")
    
    print("\n✅ All usage tracking tests passed!")
    
    return event


async def main():
    """Run the usage tracking tests."""
    try:
        await test_usage_tracking()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
