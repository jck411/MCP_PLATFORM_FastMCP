#!/usr/bin/env python3
"""
Test script to verify HTTP configuration loading in OpenAIOrchestrator
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import Config
from src.openai_orchestrator import OpenAIOrchestrator
from src.history.chat_store import ChatRepository


async def test_http_config():
    """Test that HTTP configuration is loaded correctly."""
    
    # Load configuration
    config = Config()
    await config.initialize()
    
    # Create a repository (mock for testing)
    repo = ChatRepository()
    
    # Create orchestrator
    try:
        orchestrator = OpenAIOrchestrator(
            clients=[],  # Empty clients for testing
            llm_config=config.llm,
            config=config.data,
            repo=repo,
        )
        
        # Print the HTTP configuration values
        print("HTTP Configuration Test Results:")
        print(f"  max_retries: {orchestrator.max_retries}")
        print(f"  initial_retry_delay: {orchestrator.initial_retry_delay}")
        print(f"  max_retry_delay: {orchestrator.max_retry_delay}")
        print(f"  retry_multiplier: {orchestrator.retry_multiplier}")
        print(f"  retry_jitter: {orchestrator.retry_jitter}")
        print(f"  base_url: {orchestrator.base_url}")
        print(f"  model: {orchestrator.model}")
        
        # Test HTTP client limits
        limits = orchestrator.http._limits
        print(f"  max_keepalive_connections: {limits.max_keepalive_connections}")
        print(f"  max_connections: {limits.max_connections}")
        print(f"  keepalive_expiry: {limits.keepalive_expiry}")
        
        print("\n✅ HTTP configuration loaded successfully!")
        
        # Test retry logic validation
        import httpx
        error = httpx.HTTPStatusError("Test 429", request=None, response=type('MockResponse', (), {'status_code': 429})())
        should_retry = orchestrator._should_retry(error)
        print(f"  Should retry 429 error: {should_retry}")
        
        error = httpx.ConnectError("Connection failed")
        should_retry = orchestrator._should_retry(error)
        print(f"  Should retry connection error: {should_retry}")
        
        error = ValueError("Some other error")
        should_retry = orchestrator._should_retry(error)
        print(f"  Should retry ValueError: {should_retry}")
        
        await orchestrator.cleanup()
        
    except Exception as e:
        print(f"❌ Error creating orchestrator: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_http_config())
    sys.exit(0 if success else 1)
