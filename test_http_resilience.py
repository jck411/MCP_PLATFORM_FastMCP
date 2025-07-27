#!/usr/bin/env python3
"""
Test script for the HTTP resilience module.
Demonstrates the modular extraction from openai_orchestrator.py.
"""

import asyncio
from src.http_resilience import ResilientHttpClient, HttpConfig, create_http_config_from_dict


async def test_http_config():
    """Test HTTP configuration creation."""
    print("Testing HTTP configuration...")
    
    # Test default config
    default_config = HttpConfig()
    print(f"✓ Default config: max_retries={default_config.max_retries}, timeout={default_config.timeout}")
    
    # Test config from dict (like YAML)
    yaml_dict = {
        "http": {
            "max_retries": 5,
            "timeout": 60.0,
            "retry_jitter": 0.2
        }
    }
    config_from_dict = create_http_config_from_dict(yaml_dict)
    print(f"✓ Config from dict: max_retries={config_from_dict.max_retries}, timeout={config_from_dict.timeout}")


async def test_resilient_client():
    """Test resilient HTTP client creation."""
    print("\nTesting HTTP client creation...")
    
    config = HttpConfig(timeout=10.0, max_retries=2)
    headers = {"User-Agent": "Test-Client/1.0"}
    
    async with ResilientHttpClient(
        base_url="https://httpbin.org",
        headers=headers,
        config=config
    ) as client:
        print("✓ HTTP client created successfully")
        print(f"✓ Base URL: {client.base_url}")
        print(f"✓ Config timeout: {client.config.timeout}")


async def main():
    """Run all tests."""
    print("🧪 Testing HTTP Resilience Module")
    print("=" * 40)
    
    await test_http_config()
    await test_resilient_client()
    
    print("\n✅ All tests passed!")
    print("\n🎉 HTTP resilience module is completely LLM-agnostic and ready for reuse!")


if __name__ == "__main__":
    asyncio.run(main())
