"""
HTTP Resilience Module for MCP Platform

This module provides HTTP client functionality with comprehensive reliability features:
- Exponential backoff retry logic with jitter
- Connection pooling and keep-alive optimization
- Configurable timeout and retry policies
- Support for both streaming and non-streaming requests
- Provider-agnostic HTTP error handling

This module is completely LLM-agnostic and can be used with any HTTP API.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class HttpConfig(BaseModel):
    """Configuration for HTTP resilience features."""

    # Retry configuration
    max_retries: int = 3
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 16.0
    retry_multiplier: float = 2.0
    retry_jitter: float = 0.1

    # Connection configuration
    timeout: float = 30.0
    max_keepalive_connections: int = 20
    max_connections: int = 100
    keepalive_expiry: float = 30.0


class ResilientHttpClient:
    """
    HTTP client with built-in resilience features.

    Provides exponential backoff retry logic, connection pooling,
    and comprehensive error handling for reliable HTTP communications.
    """

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        config: HttpConfig | None = None,
    ):
        """
        Initialize the resilient HTTP client.

        Args:
            base_url: Base URL for all requests
            headers: Default headers to include with all requests
            config: HTTP configuration settings
        """
        self.base_url = base_url.rstrip("/")
        self.config = config or HttpConfig()

        # Configure HTTP client with reliability features
        self.http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.timeout),
            headers=headers or {},
            # Connection pooling for better performance
            limits=httpx.Limits(
                max_keepalive_connections=self.config.max_keepalive_connections,
                max_connections=self.config.max_connections,
                keepalive_expiry=self.config.keepalive_expiry,
            ),
        )

    async def _exponential_backoff_delay(self, attempt: int) -> None:
        """Calculate and apply exponential backoff delay with jitter."""
        if attempt == 0:
            return

        # Calculate delay: base * multiplier^(attempt-1)
        delay = min(
            self.config.initial_retry_delay * (
                self.config.retry_multiplier ** (attempt - 1)
            ),
            self.config.max_retry_delay
        )

        # Add jitter to prevent thundering herd
        task_hash = hash(asyncio.current_task()) % 100
        jitter = delay * self.config.retry_jitter * (2 * task_hash / 100 - 1)
        delay += jitter

        logger.debug(f"Retrying in {delay:.2f}s (attempt {attempt})")
        await asyncio.sleep(delay)

    def _should_retry(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry."""
        if isinstance(error, httpx.HTTPStatusError):
            # Retry on rate limiting and server errors
            return error.response.status_code in (429, 500, 502, 503, 504)
        # Retry on connection and timeout errors
        return isinstance(error, httpx.ConnectError | httpx.TimeoutException)

    async def request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make HTTP request with exponential backoff retry logic."""
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                await self._exponential_backoff_delay(attempt)

                response = await self.http.request(method, url, **kwargs)
                response.raise_for_status()
                return response

            except Exception as e:
                last_error = e

                if attempt == self.config.max_retries or not self._should_retry(e):
                    break

                logger.warning(
                    f"HTTP request failed (attempt {attempt + 1}/"
                    f"{self.config.max_retries + 1}): {e}"
                )

        # Re-raise the last error
        assert last_error is not None
        raise last_error

    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        stream: bool = False,
    ) -> httpx.Response | AsyncGenerator[dict[str, Any]]:
        """
        Make a POST request with JSON payload.

        Args:
            url: Endpoint URL (relative to base_url)
            payload: JSON payload to send
            stream: Whether to return streaming response

        Returns:
            Either httpx.Response for non-streaming or AsyncGenerator for streaming
        """
        if stream:
            payload["stream"] = True
            return self._stream_json_response(url, payload)

        return await self.request_with_retry("POST", url, json=payload)

    async def _stream_json_response(
        self, url: str, payload: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any]]:
        """Generate streaming JSON responses with retry logic."""
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                await self._exponential_backoff_delay(attempt)

                async with self.http.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                return
                            yield json.loads(data)
                return  # Successful completion

            except Exception as e:
                last_error = e

                if attempt == self.config.max_retries or not self._should_retry(e):
                    break

                logger.warning(
                    f"Streaming request failed (attempt {attempt + 1}/"
                    f"{self.config.max_retries + 1}): {e}"
                )

        # Re-raise the last error
        assert last_error is not None
        raise last_error

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make GET request with retry logic."""
        return await self.request_with_retry("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make POST request with retry logic."""
        return await self.request_with_retry("POST", url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make PUT request with retry logic."""
        return await self.request_with_retry("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make DELETE request with retry logic."""
        return await self.request_with_retry("DELETE", url, **kwargs)

    async def close(self) -> None:
        """Close the HTTP client and clean up resources."""
        await self.http.aclose()

    async def __aenter__(self) -> ResilientHttpClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


def create_http_config_from_dict(config_dict: dict[str, Any]) -> HttpConfig:
    """
    Create HttpConfig from a dictionary (e.g., from YAML config).

    Args:
        config_dict: Dictionary containing HTTP configuration

    Returns:
        HttpConfig instance with validated settings
    """
    # Extract http-specific config, provide defaults
    http_config = config_dict.get("http", {})

    return HttpConfig(
        max_retries=http_config.get("max_retries", 3),
        initial_retry_delay=http_config.get("initial_retry_delay", 1.0),
        max_retry_delay=http_config.get("max_retry_delay", 16.0),
        retry_multiplier=http_config.get("retry_multiplier", 2.0),
        retry_jitter=http_config.get("retry_jitter", 0.1),
        timeout=http_config.get("timeout", 30.0),
        max_keepalive_connections=http_config.get("max_keepalive_connections", 20),
        max_connections=http_config.get("max_connections", 100),
        keepalive_expiry=http_config.get("keepalive_expiry", 30.0),
    )
