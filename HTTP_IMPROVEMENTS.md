# HTTP Reliability & Throughput Improvements

## Summary

This implementation adds comprehensive HTTP reliability and throughput improvements to the OpenAI orchestrator as requested:

## 1. HTTP Client Configuration

✅ **One httpx.AsyncClient with default headers**
- Centralized HTTP client configuration in `__init__`
- Default headers include:
  - `Authorization: Bearer {api_key}`
  - `Content-Type: application/json`
  - `User-Agent: MCP-Platform/1.0`

✅ **Configurable timeout**
- Configurable via `llm.http.timeout` in config.yaml (default: 30s)
- Also supports per-provider timeout override

✅ **Connection pooling for better performance**
- `max_keepalive_connections`: 20 (configurable)
- `max_connections`: 100 (configurable)  
- `keepalive_expiry`: 30.0s (configurable)

## 2. Exponential Backoff Retries

✅ **Automatic retries on 429/5xx errors**
- Retries on HTTP status codes: 429, 500, 502, 503, 504
- Also retries on connection/timeout errors

✅ **Exponential backoff with jitter**
- Configurable parameters:
  - `max_retries`: 3 (default)
  - `initial_retry_delay`: 1.0s (default)
  - `max_retry_delay`: 16.0s (default)
  - `retry_multiplier`: 2.0 (default)
  - `retry_jitter`: 0.1 (default)

✅ **Jitter to prevent thundering herd**
- Task-based jitter using hash of asyncio task
- Prevents multiple clients from retrying simultaneously

## 3. Unified HTTP Helpers

✅ **Unified POST helpers to reduce duplication**
- `_unified_post_request()`: Single entry point for both streaming and non-streaming
- `_make_http_request_with_retry()`: Non-streaming requests with retry logic
- `_stream_response_generator()`: Streaming requests with retry logic

✅ **Updated existing methods**
- `_call_openai_api()`: Now uses unified helpers with retry logic
- `_stream_openai_api()`: Now uses unified helpers with retry logic

## 4. Configuration Structure

Added new `llm.http` section in config.yaml:

```yaml
llm:
  http:
    timeout: 30.0                    # Request timeout in seconds
    max_retries: 3                   # Maximum number of retry attempts
    initial_retry_delay: 1.0         # Initial delay for exponential backoff
    max_retry_delay: 16.0            # Maximum delay for exponential backoff
    retry_multiplier: 2.0            # Multiplier for exponential backoff
    retry_jitter: 0.1                # Jitter factor to prevent thundering herd
    max_keepalive_connections: 20    # Maximum keep-alive connections
    max_connections: 100             # Maximum total connections
    keepalive_expiry: 30.0           # Keep-alive connection expiry
```

## 5. Benefits

### Reliability
- **Automatic retries** on transient failures (rate limiting, server errors)
- **Connection error handling** with retry logic
- **Exponential backoff** prevents overwhelming servers during outages

### Throughput  
- **Connection pooling** reduces overhead of establishing new connections
- **Keep-alive connections** reduce latency for subsequent requests
- **Configurable limits** allow tuning for specific workloads

### Maintainability
- **Unified HTTP helpers** eliminate code duplication
- **Centralized configuration** makes tuning easy
- **Consistent retry behavior** across streaming and non-streaming requests

## 6. Implementation Details

### Error Handling
- `_should_retry()`: Determines if an error warrants a retry
- Supports both HTTP status errors and connection errors
- Respects maximum retry attempts to prevent infinite loops

### Retry Logic
- `_exponential_backoff_delay()`: Calculates delay with jitter
- Task-based jitter prevents coordinated retry storms
- Respects maximum delay to prevent excessive wait times

### Code Structure
- All HTTP configuration loaded from YAML config
- Instance variables instead of global constants for flexibility
- Type hints and comprehensive error handling
- Maintains existing API compatibility

## Testing

✅ Code compiles without errors
✅ Imports work correctly  
✅ Application starts successfully
✅ No lint errors

The implementation is ready for production use and provides robust HTTP reliability with improved throughput characteristics.
