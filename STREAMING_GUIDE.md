# Streaming Configuration Guide (FAIL FAST Implementation)

Your MCP Platform supports both streaming and non-streaming modes with **strict fail-fast behavior** - no backwards compatibility fallbacks.

## Two Response Modes

### 1. **Non-Streaming Mode**
- Uses `chat_once()` method in `ChatService`
- Returns complete response after all processing is done
- Better for: batch operations, API integrations, simple use cases
- Usage tracked and persisted to chat history

### 2. **Streaming Mode** 
- Uses `process_message()` method in `ChatService` 
- Returns response chunks in real-time as they're generated
- Better for: interactive chat, real-time user feedback, ChatGPT-like experience
- Strict validation of streaming API responses

## Streaming Configuration Location

**Required setting** in `src/config.yaml`:

```yaml
chat:
  service:
    streaming:
      enabled: true    # REQUIRED: true or false (no default fallback)
```

## How to Control Streaming

### Method 1: Configuration Default
Set the default behavior in config:
```yaml
streaming:
  enabled: true   # All requests use streaming by default
```

### Method 2: Per-Message Override
Override the default per message:
```json
{
  "action": "chat",
  "request_id": "unique_id", 
  "payload": {
    "text": "Your message here",
    "streaming": false    // Override config default
  }
}
```

### Method 3: Explicit Per-Message
Always specify streaming preference:
```json
{
  "action": "chat",
  "request_id": "unique_id",
  "payload": {
    "text": "Your message here", 
    "streaming": true     // Explicit preference
  }
}
```

## FAIL FAST Behavior

### Configuration Validation
❌ **FAILS** if `streaming.enabled` is not set in config AND no explicit `streaming` in payload:
```json
{
  "status": "error",
  "chunk": {
    "error": "Streaming configuration missing. Set 'chat.service.streaming.enabled' in config.yaml or specify 'streaming: true/false' in payload."
  }
}
```

### LLM Client Validation
❌ **FAILS** if streaming is requested but LLM client doesn't support it:
```
RuntimeError: LLM client does not support streaming. Use chat_once() for non-streaming responses.
```

### Streaming API Validation
❌ **FAILS** if streaming API response is invalid:
- Non-200 HTTP status
- Wrong content-type (expects `text/event-stream`)
- Invalid JSON in streaming chunks
- No streaming chunks received

### Example Failure Responses
```json
// Missing configuration
{
  "request_id": "123",
  "status": "error", 
  "chunk": {
    "error": "Streaming configuration missing..."
  }
}

// Invalid streaming response
{
  "request_id": "123",
  "status": "error",
  "chunk": {
    "error": "Expected streaming response, got content-type: application/json"
  }
}
```

## Testing FAIL FAST Behavior

```bash
# Start server
uv run python src/main.py

# Test all modes including error cases
uv run python test_streaming.py
```

## Implementation Details

### No Backwards Compatibility
- ❌ No auto-detection
- ❌ No silent fallbacks  
- ❌ No default assumptions
- ✅ Explicit configuration required
- ✅ Clear error messages
- ✅ Immediate failure on misconfiguration

### Configuration Priority (STRICT)
1. **Client explicit setting** (`"streaming": true/false`) - **Overrides config**
2. **Config default** (`streaming.enabled: true/false`) - **Must be set**
3. **No fallback** - **FAILS if neither is specified**

### Tool Call Handling
Both modes support tool calls with strict validation:
- **Non-streaming**: Tool calls between complete LLM responses
- **Streaming**: Tool calls between streaming sessions with chunk validation

## Performance & Reliability

**FAIL FAST Benefits:**
- ✅ **Predictable behavior** - no silent degradation
- ✅ **Clear error messages** - immediate feedback on misconfigurations  
- ✅ **Robust validation** - catches API/network issues early
- ✅ **Explicit configuration** - no hidden defaults
- ✅ **Production ready** - fails cleanly rather than silently

**Current Status:**
- ✅ **Strict Configuration**: Required `streaming.enabled` setting
- ✅ **LLM Validation**: Ensures streaming support before attempting
- ✅ **API Validation**: Validates streaming response format and content
- ✅ **Error Handling**: Clear error messages with actionable fixes
- ✅ **No Fallbacks**: Fails immediately on any configuration issue

Your platform now implements streaming with **strict fail-fast behavior** - it will tell you exactly what's wrong instead of silently falling back! 🚀
