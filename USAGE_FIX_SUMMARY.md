# LLM Usage/Costs Tracking Fix Summary

## Problem
LLMClient.get_response_with_tools() was throwing away usage information, making it impossible to:
- Track billing costs
- Monitor token usage 
- Implement proper context trimming based on actual usage

## Solution Implemented

### 1. Updated LLMClient.get_response_with_tools() in `src/main.py`

**Before:**
```python
return {
    "message": choice["message"],
    "finish_reason": choice.get("finish_reason"),
    "index": choice.get("index", 0),
}
```

**After:**
```python
return {
    "message": choice["message"],
    "finish_reason": choice.get("finish_reason"),
    "index": choice.get("index", 0),
    "usage": result.get("usage"),
    "model": result.get("model", self.config["model"]),
}
```

### 2. Enhanced ChatService in `src/chat_service.py`

**Added usage conversion helper:**
```python
def _convert_usage(self, api_usage: dict[str, Any] | None) -> Usage:
    """Convert LLM API usage to our Usage model."""
    if not api_usage:
        return Usage()
    
    return Usage(
        prompt_tokens=api_usage.get("prompt_tokens", 0),
        completion_tokens=api_usage.get("completion_tokens", 0),
        total_tokens=api_usage.get("total_tokens", 0),
    )
```

**Updated _generate_assistant_response() signature:**
```python
async def _generate_assistant_response(
    self, conv: list[dict[str, Any]]
) -> tuple[str, Usage, str]:  # Now returns (content, usage, model)
```

**Added usage tracking in the method:**
- Tracks usage from initial LLM API call
- Accumulates usage from tool call follow-up API calls
- Returns total accumulated usage

**Updated chat_once() to persist usage:**
```python
assistant_ev = ChatEvent(
    conversation_id=conversation_id,
    type="assistant_message",
    role="assistant",
    content=assistant_full_text,
    usage=total_usage,           # ✅ Now includes usage data
    model=model,                 # ✅ Now includes actual model
    provider="openai",
    extra=assistant_extra,
)
```

### 3. Data Flow

1. **LLM API Call** → Returns usage data in response
2. **ChatService** → Converts and accumulates usage across multiple API calls
3. **ChatEvent** → Stores usage data for persistence
4. **JsonlRepo** → Persists usage data to events.jsonl

### 4. Benefits

- **Billing Tracking**: Each ChatEvent now contains complete usage information
- **Cost Monitoring**: Can calculate costs by model and provider
- **Context Management**: Can make informed decisions about context trimming
- **Performance Analytics**: Track token efficiency across conversations
- **Multi-call Accuracy**: Handles tool calls that require multiple LLM API calls

### 5. Backward Compatibility

- Existing ChatEvent records without usage data continue to work
- Usage field is optional (Usage | None) 
- New events automatically include usage when available

## Testing

The implementation was verified with a comprehensive test that confirmed:
- Usage data is properly extracted from LLM API responses
- Usage conversion works correctly
- ChatEvent accepts and serializes usage data
- Multiple API calls accumulate usage correctly

This fix resolves the inability to track LLM costs and enables proper billing, monitoring, and context management.
