# Idempotency and Replay Safety Implementation

## Overview

This implementation adds idempotency and replay safety to the MCP Platform FastMCP chat system to prevent double-persistence and double-billing when WebSocket clients retry messages due to network hiccups or other issues.

## Key Changes

### 1. ChatEvent Structure Enhancement
- Added support for `request_id` in the `ChatEvent.extra` dictionary
- Added support for `user_request_id` in assistant responses to track which user message they respond to

### 2. Repository Interface Updates
- Modified `ChatRepository.add_event()` to return `bool` (True if added, False if duplicate)
- Added `get_event_by_request_id()` method to find events by request ID
- Added `get_last_assistant_reply_id()` method to find assistant responses for a user request

### 3. Idempotency Logic in Repositories
Both `InMemoryRepo` and `JsonlRepo` now implement:
- Duplicate detection based on `request_id` in the `extra` field
- Prevention of duplicate event persistence
- Efficient lookup of existing events by request ID

### 4. Chat Service Enhancements
- Modified `chat_once()` to accept optional `request_id` parameter
- Added idempotency checks before processing user messages
- Added caching of assistant responses to prevent double-billing
- Refactored code to reduce complexity and improve maintainability

### 5. WebSocket Server Integration
- Updated to pass `request_id` from client messages to the chat service
- Maintains backward compatibility with clients that don't provide request IDs

## How It Works

### Request Flow with Idempotency
1. **Client sends message** with optional `request_id`
2. **WebSocket server** extracts `request_id` and passes it to chat service
3. **Chat service** checks for existing assistant response:
   - If found, returns cached response (no double-billing)
   - If not found, continues to step 4
4. **User message persistence** with idempotency check:
   - If duplicate `request_id`, skips persistence but continues
   - If new, persists the user message
5. **LLM processing** only if no cached assistant response exists
6. **Assistant response persistence** with `user_request_id` linkage

### Duplicate Detection
- Events with the same `request_id` in the same conversation are considered duplicates
- Only the first occurrence is persisted to prevent double-storage
- Assistant responses are linked to user requests via `user_request_id`

## Usage Examples

### Client Message Format
```json
{
  "action": "chat",
  "request_id": "unique-request-123",
  "payload": {
    "text": "Hello, what's the weather like?",
    "model": "gpt-4"
  }
}
```

### Event Storage
```python
# User message with request tracking
user_event = ChatEvent(
    conversation_id="conv-123",
    type="user_message",
    role="user", 
    content="Hello, what's the weather like?",
    extra={"request_id": "unique-request-123"}
)

# Assistant response with user request linkage
assistant_event = ChatEvent(
    conversation_id="conv-123",
    type="assistant_message",
    role="assistant",
    content="I'd be happy to help with weather information...",
    extra={"user_request_id": "unique-request-123"}
)
```

## Benefits

1. **Prevents Double-Billing**: LLM calls are not repeated for duplicate requests
2. **Prevents Double-Storage**: Duplicate events are not persisted multiple times
3. **Network Resilience**: Clients can safely retry requests without side effects
4. **Backward Compatible**: Works with existing clients that don't provide request IDs
5. **Performance**: Efficient duplicate detection using in-memory lookups

## Implementation Notes

- Request IDs are optional - the system works without them for backward compatibility
- Idempotency is conversation-scoped (same request ID in different conversations is allowed)
- The implementation is thread-safe using appropriate locking mechanisms
- Works with both in-memory and persistent (JSONL) storage backends

## Testing

The implementation includes comprehensive testing that verifies:
- Duplicate request detection and prevention
- Assistant response caching and retrieval
- Event lookup by request ID
- Backward compatibility with non-request-ID clients
