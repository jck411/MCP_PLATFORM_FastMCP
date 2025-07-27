

> **Goal:** Extract the *Conversation Persistence & Idempotency* logic from `OpenAIOrchestrator` into a provider‑agnostic module.
> **Do not** import or reference any provider/LLM types, HTTP clients, or streaming deltas in the new module.
>
> **Create** a new file at `src/conversation/persistence.py` with the exact contents below.
> **Then** apply the edits to `OpenAIOrchestrator` exactly as in the diff below.
> **Finally**, add the unit test `tests/test_persistence.py` as provided.
>
> #### New file: `src/conversation/persistence.py`
>
> ```python
> from __future__ import annotations
>
> from typing import Any, Optional
>
> from src.history.chat_store import ChatEvent, ChatRepository
>
>
> class ConversationPersistenceService:
>     """
>     Provider-agnostic persistence and idempotency utilities for chat conversations.
>
>     Responsibilities:
>     - Persist user messages with idempotency by request_id.
>     - Return whether to continue processing after checking for cached assistant replies.
>     - Fetch existing assistant responses for a given (conversation_id, request_id).
>     - Persist final assistant messages.
>     - Provide a cached-text accessor that is UI/transport-neutral.
>
>     This module MUST remain LLM/provider agnostic.
>     """
>
>     def __init__(self, repo: ChatRepository) -> None:
>         self.repo = repo
>
>     async def get_existing_assistant_response(
>         self, conversation_id: str, request_id: str
>     ) -> Optional[ChatEvent]:
>         """
>         Return an existing assistant ChatEvent for this request_id, if it exists.
>         """
>         existing_assistant_id = await self.repo.get_last_assistant_reply_id(
>             conversation_id, request_id
>         )
>         if not existing_assistant_id:
>             return None
>
>         events = await self.repo.get_events(conversation_id)
>         for event in events:
>             if event.id == existing_assistant_id:
>                 return event
>         return None
>
>     async def ensure_user_message_persisted(
>         self, conversation_id: str, user_msg: str, request_id: str
>     ) -> bool:
>         """
>         Persist the user message and return True if the caller should continue
>         processing (i.e., no cached assistant response exists yet).
>
>         If the same (conversation_id, request_id) was already recorded and an
>         assistant reply exists, return False to prevent duplicate work/billing.
>         """
>         user_ev = ChatEvent(
>             conversation_id=conversation_id,
>             type="user_message",
>             role="user",
>             content=user_msg,
>             extra={"request_id": request_id},
>         )
>         user_ev.compute_and_cache_tokens()
>         was_added = await self.repo.add_event(user_ev)
>
>         if not was_added:
>             existing_response = await self.get_existing_assistant_response(
>                 conversation_id, request_id
>             )
>             if existing_response:
>                 return False
>         return True
>
>     async def persist_final_assistant_message(
>         self, conversation_id: str, content: str, request_id: str
>     ) -> None:
>         """
>         Persist the final assistant message if there's non-empty content.
>         """
>         if not content or not content.strip():
>             return
>
>         assistant_ev = ChatEvent(
>             conversation_id=conversation_id,
>             type="assistant_message",
>             role="assistant",
>             content=content,
>             extra={"user_request_id": request_id},
>         )
>         assistant_ev.compute_and_cache_tokens()
>         await self.repo.add_event(assistant_ev)
>
>     @staticmethod
>     def get_cached_text(existing_response: ChatEvent) -> str:
>         """
>         Convert a cached ChatEvent into plain text for the caller to display.
>         Keeps this module UI/transport-neutral (no dependency on ChatMessage).
>         """
>         content: Any = existing_response.content
>         return content if isinstance(content, str) else ""
> ```
>
> #### Edit `OpenAIOrchestrator` to use the new service
>
> 1. **Add import** at the top:
>
> ```python
> from src.conversation.persistence import ConversationPersistenceService
> ```
>
> 2. **Add a field** in `__init__` after `self.repo = repo`:
>
> ```python
> self.persist = ConversationPersistenceService(repo)
> ```
>
> 3. **Replace** occurrences of the old methods with the service:
>
> * In `process_message(...)`:
>
>   * Replace:
>
>     ```python
>     existing_response = await self._get_existing_assistant_response(conversation_id, request_id)
>     ```
>
>     with:
>
>     ```python
>     existing_response = await self.persist.get_existing_assistant_response(conversation_id, request_id)
>     ```
>   * Replace the cached branch:
>
>     ```python
>     if existing_response:
>         logger.info(f"Returning cached response for request_id: {request_id}")
>         async for msg in self._handle_cached_response(existing_response):
>             yield msg
>         return
>     ```
>
>     with:
>
>     ```python
>     if existing_response:
>         logger.info(f"Returning cached response for request_id: {request_id}")
>         cached_text = self.persist.get_cached_text(existing_response)
>         if cached_text:
>             yield ChatMessage("text", cached_text, {"type": "cached"})
>         return
>     ```
>   * Replace the user-persist step:
>
>     ```python
>     should_continue = await self._ensure_user_message_persisted(conversation_id, user_msg, request_id)
>     ```
>
>     with:
>
>     ```python
>     should_continue = await self.persist.ensure_user_message_persisted(conversation_id, user_msg, request_id)
>     ```
>   * Replace the second cached check similarly (same pattern as above).
>   * Replace final persist:
>
>     ```python
>     await self._persist_final_assistant_message(conversation_id, full_assistant_content, request_id)
>     ```
>
>     with:
>
>     ```python
>     await self.persist.persist_final_assistant_message(conversation_id, full_assistant_content, request_id)
>     ```
> * **Delete** the now-unused methods from `OpenAIOrchestrator`:
>
>   * `_handle_cached_response`
>   * `_ensure_user_message_persisted`
>   * `_persist_final_assistant_message`
>   * `_get_existing_assistant_response`
> * In `chat_once(...)`:
>
>   * Replace both calls to `_get_existing_assistant_response` with `self.persist.get_existing_assistant_response`.
>   * Replace the “persist user message FIRST” block with:
>
>     ```python
>     should_continue = await self.persist.ensure_user_message_persisted(conversation_id, user_msg, request_id)
>     if not should_continue:
>         existing_response = await self.persist.get_existing_assistant_response(conversation_id, request_id)
>         if existing_response:
>             logger.info(f"Returning existing response for duplicate request_id: {request_id}")
>             return existing_response
>     ```
>   * Replace the final assistant persist with:
>
>     ```python
>     await self.persist.persist_final_assistant_message(conversation_id, assistant_full_text, request_id)
>     ```
>
>     and also *remove* the manual construction of `assistant_ev` in `chat_once` (that construction is now centralized in the service).
>
> **Minimal diff (illustrative):**
>
> ```diff
> --- a/src/openai_orchestrator.py
> +++ b/src/openai_orchestrator.py
> @@
> -from src.history.chat_store import ChatEvent, ChatRepository, ToolCall, Usage
> +from src.history.chat_store import ChatEvent, ChatRepository, ToolCall, Usage
> +from src.conversation.persistence import ConversationPersistenceService
> @@ class OpenAIOrchestrator:
>      def __init__(...):
>          self.repo = repo
> +        self.persist = ConversationPersistenceService(repo)
> @@ async def process_message(...):
> -    existing_response = await self._get_existing_assistant_response(conversation_id, request_id)
> +    existing_response = await self.persist.get_existing_assistant_response(conversation_id, request_id)
>      if existing_response:
>          logger.info(f"Returning cached response for request_id: {request_id}")
> -        async for msg in self._handle_cached_response(existing_response):
> -            yield msg
> +        cached_text = self.persist.get_cached_text(existing_response)
> +        if cached_text:
> +            yield ChatMessage("text", cached_text, {"type": "cached"})
>          return
> @@
> -should_continue = await self._ensure_user_message_persisted(conversation_id, user_msg, request_id)
> +should_continue = await self.persist.ensure_user_message_persisted(conversation_id, user_msg, request_id)
> @@
> -await self._persist_final_assistant_message(conversation_id, full_assistant_content, request_id)
> +await self.persist.persist_final_assistant_message(conversation_id, full_assistant_content, request_id)
> @@ async def chat_once(...):
> -existing_response = await self._get_existing_assistant_response(conversation_id, request_id)
> +existing_response = await self.persist.get_existing_assistant_response(conversation_id, request_id)
> @@
> -user_ev = ChatEvent(...); ... = await self.repo.add_event(user_ev) ...  # remove block
> +should_continue = await self.persist.ensure_user_message_persisted(conversation_id, user_msg, request_id)
> +if not should_continue:
> +    existing_response = await self.persist.get_existing_assistant_response(conversation_id, request_id)
> +    if existing_response:
> +        logger.info("Returning existing response for duplicate request_id: %s", request_id)
> +        return existing_response
> @@
> -assistant_ev = ChatEvent(...); await self.repo.add_event(assistant_ev)
> +await self.persist.persist_final_assistant_message(conversation_id, assistant_full_text, request_id)
> @@
> -# Remove these methods from the class:
> -_handle_cached_response
> -_ensure_user_message_persisted
> -_persist_final_assistant_message
> -_get_existing_assistant_response
> +# (methods removed)
> ```
>
> #### New test: `tests/test_persistence.py`
>
> ```python
> import asyncio
> import pytest
> from typing import Any, Dict, List, Optional
>
> from src.conversation.persistence import ConversationPersistenceService
> from src.history.chat_store import ChatEvent
>
>
> class FakeRepo:
>     def __init__(self):
>         self.events: Dict[str, List[ChatEvent]] = {}
>         self.assistant_idx: Dict[tuple[str, str], Optional[str]] = {}
>
>     async def add_event(self, ev: ChatEvent) -> bool:
>         # Idempotency: if an identical user_message exists with same request_id, return False
>         conv = self.events.setdefault(ev.conversation_id, [])
>         if ev.type == "user_message" and ev.extra and "request_id" in ev.extra:
>             for e in conv:
>                 if (
>                     e.type == "user_message"
>                     and e.extra
>                     and e.extra.get("request_id") == ev.extra["request_id"]
>                 ):
>                     return False
>         conv.append(ev)
>         # Track assistant reply id for (conversation_id, request_id)
>         if ev.type == "assistant_message" and ev.extra and "user_request_id" in ev.extra:
>             key = (ev.conversation_id, ev.extra["user_request_id"])
>             self.assistant_idx[key] = ev.id
>         return True
>
>     async def get_last_assistant_reply_id(self, conversation_id: str, request_id: str) -> Optional[str]:
>         return self.assistant_idx.get((conversation_id, request_id))
>
>     async def get_events(self, conversation_id: str) -> List[ChatEvent]:
>         return self.events.get(conversation_id, [])
>
>
> @pytest.mark.asyncio
> async def test_idempotent_user_persist_and_cached_response_flow():
>     repo = FakeRepo()
>     svc = ConversationPersistenceService(repo)  # type: ignore[arg-type]
>     cid = "c1"
>     rid = "r1"
>
>     # First persist of user message -> should continue
>     cont = await svc.ensure_user_message_persisted(cid, "hi", rid)
>     assert cont is True
>
>     # No assistant yet
>     assert await svc.get_existing_assistant_response(cid, rid) is None
>
>     # Finalize assistant message
>     await svc.persist_final_assistant_message(cid, "hello", rid)
>     existing = await svc.get_existing_assistant_response(cid, rid)
>     assert existing is not None
>     assert svc.get_cached_text(existing) == "hello"
>
>     # Second attempt with SAME request_id -> should NOT continue (cached)
>     cont2 = await svc.ensure_user_message_persisted(cid, "hi again", rid)
>     assert cont2 is False
> ```
>
> #### Acceptance criteria
>
> * `src/conversation/persistence.py` contains **no** imports from provider/LLM code, HTTP clients, or streaming types.
> * `OpenAIOrchestrator` compiles and uses `ConversationPersistenceService` for:
>   idempotent user persist, cached fetch, and final assistant persist.
> * All removed methods are fully replaced by the service.
> * The unit test passes (`pytest -q`), validating idempotency and cached retrieval behavior.
> * No behavior changes to provider calls/streaming; only persistence paths were refactored.

---
