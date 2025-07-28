from __future__ import annotations

from typing import Any

from src.history.chat_store import ChatEvent, ChatRepository


class ConversationPersistenceService:
    """
    Provider-agnostic persistence and idempotency utilities for chat conversations.

    Responsibilities:
    - Persist user messages with idempotency by request_id.
    - Return whether to continue processing after checking for cached assistant replies.
    - Fetch existing assistant responses for a given (conversation_id, request_id).
    - Persist final assistant messages.
    - Provide a cached-text accessor that is UI/transport-neutral.

    This module MUST remain LLM/provider agnostic.
    """

    def __init__(self, repo: ChatRepository) -> None:
        self.repo = repo

    async def get_existing_assistant_response(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None:
        """
        Return an existing assistant ChatEvent for this request_id, if it exists.
        """
        existing_assistant_id = await self.repo.get_last_assistant_reply_id(
            conversation_id, request_id
        )
        if not existing_assistant_id:
            return None

        events = await self.repo.get_events(conversation_id)
        for event in events:
            if event.id == existing_assistant_id:
                return event
        return None

    async def ensure_user_message_persisted(
        self, conversation_id: str, user_msg: str, request_id: str
    ) -> bool:
        """
        Persist the user message and return True if the caller should continue
        processing (i.e., no cached assistant response exists yet).

        If the same (conversation_id, request_id) was already recorded and an
        assistant reply exists, return False to prevent duplicate work/billing.
        """
        user_ev = ChatEvent(
            conversation_id=conversation_id,
            type="user_message",
            role="user",
            content=user_msg,
            extra={"request_id": request_id},
        )
        was_added = await self.repo.add_event(user_ev)

        if not was_added:
            existing_response = await self.get_existing_assistant_response(
                conversation_id, request_id
            )
            if existing_response:
                return False
        return True

    async def persist_final_assistant_message(
        self, conversation_id: str, content: str, request_id: str
    ) -> None:
        """
        Persist the final assistant message if there's non-empty content.
        """
        if not content or not content.strip():
            return

        assistant_ev = ChatEvent(
            conversation_id=conversation_id,
            type="assistant_message",
            role="assistant",
            content=content,
            extra={"user_request_id": request_id},
        )
        await self.repo.add_event(assistant_ev)

    @staticmethod
    def get_cached_text(existing_response: ChatEvent) -> str:
        """
        Convert a cached ChatEvent into plain text for the caller to display.
        Keeps this module UI/transport-neutral (no dependency on ChatMessage).
        """
        content: Any = existing_response.content
        return content if isinstance(content, str) else ""
