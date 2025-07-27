import pytest

from src.history.chat_store import ChatEvent
from src.history.persistence import ConversationPersistenceService


class FakeRepo:
    def __init__(self):
        self.events: dict[str, list[ChatEvent]] = {}
        self.assistant_idx: dict[tuple[str, str], str | None] = {}

    async def add_event(self, ev: ChatEvent) -> bool:
        # Idempotency: if an identical user_message exists with same request_id,
        # return False to indicate the message was already present
        conv = self.events.setdefault(ev.conversation_id, [])
        if ev.type == "user_message" and ev.extra and "request_id" in ev.extra:
            for e in conv:
                if (
                    e.type == "user_message"
                    and e.extra
                    and e.extra.get("request_id") == ev.extra["request_id"]
                ):
                    return False
        conv.append(ev)
        # Track assistant reply id for (conversation_id, request_id)
        if (
            ev.type == "assistant_message"
            and ev.extra
            and "user_request_id" in ev.extra
        ):
            key = (ev.conversation_id, ev.extra["user_request_id"])
            self.assistant_idx[key] = ev.id
        return True

    async def get_last_assistant_reply_id(
        self, conversation_id: str, request_id: str
    ) -> str | None:
        return self.assistant_idx.get((conversation_id, request_id))

    async def get_events(self, conversation_id: str) -> list[ChatEvent]:
        return self.events.get(conversation_id, [])


@pytest.mark.asyncio
async def test_idempotent_user_persist_and_cached_response_flow():
    repo = FakeRepo()
    svc = ConversationPersistenceService(repo)  # type: ignore[arg-type]
    cid = "c1"
    rid = "r1"

    # First persist of user message -> should continue
    cont = await svc.ensure_user_message_persisted(cid, "hi", rid)
    assert cont is True

    # No assistant yet
    assert await svc.get_existing_assistant_response(cid, rid) is None

    # Finalize assistant message
    await svc.persist_final_assistant_message(cid, "hello", rid)
    existing = await svc.get_existing_assistant_response(cid, rid)
    assert existing is not None
    assert svc.get_cached_text(existing) == "hello"

    # Second attempt with SAME request_id -> should NOT continue (cached)
    cont2 = await svc.ensure_user_message_persisted(cid, "hi again", rid)
    assert cont2 is False
