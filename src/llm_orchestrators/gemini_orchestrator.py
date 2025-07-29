"""
Modified Gemini ADK Orchestrator for MCP Platform

This version fixes duplicate assistant replies by distinguishing between
streaming deltas and final responses. It uses event.partial and
event.is_final_response() from the ADK to decide when to stream text
and when to finalize. Tool calls and results are persisted separately,
while user-facing text is deduplicated.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import google.genai.types as genai_types
from google.adk.agents import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.events import Event
from google.adk.models import Gemini
from google.adk.runners import Runner

try:
    from google.adk.agents.run_config import RunConfig, StreamingMode
except ImportError:
    RunConfig = None
    StreamingMode = None
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_session_manager import (
    SseConnectionParams,
    StdioConnectionParams,
)
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from mcp import StdioServerParameters
from mcp import types as mcp_types

from src.history.chat_store import ChatEvent, ChatRepository, ToolCall, Usage
from src.history.persistence import ConversationPersistenceService
from src.mcp_services.prompting import MCPResourcePromptService
from src.schema_manager import SchemaManager

if TYPE_CHECKING:
    from src.main import MCPClient

logger = logging.getLogger(__name__)

MAX_TOOL_HOPS = 8
MAX_TEXT_DELTA_SIZE = 10_000
MAX_LOG_MESSAGE_LENGTH = 100


class InitializationError(Exception):
    """Raised when initialization of the Gemini ADK orchestrator fails."""

    pass


@dataclass(slots=True)
class ChatMessage:
    """Represents a chat message with metadata."""
    mtype: Literal["text", "error"]
    content: str
    meta: dict[str, Any]

    def __init__(
        self,
        mtype: Literal["text", "error"],
        content: str,
        meta: dict[str, Any] | None = None,
    ):
        self.mtype = mtype
        self.content = content
        self.meta = meta or {}

    @property
    def type(self) -> str:  # backwards compatibility
        return self.mtype


class GeminiAdkOrchestrator:
    """
    Conversation orchestrator using Google ADK + Gemini with native MCP tools.

    It distinguishes between partial (streaming) and final responses using
    event.partial and event.is_final_response(). Only new final content is
    sent to the user, preventing duplicate replies.
    """

    def __init__(
        self,
        clients: list[MCPClient],
        llm_config: dict[str, Any],
        config: dict[str, Any],
        repo: ChatRepository,
        max_history_messages: int = 50,
    ) -> None:
        self.clients = clients
        self.llm_config = llm_config
        self.config = config
        self.repo = repo
        self.persist = ConversationPersistenceService(repo)
        self.max_history_messages = max_history_messages
        self.chat_conf = config.get("chat", {}).get("service", {}) or {}

        self.tool_mgr: SchemaManager | None = None
        self._init_lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self.mcp_prompt: MCPResourcePromptService | None = None
        self._system_prompt: str = ""

        # Provider config
        self.active_name = (llm_config.get("active") or "").lower()
        providers = llm_config.get("providers") or {}
        self.provider_cfg = providers.get(self.active_name, {}) or {}

        # API key
        self.api_key = self._extract_api_key()

        # ADK members
        self._runner: Runner | None = None
        self._agent: LlmAgent | None = None
        self._session_service: InMemorySessionService | None = None
        self._artifact_service: InMemoryArtifactService | None = None
        self._mcp_toolsets: list[MCPToolset] = []

        # Session mapping for bootstrap pattern fallback
        self._session_id_mapping: dict[str, str] = {}

    # ---------------------------------------------------------------------
    # Initialization helpers
    # ---------------------------------------------------------------------

    def _extract_api_key(self) -> str:
        """Extract API key from env or provider config."""
        env_order = [
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            (self.llm_config.get("active") or "").upper() + "_API_KEY",
        ]
        for k in env_order:
            v = os.getenv(k)
            if v:
                return v
        return self.provider_cfg.get("api_key", "") or ""

    @property
    def model(self) -> str:
        """Return the configured Gemini model name."""
        return self.provider_cfg.get("model", "gemini-1.5-pro")

    async def initialize(self) -> None:
        """Initialize MCP clients, build prompts, and configure ADK."""
        async with self._init_lock:
            if self._ready.is_set():
                return

            logger.info("GeminiAdkOrchestrator.initialize() starting...")
            # Connect to MCP clients and build prompt/resource catalog
            try:
                connection_results = await asyncio.gather(
                    *(c.connect() for c in self.clients), return_exceptions=True
                )
                connected_clients: list[MCPClient] = []
                for i, result in enumerate(connection_results):
                    if isinstance(result, Exception):
                        logger.warning(
                            "Client '%s' failed to connect: %s",
                            self.clients[i].name,
                            result,
                        )
                    else:
                        connected_clients.append(self.clients[i])

                self.tool_mgr = SchemaManager(connected_clients)
                await self.tool_mgr.initialize()
                self.mcp_prompt = MCPResourcePromptService(
                    self.tool_mgr, self.chat_conf
                )
                await self.mcp_prompt.update_resource_catalog_on_availability()
                self._system_prompt = await self.mcp_prompt.make_system_prompt()

                logger.info(
                    "GeminiAdkOrchestrator: %d MCP tools (prompt-side), %d resources.",
                    len(self.tool_mgr.get_mcp_tools()),
                    len(
                        self.mcp_prompt.resource_catalog if self.mcp_prompt else []
                    ),
                )
            except Exception as e:
                logger.warning(
                    "SchemaManager/MCP prompt init failed (non-fatal): %s", e
                )

            # Build ADK MCP toolsets
            try:
                mcp_config = self._load_mcp_servers_config()
                self._mcp_toolsets = self._build_adk_mcp_toolsets_from_config(
                    mcp_config
                )
                if not self._mcp_toolsets:
                    logger.warning(
                        "No ADK MCP toolsets configured. "
                        "Proceeding without external tools."
                    )
                else:
                    logger.info(
                        "Configured %d ADK MCP toolsets", len(self._mcp_toolsets)
                    )
            except Exception as e:
                logger.error("Failed to configure ADK MCP toolsets: %s", e)
                raise InitializationError(f"ADK MCP setup failed: {e}") from e

            # Create ADK Agent (Gemini + MCP tools)
            try:
                llm = Gemini(model=self.model)
                tools = list(self._mcp_toolsets)
                self._agent = LlmAgent(
                    name="gemini_mcp_agent",
                    model=llm,
                    instruction=self._system_prompt or "",
                    tools=tools,  # type: ignore[arg-type]
                )
            except Exception as e:
                logger.error("Failed to create ADK LlmAgent: %s", e)
                raise InitializationError(f"ADK agent creation failed: {e}") from e

            # Create ADK Runner and in-memory services
            try:
                self._session_service = InMemorySessionService()
                self._artifact_service = InMemoryArtifactService()
                self._runner = Runner(
                    agent=self._agent,
                    session_service=self._session_service,
                    artifact_service=self._artifact_service,
                    app_name="mcp-platform",
                )
            except Exception as e:
                logger.error("Failed to create ADK Runner: %s", e)
                raise InitializationError(f"ADK runner creation failed: {e}") from e

            self._ready.set()

    def _build_adk_mcp_toolsets_from_config(
        self, mcp_cfg: dict[str, Any]
    ) -> list[MCPToolset]:
        """Build ADK MCPToolset objects from configuration."""
        servers = mcp_cfg.get("servers") or []
        toolsets: list[MCPToolset] = []
        for s in servers:
            name = s.get("name")
            kind = (s.get("kind") or "stdio").lower()
            if not name:
                raise InitializationError("MCP server config missing 'name'")
            if kind == "stdio":
                cmd = s.get("command")
                args = s.get("args") or []
                env = s.get("env") or {}
                if not cmd:
                    raise InitializationError(
                        f"MCP stdio server '{name}' missing 'command'"
                    )
                server_params = StdioServerParameters(command=cmd, args=args, env=env)
                params = StdioConnectionParams(server_params=server_params)
                toolsets.append(MCPToolset(connection_params=params))
            elif kind == "sse":
                url = s.get("url")
                headers = s.get("headers") or {}
                if not url:
                    raise InitializationError(f"MCP SSE server '{name}' missing 'url'")
                params = SseConnectionParams(url=url, headers=headers)
                toolsets.append(MCPToolset(connection_params=params))
            else:
                raise InitializationError(
                    f"Unknown MCP server kind '{kind}' for '{name}'"
                )
        return toolsets

    def _load_mcp_servers_config(self) -> dict[str, Any]:
        """Load MCP servers config from servers_config.json and convert to ADK."""
        try:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "servers_config.json"
            )
            config_path = os.path.abspath(config_path)
            if not os.path.exists(config_path):
                logger.warning(f"MCP servers config not found at {config_path}")
                return {"servers": []}
            with open(config_path) as f:
                config_data = json.load(f)
            mcp_servers = config_data.get("mcpServers", {})
            adk_servers: list[dict[str, Any]] = []
            for server_name, server_config in mcp_servers.items():
                if not server_config.get("enabled", False):
                    continue
                command = server_config.get("command")
                args = server_config.get("args", [])
                cwd = server_config.get("cwd")
                if not command:
                    logger.warning(
                        f"MCP server '{server_name}' missing command, skipping"
                    )
                    continue
                adk_server: dict[str, Any] = {
                    "name": server_name,
                    "kind": "stdio",
                    "command": command,
                    "args": args,
                }
                if cwd:
                    adk_server["env"] = {"PYTHONPATH": cwd}
                adk_servers.append(adk_server)
                logger.info(f"Converted MCP server '{server_name}' for ADK")
            return {"servers": adk_servers}
        except Exception as e:
            logger.error(f"Failed to load MCP servers config: {e}")
            return {"servers": []}

    # ---------------------------------------------------------------------
    # ADK session management
    # ---------------------------------------------------------------------

    async def _get_or_create_adk_session(self, conversation_id: str) -> str:
        """Get or create an ADK session for a conversation."""
        assert self._runner is not None and self._session_service is not None
        user_id = "default"
        if conversation_id in self._session_id_mapping:
            return self._session_id_mapping[conversation_id]
        try:
            await self._ensure_adk_session_with_id(user_id, conversation_id)
            return conversation_id
        except Exception as e:
            logger.warning(
                "Direct session creation failed: %s. Will try bootstrap pattern.", e
            )
        return "bootstrap"

    async def _ensure_adk_session_with_id(self, user_id: str, session_id: str) -> None:
        """Ensure an ADK session exists with a specific session_id."""
        if not self._session_service:
            raise InitializationError("ADK session service not initialized")
        try:
            try:
                await self._session_service.create_session(
                    app_name="mcp-platform", user_id=user_id, session_id=session_id
                )
            except TypeError:
                await self._session_service.create_session(  # type: ignore[call-arg]
                    user_id=user_id, session_id=session_id
                )
            return
        except Exception:
            try:
                await self._session_service.get_session(
                    app_name="mcp-platform", user_id=user_id, session_id=session_id
                )
                return
            except Exception:
                try:
                    await self._session_service.get_session(  # type: ignore[call-arg]
                        user_id=user_id, session_id=session_id
                    )
                    return
                except Exception as e:
                    logger.error(
                        "Failed to ensure ADK session %s/%s: %s",
                        user_id,
                        session_id,
                        e,
                    )
                    raise InitializationError(f"ADK session setup failed: {e}") from e

    # ---------------------------------------------------------------------
    # History
    # ---------------------------------------------------------------------

    async def _build_conversation_history(
        self, conversation_id: str
    ) -> list[dict[str, Any]]:
        """Build canonical conversation history for analytics/logging."""
        events = await self.repo.get_events(conversation_id, self.max_history_messages)
        conv: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt}
        ]
        for ev in events:
            if ev.type in ("user_message", "assistant_message", "system_update"):
                conv.append({"role": ev.role, "content": ev.content})
        return conv

    # ---------------------------------------------------------------------
    # Event processing helpers
    # ---------------------------------------------------------------------

    async def _handle_event_processing(
        self,
        event: Event,
        conversation_id: str,
        request_id: str,
        tool_hops: int,
    ) -> tuple[int, str | None]:
        """
        Persist tool calls and results from the event and update tool_hops.

        Returns a tuple of (updated_tool_hops, warning_message). The warning
        message is non-None only if the maximum tool hop limit is exceeded.
        """
        tool_calls = self._extract_tool_calls(event)
        if tool_calls:
            await self._persist_tool_calls(conversation_id, request_id, tool_calls)
            tool_hops += 1
            if tool_hops > MAX_TOOL_HOPS:
                warn = f"⚠️ Reached maximum tool call limit ({MAX_TOOL_HOPS})."
                return tool_hops, warn
        tool_results = self._extract_tool_results(event)
        if tool_results:
            await self._persist_tool_results(conversation_id, request_id, tool_results)
        return tool_hops, None

    # ---------------------------------------------------------------------
    # Streaming processing (bootstrap)
    # ---------------------------------------------------------------------

    async def _process_events_bootstrap(  # noqa: PLR0912
        self,
        user_id: str,
        conversation_id: str,
        request_id: str,
        new_message: genai_types.Content,
    ) -> AsyncGenerator[ChatMessage]:
        """
        Process events using the bootstrap pattern (ADK chooses a session_id).

        This streaming handler distinguishes between partial deltas and final
        responses using event.partial and event.is_final_response(). It
        yields ChatMessage objects to the caller with meta types 'delta' or
        'final', ensuring that duplicate final messages are not sent to the user.
        """
        tool_hops = 0
        partial_buffer: str = ""
        first_event = True
        async for event in self._runner.run_async(  # type: ignore[call-arg]
            user_id=user_id,
            new_message=new_message,
        ):
            # capture the real session_id on the first event
            if first_event and hasattr(event, "session_id"):
                real_session_id = getattr(event, "session_id", None)
                if real_session_id:
                    self._session_id_mapping[conversation_id] = real_session_id
                    logger.info(
                        "Mapped conversation %s to ADK session %s",
                        conversation_id,
                        real_session_id,
                    )
                first_event = False
            # Persist tool calls/results and update tool hops
            tool_hops, warn = await self._handle_event_processing(
                event, conversation_id, request_id, tool_hops
            )
            if warn:
                yield ChatMessage("text", warn)
                break
            # Handle partial text deltas
            is_partial = False
            try:
                is_partial = bool(getattr(event, "partial", False))
            except Exception:
                is_partial = False
            if is_partial:
                delta = self._safe_text_from_event(event)
                if delta:
                    if len(delta) > MAX_TEXT_DELTA_SIZE:
                        delta = delta[:MAX_TEXT_DELTA_SIZE]
                    yield ChatMessage("text", delta, {"type": "delta"})
                    partial_buffer += delta
            # Handle final responses
            is_final = False
            try:
                is_final = bool(event.is_final_response())
            except Exception:
                is_final = False
            if is_final:
                final_current_text = self._safe_text_from_event(event) or ""
                if not partial_buffer:
                    if final_current_text:
                        if len(final_current_text) > MAX_TEXT_DELTA_SIZE:
                            final_current_text = final_current_text[
                                :MAX_TEXT_DELTA_SIZE
                            ]
                        yield ChatMessage("text", final_current_text, {"type": "final"})
                elif (
                    final_current_text
                    and final_current_text.strip() != partial_buffer.strip()
                ):
                    if len(final_current_text) > MAX_TEXT_DELTA_SIZE:
                        final_current_text = final_current_text[
                            :MAX_TEXT_DELTA_SIZE
                        ]
                    yield ChatMessage("text", final_current_text, {"type": "final"})
                partial_buffer = ""

    # ---------------------------------------------------------------------
    # Streaming processing (direct)
    # ---------------------------------------------------------------------

    async def _process_events_direct(  # noqa: PLR0912
        self,
        user_id: str,
        adk_session_id: str,
        conversation_id: str,
        request_id: str,
        new_message: genai_types.Content,
    ) -> AsyncGenerator[ChatMessage]:
        """
        Process events using the direct session pattern (predetermined session_id).

        Like `_process_events_bootstrap`, this method streams delta events and
        final responses with deduplication. It respects the ADK streaming
        configuration if available.
        """
        tool_hops = 0
        partial_buffer: str = ""
        # Configure streaming if possible
        run_config = None
        if RunConfig and StreamingMode:
            try:
                run_config = RunConfig(
                    streaming_mode=StreamingMode.SSE,
                    max_llm_calls=10,
                )
                logger.info("Configured ADK streaming with SSE mode")
            except Exception as e:
                logger.warning(f"Could not configure ADK streaming: {e}")
        # Obtain event stream
        if not self._runner:
            raise InitializationError("ADK runner is not initialized")
        if run_config:
            event_stream = self._runner.run_async(
                user_id=user_id,
                session_id=adk_session_id,
                new_message=new_message,
                run_config=run_config,
            )
        else:
            event_stream = self._runner.run_async(
                user_id=user_id,
                session_id=adk_session_id,
                new_message=new_message,
            )
        async for event in event_stream:
            # Persist tool calls/results and update tool hops
            tool_hops, warn = await self._handle_event_processing(
                event, conversation_id, request_id, tool_hops
            )
            if warn:
                yield ChatMessage("text", warn)
                break
            # Handle partial text deltas
            is_partial = False
            try:
                is_partial = bool(getattr(event, "partial", False))
            except Exception:
                is_partial = False
            if is_partial:
                delta = self._safe_text_from_event(event)
                if delta:
                    if len(delta) > MAX_TEXT_DELTA_SIZE:
                        delta = delta[:MAX_TEXT_DELTA_SIZE]
                    yield ChatMessage("text", delta, {"type": "delta"})
                    partial_buffer += delta
            # Handle final responses
            is_final = False
            try:
                is_final = bool(event.is_final_response())
            except Exception:
                is_final = False
            if is_final:
                final_current_text = self._safe_text_from_event(event) or ""
                if not partial_buffer:
                    if final_current_text:
                        if len(final_current_text) > MAX_TEXT_DELTA_SIZE:
                            final_current_text = final_current_text[
                                :MAX_TEXT_DELTA_SIZE
                            ]
                        yield ChatMessage("text", final_current_text, {"type": "final"})
                elif (
                    final_current_text
                    and final_current_text.strip() != partial_buffer.strip()
                ):
                    if len(final_current_text) > MAX_TEXT_DELTA_SIZE:
                        final_current_text = final_current_text[
                            :MAX_TEXT_DELTA_SIZE
                        ]
                    yield ChatMessage("text", final_current_text, {"type": "final"})
                partial_buffer = ""

    # ---------------------------------------------------------------------
    # Public API: process_message (streaming)
    # ---------------------------------------------------------------------

    async def process_message(  # noqa: PLR0912
        self,
        conversation_id: str,
        user_msg: str,
        request_id: str,
    ) -> AsyncGenerator[ChatMessage]:
        """
        Streaming response handler.

        1. Persist the user message.
        2. Send the message to the ADK runner.
        3. Stream ADK events, yielding text deltas or final responses to the UI.
        4. Persist the final assistant message once streaming is complete.
        """
        await self._ready.wait()
        if not self._runner or not self._agent:
            raise InitializationError("ADK runner/agent not initialized")

        logger.info(
            (
                "GeminiAdkOrchestrator.process_message called: "
                "conversation_id=%s, msg='%s'"
            ),
            conversation_id,
            (
                user_msg[:MAX_LOG_MESSAGE_LENGTH] + "..."
                if len(user_msg) > MAX_LOG_MESSAGE_LENGTH
                else user_msg
            ),
        )

        # Persist user message idempotently
        should_continue = await self.persist.ensure_user_message_persisted(
            conversation_id, user_msg, request_id
        )
        if not should_continue:
            existing = await self.persist.get_existing_assistant_response(
                conversation_id, request_id
            )
            if existing and (txt := self.persist.get_cached_text(existing)):
                yield ChatMessage("text", txt, {"type": "cached"})
            return

        # Optional: build canonical history for analytics
        _ = await self._build_conversation_history(conversation_id)

        # Get or create ADK session
        user_id = "default"
        adk_session_id = await self._get_or_create_adk_session(conversation_id)

        # Build ADK content
        new_message = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_msg)],
        )

        # Track final assistant message text
        final_text_candidate: str | None = None
        partial_accumulation: str = ""

        try:
            if adk_session_id == "bootstrap":
                # Bootstrap: ADK chooses session_id
                async for msg in self._process_events_bootstrap(
                    user_id, conversation_id, request_id, new_message
                ):
                    yield msg
                    if msg.mtype == "text":
                        mtype = msg.meta.get("type")
                        if mtype == "delta":
                            partial_accumulation += msg.content
                        elif mtype == "final":
                            final_text_candidate = msg.content
                            partial_accumulation = ""
                        else:
                            # treat other text types (warnings, cached)
                            # as part of accumulation
                            partial_accumulation += msg.content
            else:
                # Direct: predetermined session_id
                async for msg in self._process_events_direct(
                    user_id, adk_session_id, conversation_id, request_id, new_message
                ):
                    yield msg
                    if msg.mtype == "text":
                        mtype = msg.meta.get("type")
                        if mtype == "delta":
                            partial_accumulation += msg.content
                        elif mtype == "final":
                            final_text_candidate = msg.content
                            partial_accumulation = ""
                        else:
                            partial_accumulation += msg.content
        except asyncio.CancelledError:
            logger.info("Streaming cancelled by upstream")
            raise
        except Exception as e:
            logger.error("ADK streaming error: %s", e)
            yield ChatMessage("error", f"Streaming interrupted: {e!s}")
            return

        # Determine the final assistant text to persist
        full_assistant_text = (
            final_text_candidate
            if final_text_candidate is not None
            else partial_accumulation
        )
        # Persist final assistant message
        await self.persist.persist_final_assistant_message(
            conversation_id, full_assistant_text, request_id
        )

    # ---------------------------------------------------------------------
    # Public API: non-streaming (optional)
    # ---------------------------------------------------------------------

    async def chat_once(
        self,
        conversation_id: str,
        user_msg: str,
        request_id: str,
    ) -> ChatEvent:
        """Non-streaming wrapper around process_message."""
        chunks: list[str] = []
        async for m in self.process_message(conversation_id, user_msg, request_id):
            if m.mtype == "text":
                chunks.append(m.content)
        final = "".join(chunks)
        return ChatEvent(
            conversation_id=conversation_id,
            type="assistant_message",
            role="assistant",
            content=final,
            usage=Usage(),
            provider="gemini-adk",
            model=self.model,
            extra={"user_request_id": request_id},
        )

    # ---------------------------------------------------------------------
    # Event helpers
    # ---------------------------------------------------------------------

    def _safe_text_from_event(self, event: Event) -> str:
        """Extract concatenated text from an event's content parts."""
        try:
            content = getattr(event, "content", None)
            if not content:
                return ""
            parts = getattr(content, "parts", None)
            if not parts:
                return ""
            out: list[str] = []
            for p in parts:
                text = getattr(p, "text", None)
                if text:
                    out.append(text)
            return "".join(out)
        except Exception:
            return ""

    def _extract_tool_calls(self, event: Event) -> list[dict[str, Any]]:
        """Extract function calls from an event."""
        calls: list[dict[str, Any]] = []
        try:
            for fc in event.get_function_calls() or []:
                calls.append(
                    {
                        "id": getattr(fc, "id", "") or "",
                        "name": getattr(fc, "name", "") or "",
                        "arguments": getattr(fc, "arguments", {}) or {},
                    }
                )
        except Exception:
            pass
        return calls

    def _extract_tool_results(self, event: Event) -> list[dict[str, Any]]:
        """Extract function responses from an event."""
        results: list[dict[str, Any]] = []
        try:
            for fr in event.get_function_responses() or []:
                name = getattr(fr, "name", "") or ""
                call_id = getattr(fr, "function_call_id", "") or ""
                content = getattr(fr, "content", None)
                if content is None:
                    content_str = "✓ done"
                elif isinstance(content, str):
                    content_str = content
                else:
                    try:
                        content_str = json.dumps(content, ensure_ascii=False)
                    except Exception:
                        content_str = str(content)
                results.append({
                    "call_id": call_id,
                    "name": name,
                    "content": content_str,
                })
        except Exception:
            pass
        return results

    async def _persist_tool_calls(
        self,
        conversation_id: str,
        request_id: str,
        calls: list[dict[str, Any]],
    ) -> None:
        """Persist tool_call events in the chat history."""
        if not calls:
            return
        tool_call_objs = [
            ToolCall(id=c["id"], name=c["name"], arguments=c.get("arguments") or {})
            for c in calls
        ]
        ev = ChatEvent(
            conversation_id=conversation_id,
            type="tool_call",
            role="assistant",
            tool_calls=tool_call_objs,
            extra={"user_request_id": request_id},
        )
        await self.repo.add_event(ev)

    async def _persist_tool_results(
        self,
        conversation_id: str,
        request_id: str,
        results: list[dict[str, Any]],
    ) -> None:
        """Persist tool_result events in the chat history."""
        for r in results:
            ev = ChatEvent(
                conversation_id=conversation_id,
                type="tool_result",
                role="tool",
                content=r["content"],
                extra={
                    "user_request_id": request_id,
                    "tool_call_id": r["call_id"],
                    "tool_name": r["name"],
                },
            )
            await self.repo.add_event(ev)

    # ---------------------------------------------------------------------
    # Misc
    # ---------------------------------------------------------------------

    def _pluck_content(self, res: mcp_types.CallToolResult) -> str:
        """Extract content from a tool call result (not used in ADK path)."""
        if not getattr(res, "content", None):
            return "✓ done"
        out: list[str] = []
        for item in res.content:
            if isinstance(item, mcp_types.TextContent):
                out.append(item.text)
            else:
                out.append(f"[{type(item).__name__}]")
        return "\n".join(out)

    async def cleanup(self) -> None:
        """Close ADK sessions and MCP clients."""
        with contextlib.suppress(Exception):
            if self._runner and self._session_service:
                # Runner stores sessions internally; explicit closing not required
                pass
        if self.tool_mgr:
            for client in self.tool_mgr.clients:
                with contextlib.suppress(Exception):
                    await client.close()
                    logger.debug("Closed MCP client: %s", client.name)
