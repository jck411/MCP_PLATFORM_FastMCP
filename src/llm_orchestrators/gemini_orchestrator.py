"""
Gemini ADK Orchestrator for MCP Platform

This orchestrator uses Google's Agent Development Kit (ADK) 1.8.0 with native
MCP support:
  • Connects to MCP servers natively using MCPToolset
  • Runs a Gemini model with first-class tool use via ADK
  • Streams ADK events and persists user/assistant text, tool calls, and tool
    results
  • Preserves the public API and persistence flow consistent with other
    orchestrators

ADK 1.8.0 provides:
- MCPToolset for direct MCP server integration
- LlmAgent for Gemini model orchestration
- Runner for session management and streaming
- Proper handling of function calls/responses automatically

This avoids the manual functionCall/functionResponse protocol issues that
plagued the hand-rolled Gemini implementation.
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

# Google GenAI types for message building (ADK-compatible)
import google.genai.types as genai_types

# --- ADK imports (correct for 1.8.0) ---
from google.adk.agents import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.events import Event
from google.adk.models import Gemini
from google.adk.runners import Runner
# RunConfig and StreamingMode for streaming configuration
try:
    from google.adk.agents.run_config import RunConfig, StreamingMode
except ImportError:
    # If not available, we'll try to configure streaming via other means
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

# --- Your platform bits (unchanged) ---
from src.history.chat_store import ChatEvent, ChatRepository, ToolCall, Usage
from src.history.persistence import ConversationPersistenceService
from src.mcp_services.prompting import MCPResourcePromptService
from src.schema_manager import SchemaManager

if TYPE_CHECKING:
    from src.main import MCPClient

logger = logging.getLogger(__name__)

# Parity with your other orchestrators
MAX_TOOL_HOPS = 8
MAX_TEXT_DELTA_SIZE = 10_000


# -----------------------------
# Exceptions
# -----------------------------
class OrchestratorError(Exception):
    """Base class for orchestrator errors."""


class InitializationError(OrchestratorError):
    """Raised when orchestrator is used before initialization."""


class TransportError(OrchestratorError):
    """Raised when HTTP/stream transport contracts are violated."""


class UpstreamProtocolError(OrchestratorError):
    """Raised when upstream provider protocol is violated/unexpected."""


class ToolExecutionError(OrchestratorError):
    """Raised for tool execution / argument parse failures."""


# -----------------------------
# Data Models (compat)
# -----------------------------
@dataclass(slots=True)
class ChatMessage:
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
    def type(self) -> str:
        return self.mtype


class GeminiAdkOrchestrator:
    """
    Conversation orchestrator using Google ADK + Gemini with native MCP tools.

    Key advantages over hand-rolled Gemini implementation:
      • ADK handles functionCall/functionResponse protocol correctly
      • MCPToolset introspects servers and wires tools automatically
      • Tool calls/results are surfaced as ADK Events for persistence

    Public API mirrors your other orchestrators:
      - initialize()
      - process_message(conversation_id, user_msg, request_id)
        -> AsyncGenerator[ChatMessage]
      - chat_once(...)
      - cleanup()
    """

    def __init__(
        self,
        clients: list[MCPClient],  # kept for parity with other orchestrators
        llm_config: dict[str, Any],
        config: dict[str, Any],
        repo: ChatRepository,
        max_history_messages: int = 50,
    ):
        self.clients = clients
        self.llm_config = llm_config
        self.config = config
        self.repo = repo
        self.persist = ConversationPersistenceService(repo)
        self.max_history_messages = max_history_messages
        self.chat_conf = config.get("chat", {}).get("service", {}) or {}

        # We retain SchemaManager just to build a system prompt and resource catalog
        self.tool_mgr: SchemaManager | None = None

        self._init_lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self.mcp_prompt: MCPResourcePromptService | None = None
        self._system_prompt: str = ""

        # Provider config (model, temperature, etc.)
        self.active_name = (llm_config.get("active") or "").lower()
        providers = llm_config.get("providers") or {}
        self.provider_cfg = providers.get(self.active_name, {}) or {}

        # API Key for Gemini (AI Studio or Vertex, depending on env)
        self.api_key = self._extract_api_key()

        # ADK members
        self._runner: Runner | None = None
        self._agent: LlmAgent | None = None
        self._session_service: InMemorySessionService | None = None
        self._artifact_service: InMemoryArtifactService | None = None
        self._mcp_toolsets: list[MCPToolset] = []

        # Session mapping for bootstrap pattern fallback
        # Maps conversation_id -> adk_session_id for cases where ADK creates its own IDs
        self._session_id_mapping: dict[str, str] = {}

    # -----------------------------------
    # Configuration helpers
    # -----------------------------------
    def _extract_api_key(self) -> str:
        """
        Extract API key from env or provider config; raise if absent.

        For AI Studio style:
          export GOOGLE_API_KEY=...
        Or use providers['gemini']['api_key'] in llm_config.
        """
        # Try a few common envs
        env_order = [
            "GOOGLE_API_KEY",              # AI Studio
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
        """
        Gemini model, e.g. "gemini-1.5-pro" / "gemini-2.0-flash-thinking-exp".
        """
        return self.provider_cfg.get("model", "gemini-1.5-pro")

    # -----------------------------------
    # Initialization
    # -----------------------------------
    async def initialize(self) -> None:
        async with self._init_lock:
            if self._ready.is_set():
                return

            logger.info("GeminiAdkOrchestrator.initialize() starting...")

            # 1) Connect your existing MCP clients just to harvest prompts/resources
            #    (ADK will connect to MCP separately; this preserves your UX)
            try:
                connection_results = await asyncio.gather(
                    *(c.connect() for c in self.clients), return_exceptions=True
                )

                connected_clients = []
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

                # Build system prompt from your MCP resource catalog (same as others)
                self.mcp_prompt = MCPResourcePromptService(
                    self.tool_mgr, self.chat_conf
                )
                await self.mcp_prompt.update_resource_catalog_on_availability()
                self._system_prompt = await self.mcp_prompt.make_system_prompt()

                logger.info(
                    "GeminiAdkOrchestrator: %d MCP tools (prompt-side), %d resources.",
                    len(self.tool_mgr.get_mcp_tools()),
                    len(self.mcp_prompt.resource_catalog if self.mcp_prompt else []),
                )
            except Exception as e:
                logger.warning(
                    "SchemaManager/MCP prompt init failed (non-fatal): %s", e
                )

            # 2) Build ADK MCP toolsets (native)
            try:
                # Load servers config and convert to ADK format
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
                        f"Configured {len(self._mcp_toolsets)} ADK MCP toolsets"
                    )
            except Exception as e:
                logger.error("Failed to configure ADK MCP toolsets: %s", e)
                raise InitializationError(f"ADK MCP setup failed: {e}") from e

            # 3) Create ADK Agent (Gemini + MCP tools)
            try:
                llm = Gemini(model=self.model)
                tools = list(self._mcp_toolsets)  # MCPToolset objects are tools
                self._agent = LlmAgent(
                    name="gemini_mcp_agent",  # Required name field (valid identifier)
                    model=llm,
                    instruction=self._system_prompt or "",
                    tools=tools,  # type: ignore[arg-type]
                    # You can pass generation config overrides here if desired.
                )
            except Exception as e:
                logger.error("Failed to create ADK LlmAgent: %s", e)
                raise InitializationError(f"ADK agent creation failed: {e}") from e

            # 4) Create ADK Runner + in-memory services (we still persist to your repo)
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
        """
        Build ADK MCPToolset objects from config.
        Example config:
          config = {
            "mcp": {
              "servers": [
                {
                  "name": "fs",
                  "kind": "stdio",
                  "command": "uvx",
                  "args": ["my-fs-server"],
                  "env": {"FOO": "bar"}
                },
                {
                  "name": "calendar",
                  "kind": "sse",
                  "url": "http://localhost:3000/sse",
                  "headers": {"Authorization": "Bearer ..."}
                }
              ]
            }
          }
        """
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
                    msg = f"MCP stdio server '{name}' missing 'command'"
                    raise InitializationError(msg)
                server_params = StdioServerParameters(command=cmd, args=args, env=env)
                params = StdioConnectionParams(server_params=server_params)
                toolsets.append(
                    MCPToolset(connection_params=params)
                )
            elif kind == "sse":
                url = s.get("url")
                headers = s.get("headers") or {}
                if not url:
                    msg = f"MCP SSE server '{name}' missing 'url'"
                    raise InitializationError(msg)
                params = SseConnectionParams(url=url, headers=headers)
                toolsets.append(
                    MCPToolset(connection_params=params)
                )
            else:
                msg = f"Unknown MCP server kind '{kind}' for '{name}'"
                raise InitializationError(msg)

        return toolsets

    def _load_mcp_servers_config(self) -> dict[str, Any]:
        """
        Load MCP servers config from servers_config.json and convert to ADK format.
        
        Converts from:
          {
            "mcpServers": {
              "demo": {
                "enabled": true,
                "command": "uv",
                "args": ["run", "python", "Servers/demo_server.py"],
                "cwd": "/path/to/workspace"
              }
            }
          }
        
        To ADK format:
          {
            "servers": [
              {
                "name": "demo",
                "kind": "stdio",
                "command": "uv",
                "args": ["run", "python", "Servers/demo_server.py"],
                "env": {"PYTHONPATH": "/path/to/workspace"}
              }
            ]
          }
        """
        try:
            # Look for servers_config.json in the src directory
            config_path = os.path.join(os.path.dirname(__file__), "..", "servers_config.json")
            config_path = os.path.abspath(config_path)
            
            if not os.path.exists(config_path):
                logger.warning(f"MCP servers config not found at {config_path}")
                return {"servers": []}
            
            with open(config_path, "r") as f:
                config_data = json.load(f)
            
            mcp_servers = config_data.get("mcpServers", {})
            adk_servers = []
            
            for server_name, server_config in mcp_servers.items():
                if not server_config.get("enabled", False):
                    continue
                
                command = server_config.get("command")
                args = server_config.get("args", [])
                cwd = server_config.get("cwd")
                
                if not command:
                    logger.warning(f"MCP server '{server_name}' missing command, skipping")
                    continue
                
                # Convert to ADK format
                adk_server = {
                    "name": server_name,
                    "kind": "stdio",  # All our servers are stdio for now
                    "command": command,
                    "args": args,
                }
                
                # Add working directory as environment variable if specified
                if cwd:
                    adk_server["env"] = {"PYTHONPATH": cwd}
                
                adk_servers.append(adk_server)
                logger.info(f"Converted MCP server '{server_name}' for ADK")
            
            return {"servers": adk_servers}
            
        except Exception as e:
            logger.error(f"Failed to load MCP servers config: {e}")
            return {"servers": []}

    # -----------------------------------
    # ADK Session Management
    # -----------------------------------
    async def _get_or_create_adk_session(self, conversation_id: str) -> str:
        """
        Get or create an ADK session for a conversation.

        This implements a hybrid approach:
        1. Try to use conversation_id directly (preferred)
        2. Fallback to bootstrap pattern if ADK creates its own session IDs

        Returns the actual session_id to use with Runner.run_async
        """
        assert self._runner is not None and self._session_service is not None

        user_id = "default"

        # Check if we have a mapped session ID from bootstrap
        if conversation_id in self._session_id_mapping:
            adk_session_id = self._session_id_mapping[conversation_id]
            logger.debug(
                "Using mapped ADK session: %s -> %s", conversation_id, adk_session_id
            )
            return adk_session_id

        # Try to create/use conversation_id directly (preferred pattern)
        try:
            await self._ensure_adk_session_with_id(user_id, conversation_id)
            logger.debug("Using conversation_id as session_id: %s", conversation_id)
            return conversation_id
        except Exception as e:
            logger.warning(
                "Direct session creation failed: %s. Will try bootstrap pattern.", e
            )

        # Bootstrap fallback: let ADK create its own session ID
        logger.info("Using bootstrap pattern for conversation: %s", conversation_id)
        return "bootstrap"  # Special marker to indicate bootstrap mode

    async def _ensure_adk_session_with_id(self, user_id: str, session_id: str) -> None:
        """
        Ensure an ADK session exists with a specific session_id.

        Based on official ADK tutorial pattern: create session explicitly
        using session_service.create_session() before any Runner.run_async calls.
        """
        # ADK requires explicit session creation BEFORE Runner.run_async
        # This follows the exact pattern from the official ADK tutorial
        try:
            # Try to create the session (idempotent - won't fail if exists)
            try:
                await self._session_service.create_session(
                    app_name="mcp-platform",
                    user_id=user_id,
                    session_id=session_id
                )
                logger.debug(
                    "ADK session created/verified: App='%s', User='%s', Session='%s'",
                    "mcp-platform", user_id, session_id
                )
                return
            except TypeError:
                # Some ADK builds don't require app_name parameter
                # Type ignore since some builds require app_name
                await self._session_service.create_session(  # type: ignore[call-arg]
                    user_id=user_id,
                    session_id=session_id
                )
                logger.debug(
                    "ADK session created/verified (no app_name): "
                    "User='%s', Session='%s'",
                    user_id, session_id
                )
                return
        except Exception as e:
            # If create fails, try to get existing session
            try:
                await self._session_service.get_session(
                    app_name="mcp-platform",
                    user_id=user_id,
                    session_id=session_id
                )
                logger.debug(
                    "ADK session already exists: %s/%s", user_id, session_id
                )
                return
            except (TypeError, Exception):
                # Final fallback - try get without app_name
                try:
                    # Type ignore since some builds require app_name
                    await self._session_service.get_session(  # type: ignore[call-arg]
                        user_id=user_id,
                        session_id=session_id
                    )
                    logger.debug(
                        "ADK session found (no app_name): %s/%s", user_id, session_id
                    )
                    return
                except Exception:
                    pass

            logger.error(
                "Failed to ensure ADK session %s/%s: %s",
                user_id, session_id, e
            )
            raise InitializationError(f"ADK session setup failed: {e}") from e

    # -----------------------------------
    # Canonical history (for your repo)
    # -----------------------------------
    async def _build_conversation_history(
        self, conversation_id: str
    ) -> list[dict[str, Any]]:
        """
        We still read your stored events to understand prior turns, but:
        - We do NOT replay them into ADK (ADK maintains its own session state).
        - We only use this history for persistence parity / analytics.
        """
        events = await self.repo.get_events(conversation_id, self.max_history_messages)
        conv: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt}
        ]
        for ev in events:
            if ev.type in ("user_message", "assistant_message", "system_update"):
                conv.append({"role": ev.role, "content": ev.content})
        return conv

    # -----------------------------------
    # Public API: process_message (streaming)
    # -----------------------------------
    async def _process_events_bootstrap(
        self,
        user_id: str,
        conversation_id: str,
        request_id: str,
        new_message: genai_types.Content,
    ) -> AsyncGenerator[ChatMessage]:
        """Process events using bootstrap pattern (no predetermined session_id)."""
        tool_hops = 0
        first_event = True

        # Type ignore since session_id is optional in some ADK builds for bootstrap
        async for event in self._runner.run_async(  # type: ignore[call-arg]
            user_id=user_id,
            # Don't pass session_id for bootstrap - let ADK create one
            new_message=new_message,
        ):
            # Capture the real session ID from the first event
            if first_event and hasattr(event, 'session_id'):
                real_session_id = getattr(event, 'session_id', None)
                if real_session_id:
                    self._session_id_mapping[conversation_id] = real_session_id
                    logger.info(
                        "Mapped conversation %s to ADK session %s",
                        conversation_id, real_session_id
                    )
                first_event = False

            tool_hops, yielded_text = await self._handle_event_processing(
                event, conversation_id, request_id, tool_hops
            )

            if yielded_text:
                if tool_hops > MAX_TOOL_HOPS:
                    # This is a warning message about hitting tool limit
                    yield ChatMessage("text", yielded_text)
                    break

                # This is final response text
                is_truncated = len(yielded_text) <= MAX_TEXT_DELTA_SIZE
                msg_type = "final" if is_truncated else "final_truncated"
                yield ChatMessage("text", yielded_text, {"type": msg_type})

    async def _process_events_direct(
        self,
        user_id: str,
        adk_session_id: str,
        conversation_id: str,
        request_id: str,
        new_message: genai_types.Content,
    ) -> AsyncGenerator[ChatMessage]:
        """Process events using direct session pattern (predetermined session_id)."""
        tool_hops = 0

        # Configure streaming if available
        run_config = None
        if RunConfig and StreamingMode:
            try:
                run_config = RunConfig(
                    streaming_mode=StreamingMode.SSE,  # Enable streaming
                    max_llm_calls=10  # Reasonable limit
                )
                logger.info("Configured ADK streaming with SSE mode")
            except Exception as e:
                logger.warning(f"Could not configure ADK streaming: {e}")

        # Call run_async with or without run_config based on availability
        if run_config:
            event_stream = self._runner.run_async(
                user_id=user_id,
                session_id=adk_session_id,
                new_message=new_message,
                run_config=run_config,  # Pass the streaming config
            )
        else:
            # Fallback without run_config
            event_stream = self._runner.run_async(
                user_id=user_id,
                session_id=adk_session_id,
                new_message=new_message,
            )

        async for event in event_stream:
            tool_hops, yielded_text = await self._handle_event_processing(
                event, conversation_id, request_id, tool_hops
            )

            if yielded_text:
                if tool_hops > MAX_TOOL_HOPS:
                    # This is a warning message about hitting tool limit
                    yield ChatMessage("text", yielded_text)
                    break

                # This is final response text
                is_truncated = len(yielded_text) <= MAX_TEXT_DELTA_SIZE
                msg_type = "final" if is_truncated else "final_truncated"
                yield ChatMessage("text", yielded_text, {"type": msg_type})

    async def _handle_event_processing(
        self,
        event: Event,
        conversation_id: str,
        request_id: str,
        tool_hops: int,
    ) -> tuple[int, str | None]:
        """
        Process a single ADK event and return (updated_tool_hops, yielded_text).
        Returns None for yielded_text if no text should be yielded.
        """
        # Handle tool calls
        tool_calls = self._extract_tool_calls(event)
        if tool_calls:
            await self._persist_tool_calls(
                conversation_id, request_id, tool_calls
            )
            tool_hops += 1
            if tool_hops > MAX_TOOL_HOPS:
                warn = f"⚠️ Reached maximum tool call limit ({MAX_TOOL_HOPS})."
                return tool_hops, warn

        # Handle tool results
        tool_results = self._extract_tool_results(event)
        if tool_results:
            await self._persist_tool_results(
                conversation_id, request_id, tool_results
            )

        # Handle streaming text content (both partial and final)
        delta = self._safe_text_from_event(event)
        if delta:
            if len(delta) <= MAX_TEXT_DELTA_SIZE:
                return tool_hops, delta

            # If text is too large, truncate but still stream it
            logger.warning("Text chunk too large; truncating for UI")
            return tool_hops, delta[:MAX_TEXT_DELTA_SIZE]

        return tool_hops, None

    async def process_message(
        self,
        conversation_id: str,
        user_msg: str,
        request_id: str,
    ) -> AsyncGenerator[ChatMessage]:
        """
        Streaming response. We:
          1) persist the user message
          2) send the user message to the ADK runner (session_id = conversation_id)
          3) stream ADK Events:
                - final assistant text -> yield to UI + buffer
                - function_calls -> persist tool_call
                - function_responses -> persist tool_result
          4) persist final assistant message
        """
        await self._ready.wait()
        if not self._runner or not self._agent:
            raise InitializationError("ADK runner/agent not initialized")

        logger.info(
            "GeminiAdkOrchestrator.process_message called: "
            "conversation_id=%s, msg='%s'",
            conversation_id,
            user_msg[:100] + "..." if len(user_msg) > 100 else user_msg
        )

        # idempotent persistence of user message
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

        # build canonical history (for your analytics/logging, optional)
        _ = await self._build_conversation_history(conversation_id)

        # Get or create ADK session with hybrid approach
        user_id = "default"
        adk_session_id = await self._get_or_create_adk_session(conversation_id)

        # Build ADK message content using correct genai types
        new_message = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_msg)],
        )

        full_assistant_text: str = ""

        try:
            # Use appropriate event processing pattern based on session type
            if adk_session_id == "bootstrap":
                # Bootstrap pattern: let ADK create session, capture the real ID
                async for msg in self._process_events_bootstrap(
                    user_id, conversation_id, request_id, new_message
                ):
                    yield msg
                    if msg.mtype == "text":
                        full_assistant_text += msg.content
            else:
                # Direct session pattern: use predetermined session_id
                async for msg in self._process_events_direct(
                    user_id, adk_session_id, conversation_id, request_id, new_message
                ):
                    yield msg
                    if msg.mtype == "text":
                        full_assistant_text += msg.content

        except asyncio.CancelledError:
            logger.info("Streaming cancelled by upstream")
            raise
        except Exception as e:
            logger.error("ADK streaming error: %s", e)
            yield ChatMessage("error", f"Streaming interrupted: {e!s}")
            return

        # Persist final assistant message
        await self.persist.persist_final_assistant_message(
            conversation_id, full_assistant_text, request_id
        )

    # -----------------------------------
    # Public API: non-streaming (optional)
    # -----------------------------------
    async def chat_once(
        self,
        conversation_id: str,
        user_msg: str,
        request_id: str,
    ) -> ChatEvent:
        """
        Non-streaming fire-and-wait wrapper around process_message.
        """
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
            usage=Usage(),            # ADK does not yet expose token usage
            provider="gemini-adk",
            model=self.model,
            extra={"user_request_id": request_id},
        )

    # -----------------------------------
    # ADK Event helpers
    # -----------------------------------
    def _safe_text_from_event(self, event: Event) -> str:
        """
        Extract text from a final ADK Event (robust to None parts).
        """
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
        """
        Use ADK's helper to read function calls from an Event.
        Returns a list of dicts: { "id": str, "name": str, "arguments": dict }
        """
        calls = []
        try:
            for fc in event.get_function_calls() or []:
                # fc has: id, name, arguments (already a dict)
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
        """
        Use ADK's helper to read function responses from an Event.
        Returns a list of dicts: { "call_id": str, "name": str, "content": str }
        """
        results = []
        try:
            for fr in event.get_function_responses() or []:
                # fr has: function_call_id, name, content (string or structured)
                name = getattr(fr, "name", "") or ""
                call_id = getattr(fr, "function_call_id", "") or ""
                content = getattr(fr, "content", None)
                if content is None:
                    content_str = "✓ done"
                elif isinstance(content, str):
                    content_str = content
                else:
                    # serialize non-string content defensively
                    try:
                        content_str = json.dumps(content, ensure_ascii=False)
                    except Exception:
                        content_str = str(content)
                results.append({
                    "call_id": call_id,
                    "name": name,
                    "content": content_str
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
        """
        Persist tool_call events in your existing repo format.
        """
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
        """
        Persist tool_result events in your existing repo format.
        """
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

    # -----------------------------------
    # Misc
    # -----------------------------------
    def _pluck_content(self, res: mcp_types.CallToolResult) -> str:
        """
        Kept for parity; not used by ADK path since ADK executes tools natively.
        """
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
        """
        Close ADK session (idempotent) and your MCP clients used for prompts.
        """
        with contextlib.suppress(Exception):
            if self._runner and self._session_service:
                # Runner stores sessions internally; close out the conversation sessions
                # (If you want to explicitly close, you can call close_session
                # for ids you tracked)
                pass

        if self.tool_mgr:
            for client in self.tool_mgr.clients:
                with contextlib.suppress(Exception):
                    await client.close()
                    logger.debug("Closed MCP client: %s", client.name)
