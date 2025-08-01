"""
WebSocket Server for MCP Platform

This module provides a thin communication layer between the frontend and chat service.
It handles WebSocket connections and message routing only.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.history.chat_store import ChatRepository
from src.llm_orchestrators import (
    AnthropicOrchestrator,
    OpenAIOrchestrator,
    OpenAIResponsesOrchestrator,
)
from src.llm_orchestrators.gemini_orchestrator import GeminiAdkOrchestrator

# TYPE_CHECKING imports to avoid circular imports
if TYPE_CHECKING:
    from src.main import MCPClient

logger = logging.getLogger(__name__)


def create_orchestrator(
    clients: list[MCPClient],
    llm_config: dict[str, Any],
    config: dict[str, Any],
    repo: ChatRepository,
) -> (
    OpenAIOrchestrator
    | AnthropicOrchestrator
    | GeminiAdkOrchestrator
    | OpenAIResponsesOrchestrator
):
    """
    Factory function to create the appropriate orchestrator based on config.

    Args:
        clients: List of MCP clients
        llm_config: LLM configuration dictionary
        config: Full configuration dictionary
        repo: Chat repository for persistence

    Returns:
        Appropriate orchestrator instance based on llm.active config

    Raises:
        ValueError: If the provider is not supported
    """
    active_provider = llm_config.get("active", "").lower()

    if active_provider == "anthropic":
        return AnthropicOrchestrator(clients, llm_config, config, repo)
    if active_provider == "gemini":
        return GeminiAdkOrchestrator(clients, llm_config, config, repo)
    if active_provider == "openai_responses":
        return OpenAIResponsesOrchestrator(clients, llm_config, config, repo)
    if active_provider in ("openai", "groq", "openrouter", "azure"):
        return OpenAIOrchestrator(clients, llm_config, config, repo)

    raise ValueError(
        f"Unsupported LLM provider: '{active_provider}'. "
        f"Supported providers: openai, openai_responses, groq, openrouter, "
        f"azure, anthropic, gemini"
    )


class WebSocketServer:
    """
    Pure WebSocket communication server.

    This class only handles:
    - WebSocket connections
    - Message parsing and routing
    - Response streaming

    All business logic is delegated to the appropriate orchestrator.
    """

    def __init__(
        self,
        clients: list[MCPClient],
        llm_config: dict[str, Any],
        config: dict[str, Any],
        repo: ChatRepository,
    ):
        self.chat_service = create_orchestrator(clients, llm_config, config, repo)
        self.repo = repo
        self.config = config
        self.llm_config = llm_config  # Store for provider info
        self.app = self._create_app()
        self.active_connections: list[WebSocket] = []
        # Store conversation id per socket
        self.conversation_ids: dict[WebSocket, str] = {}

    def _get_provider_info(self) -> dict[str, Any]:
        """Get current provider and model information for frontend optimization."""
        provider = self.llm_config.get("active", "unknown")
        providers_config = self.llm_config.get("providers", {})
        provider_config = providers_config.get(provider, {})
        model = provider_config.get("model", "unknown")

        return {
            "provider": provider,
            "model": model,
            "orchestrator_type": type(self.chat_service).__name__
        }

    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI app."""
        app = FastAPI(title="MCP WebSocket Chat Server")

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.websocket("/ws/chat")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_websocket_connection(websocket)

        @app.get("/")
        async def root():
            return {"message": "MCP WebSocket Chat Server"}

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        return app

    async def _handle_websocket_connection(self, websocket: WebSocket):
        """Handle a WebSocket connection."""
        await self._connect_websocket(websocket)

        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)

                # Handle message based on action
                if message_data.get("action") == "chat":
                    await self._handle_chat_message(websocket, message_data)
                elif message_data.get("action") == "clear_session":
                    await self._handle_clear_session(websocket, message_data)
                else:
                    # Unknown message format
                    logger.warning(f"Unknown message format: {message_data}")
                    await websocket.send_text(
                        json.dumps(
                            {
                                "status": "error",
                                "chunk": {
                                    "error": (
                                        "Unknown message format. "
                                        "Expected 'action': 'chat' or 'clear_session'"
                                    )
                                },
                            }
                        )
                    )

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.send_text(
                json.dumps(
                    {
                        "status": "error",
                        "chunk": {"error": f"Server error: {e!s}"},
                    }
                )
            )
        finally:
            self._disconnect_websocket(websocket)

    async def _handle_chat_message(
        self, websocket: WebSocket, message_data: dict[str, Any]
    ):
        """Handle a chat message from the frontend."""
        request_id = message_data.get("request_id")
        if not request_id:
            await websocket.send_text(
                json.dumps(
                    {
                        "status": "error",
                        "chunk": {"error": "request_id is required"},
                    }
                )
            )
            return

        payload = message_data.get("payload", {})
        user_message = payload.get("text", "")

        # Check streaming configuration - FAIL FAST approach
        service_config = self.config.get("chat", {}).get("service", {})
        streaming_config = service_config.get("streaming", {})

        if "streaming" in payload:
            # Client explicitly set streaming preference - use it
            streaming = payload["streaming"]
        elif streaming_config.get("enabled") is not None:
            # Use configured default - must be explicitly set
            streaming = streaming_config["enabled"]
        else:
            # FAIL FAST: No streaming configuration found
            await websocket.send_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": "error",
                        "chunk": {
                            "error": (
                                "Streaming configuration missing. "
                                "Set 'chat.service.streaming.enabled' in config.yaml "
                                "or specify 'streaming: true/false' in payload."
                            )
                        },
                    }
                )
            )
            return

        logger.info(f"Processing message with streaming={streaming}")

        logger.info(f"Received chat message: {user_message[:50]}...")

        try:
            provider_info = self._get_provider_info()
            await websocket.send_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": "processing",
                        "chunk": {
                            "metadata": {
                                "user_message": user_message,
                                "provider_info": provider_info
                            }
                        },
                    }
                )
            )

            # get or assign conversation_id
            conversation_id = self.conversation_ids.get(websocket)
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
                self.conversation_ids[websocket] = conversation_id

            if streaming:
                # Streaming mode: real-time chunks
                await self._handle_streaming_chat(
                    websocket, request_id, conversation_id, user_message
                )
            else:
                # Non-streaming mode: single final assistant message
                await self._handle_non_streaming_chat(
                    websocket, request_id, conversation_id, user_message
                )

        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            await websocket.send_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": "error",
                        "chunk": {"error": str(e)},
                    }
                )
            )

    async def _handle_clear_session(
        self, websocket: WebSocket, message_data: dict[str, Any]
    ):
        """Handle a clear session request from the frontend."""
        request_id = message_data.get("request_id")
        if not request_id:
            await websocket.send_text(
                json.dumps(
                    {
                        "status": "error",
                        "chunk": {"error": "request_id is required"},
                    }
                )
            )
            return

        try:
            # Generate a new conversation ID for this WebSocket connection
            new_conversation_id = str(uuid.uuid4())
            self.conversation_ids[websocket] = new_conversation_id

            logger.info(
                f"Session cleared for websocket, "
                f"new conversation_id: {new_conversation_id}"
            )

            # Send success response
            await websocket.send_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": "complete",
                        "chunk": {
                            "type": "session_cleared",
                            "data": "Session cleared successfully",
                            "metadata": {"new_conversation_id": new_conversation_id},
                        },
                    }
                )
            )

        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            await websocket.send_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": "error",
                        "chunk": {"error": str(e)},
                    }
                )
            )

    async def _handle_streaming_chat(
        self,
        websocket: WebSocket,
        request_id: str,
        conversation_id: str,
        user_message: str,
    ):
        """Handle streaming chat response with unified persistence."""
        # Use the unified streaming approach - no manual persistence needed
        # The chat service now handles all persistence internally
        async for chat_message in self.chat_service.process_message(
            conversation_id, user_message, request_id
        ):
            await self._send_chat_response(websocket, request_id, chat_message)

        # Send completion signal
        await websocket.send_text(
            json.dumps(
                {
                    "request_id": request_id,
                    "status": "complete",
                    "chunk": {},
                }
            )
        )

    async def _handle_non_streaming_chat(
        self,
        websocket: WebSocket,
        request_id: str,
        conversation_id: str,
        user_message: str,
    ):
        """Handle non-streaming chat response using chat_once for true non-streaming."""
        try:
            # Use chat_once for true non-streaming (no streaming API calls)
            chat_event = await self.chat_service.chat_once(
                conversation_id, user_message, request_id
            )

            # Send the complete response
            if chat_event.content:
                provider_info = self._get_provider_info()
                metadata = {
                    "provider_info": provider_info,
                    "non_streaming": True,
                    "usage": (
                        chat_event.usage.model_dump()
                        if chat_event.usage else None
                    ),
                    "model": chat_event.model,
                    "provider": chat_event.provider
                }

                await websocket.send_text(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "status": "chunk",
                            "chunk": {
                                "type": "text",
                                "data": chat_event.content,
                                "metadata": metadata,
                            },
                        }
                    )
                )

            # Send completion signal
            await websocket.send_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": "complete",
                    }
                )
            )
        except Exception as e:
            logger.error(f"Error in non-streaming chat: {e}")
            await websocket.send_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": "error",
                        "error": str(e),
                    }
                )
            )

    async def _send_chat_response(
        self, websocket: WebSocket, request_id: str, chat_message
    ):
        """Send a chat response to the frontend."""
        logger.info(
            "Sending WebSocket message: "
            f"type={chat_message.type}, "
            f"content={chat_message.content[:50]}..."
        )

        # Get provider info for frontend optimization
        provider_info = self._get_provider_info()

        # Convert chat service message to frontend format
        if chat_message.type == "text":
            # Only send text messages that aren't tool results
            if not chat_message.meta.get("tool_result"):
                # Add provider info to metadata for frontend optimization
                enhanced_metadata = {
                    **chat_message.meta,
                    "provider_info": provider_info
                }
                await websocket.send_text(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "status": "chunk",
                            "chunk": {
                                "type": "text",
                                "data": chat_message.content,
                                "metadata": enhanced_metadata,
                            },
                        }
                    )
                )

        elif chat_message.type == "tool_execution":
            await websocket.send_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": "processing",
                        "chunk": {
                            "type": "tool_execution",
                            "data": chat_message.content,
                            "metadata": chat_message.meta,
                        },
                    }
                )
            )

        elif chat_message.type == "error":
            # Add provider info to error metadata for context
            enhanced_error_metadata = {
                **chat_message.meta,
                "provider_info": provider_info
            }
            await websocket.send_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": "error",
                        "chunk": {
                            "error": chat_message.content,
                            "metadata": enhanced_error_metadata,
                        },
                    }
                )
            )

    async def _connect_websocket(self, websocket: WebSocket):
        """Connect a WebSocket."""
        try:
            logger.info(f"WebSocket connection attempt from {websocket.client}")
            await websocket.accept()
            self.active_connections.append(websocket)
            # Initialize conversation id for this connection
            self.conversation_ids[websocket] = str(uuid.uuid4())
            logger.info(
                f"WebSocket connection established. Total connections: "
                f"{len(self.active_connections)}"
            )
        except Exception as e:
            logger.error(f"Failed to accept WebSocket connection: {e}")
            raise

    def _disconnect_websocket(self, websocket: WebSocket):
        """Disconnect a WebSocket."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        # Clean up conversation id
        if websocket in self.conversation_ids:
            del self.conversation_ids[websocket]
        logger.info(
            f"WebSocket connection closed. Total connections: "
            f"{len(self.active_connections)}"
        )

    async def start_server(self):
        """Start the WebSocket server."""
        # Initialize chat service
        await self.chat_service.initialize()

        # Start server
        host = self.config.get("websocket", {}).get("host", "localhost")
        port = self.config.get("websocket", {}).get("port", 8000)

        logger.info(f"Starting WebSocket server on {host}:{port}")

        server_config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(server_config)

        try:
            await server.serve()
        finally:
            await self.chat_service.cleanup()


async def run_websocket_server(
    clients: list[MCPClient],
    llm_config: dict[str, Any],
    config: dict[str, Any],
    repo: ChatRepository,
) -> None:
    """
    Run the WebSocket server.

    This function maintains the same interface as before but now uses
    the clean separation between communication and business logic.
    """
    server = WebSocketServer(clients, llm_config, config, repo)
    await server.start_server()
