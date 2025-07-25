"""
WebSocket Server for MCP Platform

This module provides a thin communication layer between the frontend and chat service.
It handles WebSocket connections and message routing only.
"""

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.chat_service import ChatService
from src.history.chat_store import ChatRepository

# TYPE_CHECKING imports to avoid circular imports
if TYPE_CHECKING:
    from src.main import LLMClient, MCPClient

logger = logging.getLogger(__name__)


class WebSocketServer:
    """
    Pure WebSocket communication server.

    This class only handles:
    - WebSocket connections
    - Message parsing and routing
    - Response streaming

    All business logic is delegated to ChatService.
    """

    def __init__(
        self,
        clients: list["MCPClient"],
        llm_client: "LLMClient",
        config: dict[str, Any],
        repo: ChatRepository,
    ):
        self.chat_service = ChatService(clients, llm_client, config, repo=repo)
        self.repo = repo
        self.config = config
        self.app = self._create_app()
        self.active_connections: list[WebSocket] = []
        # Store conversation id per socket
        self.conversation_ids: dict[WebSocket, str] = {}

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
                                        "Expected 'action': 'chat'"
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
        request_id = message_data.get("request_id", str(uuid.uuid4()))
        payload = message_data.get("payload", {})
        user_message = payload.get("text", "")
        model = payload.get("model")  # optional

        logger.info(f"Received chat message: {user_message[:50]}...")

        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": "processing",
                        "chunk": {"metadata": {"user_message": user_message}},
                    }
                )
            )

            # get or assign conversation_id
            conversation_id = self.conversation_ids.get(websocket)
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
                self.conversation_ids[websocket] = conversation_id

            # Non-streaming: single final assistant message
            assistant_ev = await self.chat_service.chat_once(
                conversation_id=conversation_id,
                user_msg=user_message,
                model=model,
                request_id=request_id,
            )

            await websocket.send_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": "chunk",
                        "chunk": {
                            "type": "text",
                            "data": assistant_ev.content,
                            "metadata": {},
                        },
                    }
                )
            )

            await websocket.send_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": "complete",
                        "chunk": {},
                    }
                )
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

    async def _send_chat_response(
        self, websocket: WebSocket, request_id: str, chat_message
    ):
        """Send a chat response to the frontend."""
        logger.info(
            "Sending WebSocket message: "
            f"type={chat_message.type}, "
            f"content={chat_message.content[:50]}..."
        )

        # Convert chat service message to frontend format
        if chat_message.type == "text":
            # Only send text messages that aren't tool results
            if not chat_message.metadata.get("tool_result"):
                await websocket.send_text(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "status": "chunk",
                            "chunk": {
                                "type": "text",
                                "data": chat_message.content,
                                "metadata": chat_message.metadata,
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
                            "metadata": chat_message.metadata,
                        },
                    }
                )
            )

        elif chat_message.type == "error":
            await websocket.send_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": "error",
                        "chunk": {
                            "error": chat_message.content,
                            "metadata": chat_message.metadata,
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
    clients: list["MCPClient"],
    llm_client: "LLMClient",
    config: dict[str, Any],
    repo: ChatRepository,
) -> None:
    """
    Run the WebSocket server.

    This function maintains the same interface as before but now uses
    the clean separation between communication and business logic.
    """
    server = WebSocketServer(clients, llm_client, config, repo)
    await server.start_server()
