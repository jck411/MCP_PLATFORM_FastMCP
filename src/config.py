"""Configuration management for the MCP client."""

import json
import os
from typing import Any

import yaml
from dotenv import load_dotenv


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration from YAML and environment variables."""
        self.load_env()  # Load .env for API keys
        self._config = self._load_yaml_config()  # Load YAML config

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    def _load_yaml_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path) as file:
            return yaml.safe_load(file)

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path) as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the API key for the active LLM provider.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        # Get active provider
        llm_config = self._config.get("llm", {})
        active_provider = llm_config.get("active", "openai")

        # Map provider names to environment variable names
        provider_key_map = {
            "openai": "OPENAI_API_KEY",
            "openai_responses": "OPENAI_API_KEY",  # Uses same OpenAI API key
            "groq": "GROQ_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "mistral": "MISTRAL_API_KEY",
        }

        env_key = provider_key_map.get(active_provider)
        if not env_key:
            raise ValueError(
                f"Unknown provider '{active_provider}' - no API key mapping found"
            )

        api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError(
                f"API key '{env_key}' not found in environment variables "
                f"for provider '{active_provider}'"
            )

        return api_key

    def get_config_dict(self) -> dict[str, Any]:
        """Get the full configuration dictionary for websocket server.

        Returns:
            The complete configuration dictionary.
        """
        return self._config

    def get_llm_config(self) -> dict[str, Any]:
        """Get active LLM provider configuration from YAML.

        Returns:
            Active LLM provider configuration dictionary.
        """
        llm_config = self._config.get("llm", {})
        active_provider = llm_config.get("active", "openai")
        providers = llm_config.get("providers", {})

        if active_provider not in providers:
            raise ValueError(
                f"Active provider '{active_provider}' not found in providers config"
            )

        return providers[active_provider]

    def get_full_llm_config(self) -> dict[str, Any]:
        """Get full LLM configuration including active provider and all providers.

        Returns:
            Full LLM configuration dictionary.
        """
        return self._config.get("llm", {})

    def get_websocket_config(self) -> dict[str, Any]:
        """Get WebSocket configuration from YAML.

        Returns:
            WebSocket configuration dictionary.
        """
        return self._config.get("chat", {}).get("websocket", {})

    def get_logging_config(self) -> dict[str, Any]:
        """Get logging configuration from YAML.

        Returns:
            Logging configuration dictionary.
        """
        return self._config.get("logging", {})

    def get_chat_service_config(self) -> dict[str, Any]:
        """Get chat service configuration from YAML.

        Returns:
            Chat service configuration dictionary.
        """
        return self._config.get("chat", {}).get("service", {})
