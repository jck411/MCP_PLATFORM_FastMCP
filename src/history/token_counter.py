"""
Token counting utilities for chat events.

This module provides accurate token counting using tiktoken and caching
to prevent re-computation of tokens for the same content.
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Any

import tiktoken
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Cache for token counts - maps content hash to token count
_token_cache: dict[str, int] = {}

# Default model encoding (OpenAI's default tokenizer)
DEFAULT_ENCODING = "cl100k_base"  # Used by GPT-4, GPT-3.5-turbo


@lru_cache(maxsize=32)  # Cache encodings since they're expensive to create
def _get_encoding(encoding_name: str) -> tiktoken.Encoding:
    """Get tiktoken encoding with caching."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(
            f"Failed to get encoding '{encoding_name}', falling back to "
            f"default: {e}"
        )
        return tiktoken.get_encoding(DEFAULT_ENCODING)


class TokenCounter:
    """
    Manages token counting with caching and multiple encoding support.
    """

    def __init__(self, encoding_name: str = DEFAULT_ENCODING):
        self.encoding_name = encoding_name
        self._encoding = _get_encoding(encoding_name)

    def compute_content_hash(self, content: str) -> str:
        """Compute a hash of the content for caching."""
        return hashlib.sha256(
            f"{self.encoding_name}:{content}".encode()
        ).hexdigest()[:16]

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text with caching.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens in the text
        """
        if not text:
            return 0

        # Check cache first
        content_hash = self.compute_content_hash(text)
        if content_hash in _token_cache:
            return _token_cache[content_hash]

        try:
            # Count tokens using tiktoken
            tokens = len(self._encoding.encode(text))

            # Cache the result
            _token_cache[content_hash] = tokens

            return tokens
        except Exception as e:
            logger.warning(f"Token counting failed, using fallback: {e}")
            # Fallback to word-based estimation
            fallback_count = max(1, int(len(text.split()) / 0.75))
            _token_cache[content_hash] = fallback_count
            return fallback_count

    def count_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        """
        Count tokens for a list of OpenAI-style messages.

        This accounts for the message formatting overhead that OpenAI models use.
        Based on: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        Args:
            messages: List of message dictionaries

        Returns:
            Total token count including message formatting
        """
        # Every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_message = 3
        tokens_per_name = 1  # if there's a name, the role is omitted

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += self.count_tokens(value)
                elif key == "tool_calls" and isinstance(value, list):
                    # Handle tool calls
                    for tool_call in value:
                        if isinstance(tool_call, dict):
                            for _tool_key, tool_value in tool_call.items():
                                if isinstance(tool_value, str):
                                    num_tokens += self.count_tokens(tool_value)
                                elif isinstance(tool_value, dict):
                                    # Handle function calls
                                    for _func_key, func_value in tool_value.items():
                                        if isinstance(func_value, str):
                                            num_tokens += self.count_tokens(func_value)
                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def get_cache_stats(self) -> dict[str, int]:
        """Get statistics about the token cache."""
        return {
            "cache_size": len(_token_cache),
            "total_cached_tokens": sum(_token_cache.values()),
        }

    def clear_cache(self) -> None:
        """Clear the token cache."""
        _token_cache.clear()


# Global default counter instance
_default_counter = TokenCounter()


def estimate_tokens(text: str, encoding: str = DEFAULT_ENCODING) -> int:
    """
    Estimate tokens in text using tiktoken with caching.

    This replaces the stub implementation with proper token counting.

    Args:
        text: The text to count tokens for
        encoding: The encoding to use (defaults to cl100k_base)

    Returns:
        Number of tokens in the text
    """
    if encoding != _default_counter.encoding_name:
        # Create a new counter for different encoding
        counter = TokenCounter(encoding)
        return counter.count_tokens(text)

    return _default_counter.count_tokens(text)


def count_conversation_tokens(messages: list[dict[str, Any]]) -> int:
    """
    Count tokens for an entire conversation.

    Args:
        messages: List of OpenAI-style message dictionaries

    Returns:
        Total token count for the conversation
    """
    return _default_counter.count_messages_tokens(messages)


def get_token_cache_stats() -> dict[str, int]:
    """Get token cache statistics."""
    return _default_counter.get_cache_stats()


def clear_token_cache() -> None:
    """Clear the global token cache."""
    _default_counter.clear_cache()


class TokenizedContent(BaseModel):
    """Represents content with its pre-computed token count."""

    content: str
    token_count: int
    encoding: str = DEFAULT_ENCODING
    content_hash: str

    @classmethod
    def create(cls, content: str, encoding: str = DEFAULT_ENCODING) -> TokenizedContent:
        """Create a TokenizedContent with computed token count."""
        counter = TokenCounter(encoding)
        token_count = counter.count_tokens(content)
        content_hash = counter.compute_content_hash(content)

        return cls(
            content=content,
            token_count=token_count,
            encoding=encoding,
            content_hash=content_hash,
        )

    def verify_count(self) -> bool:
        """Verify that the stored token count is still accurate."""
        counter = TokenCounter(self.encoding)
        current_hash = counter.compute_content_hash(self.content)
        return current_hash == self.content_hash
