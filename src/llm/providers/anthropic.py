# src/llm/providers/anthropic.py
from __future__ import annotations

import json
from typing import Any

from ..base import ProviderAdapter

JSON = dict[str, Any]
MsgList = list[dict[str, Any]]
ToolList = list[dict[str, Any]] | None

def _system_prompt(msgs: MsgList) -> str | None:
    for m in msgs:
        if m["role"] == "system":
            return m["content"]
    return None

def _to_anthropic_msgs(msgs: MsgList) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in msgs:
        role = m["role"]
        if role == "system":
            continue
        if role == "tool":
            # Convert OpenAI tool format to Anthropic tool_result format
            # Tool results should be in user messages with tool_result content blocks
            tool_result = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m["content"]
            }
            # Check if last message is a user message and append to it
            if out and out[-1]["role"] == "user":
                if isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(tool_result)
                else:
                    # Convert string content to list and add tool_result
                    out[-1]["content"] = [
                        {"type": "text", "text": out[-1]["content"]},
                        tool_result
                    ]
            else:
                # Create new user message with tool_result
                out.append({
                    "role": "user",
                    "content": [tool_result]
                })
        elif role == "assistant":
            # Handle assistant messages with potential tool_calls
            content_blocks = []
            if text_content := m.get("content"):
                content_blocks.append({"type": "text", "text": text_content})

            if tool_calls := m.get("tool_calls"):
                for call in tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "input": json.loads(call["function"]["arguments"])
                    })

            out.append({
                "role": "assistant",
                "content": content_blocks if content_blocks else ""
            })
        else:
            # Regular user message - use simple string format for text-only messages
            out.append({"role": role, "content": m["content"]})
    return out

def _to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    res = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t["function"]
        res.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn["parameters"],
            }
        )
    return res

class AnthropicAdapter(ProviderAdapter):
    def build_request(
        self, messages: MsgList, tools: ToolList
    ) -> tuple[str, dict[str, str], dict[str, str], JSON]:
        payload: JSON = {
            "model": self._model(),
            "messages": _to_anthropic_msgs(messages),
            "max_tokens": self._max_tokens(),
            "temperature": self._temperature(),
        }
        sys_prompt = _system_prompt(messages)
        if sys_prompt:
            payload["system"] = sys_prompt
        if tools:
            payload["tools"] = _to_anthropic_tools(tools)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        return ("/messages", {}, headers, payload)

    def parse_response(self, data: JSON) -> JSON:
        text_parts, tool_calls = [], []

        for item in data["content"]:
            if item["type"] == "text":
                text_content = item["text"]
                text_parts.append(text_content)

            elif item["type"] == "tool_use":
                tool_calls.append(
                    {
                        "id": item["id"],
                        "type": "function",
                        "function": {
                            "name": item["name"],
                            "arguments": json.dumps(item["input"]),
                        },
                    }
                )

        usage = data.get("usage", {})
        normalized_usage = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": (
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            ),
        }
        stop_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        stop_reason = data.get("stop_reason")
        finish = stop_map.get(stop_reason) if stop_reason else None

        return {
            "message": {
                "content": "\n".join(text_parts),
                "tool_calls": tool_calls or None,
                "finish_reason": finish,
            },
            "finish_reason": finish,
            "usage": normalized_usage,
            "model": data.get("model", self._model()),
        }
