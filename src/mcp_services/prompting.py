from __future__ import annotations

import logging
from typing import Any

from mcp import types

# Deliberately no imports from provider/LLM code.
# This module is MCP-only and provider-agnostic.

logger = logging.getLogger(__name__)


class MCPResourcePromptService:
    """
    Handles:
      - Cataloging readable MCP resources (availability-aware)
      - Building the final system prompt by embedding resources and prompts

    Dependencies:
      - ToolSchemaManager-like object with:
          list_available_resources() -> list[str]
          read_resource(uri) -> object with .contents
          get_resource_info(uri) -> object with .resource.name (optional)
          list_available_prompts() -> list[str]
          get_prompt_info(name) -> object with .prompt.description (optional)
    """

    def __init__(self, tool_mgr: Any, chat_conf: dict[str, Any]) -> None:
        self.tool_mgr = tool_mgr
        self.chat_conf = chat_conf
        self.resource_catalog: list[str] = []

    async def update_resource_catalog_on_availability(self) -> None:
        """
        Build `self.resource_catalog` with only those resources that currently
        return content.
        """
        if not self.tool_mgr:
            self.resource_catalog = []
            return

        all_resource_uris = self.tool_mgr.list_available_resources()
        available_uris: list[str] = []

        for uri in all_resource_uris:
            try:
                resource_result = await self.tool_mgr.read_resource(uri)
                if getattr(resource_result, "contents", None):
                    available_uris.append(uri)
            except Exception:
                # Skip unavailable resources silently
                continue

        self.resource_catalog = available_uris
        logger.debug(
            "Updated resource catalog: %d of %d resources are available",
            len(available_uris),
            len(all_resource_uris),
        )

    async def get_available_resources(
        self,
    ) -> dict[str, list[types.TextResourceContents | types.BlobResourceContents]]:
        """
        Return a mapping of URI -> list(contents) for resources that are
        currently readable.
        Only URIs in `self.resource_catalog` are considered.
        """
        out: dict[
            str, list[types.TextResourceContents | types.BlobResourceContents]
        ] = {}

        if not self.resource_catalog or not self.tool_mgr:
            return out

        for uri in self.resource_catalog:
            try:
                resource_result = await self.tool_mgr.read_resource(uri)
                if resource_result.contents:
                    out[uri] = resource_result.contents
                    logger.debug("Resource %s is available and loaded", uri)
                else:
                    logger.debug("Resource %s has no content, skipping", uri)
            except Exception as e:
                # Log and skip — do not leak broken resources into the prompt.
                logger.warning(
                    "Resource %s is unavailable and will be excluded from "
                    "system prompt: %s",
                    uri,
                    e,
                )
                continue

        if out:
            logger.info("Including %d available resources in system prompt", len(out))
        else:
            logger.info(
                "No resources are currently available - system prompt will not "
                "include resource section"
            )

        return out

    async def make_system_prompt(self) -> str:
        """
        Build the system prompt by embedding:
          - The base system prompt from chat_conf["system_prompt"]
          - The available resources' content (text lines or binary size)
          - The available prompt catalog (name + description)

        This function is stable even if portions of the MCP surface are missing.
        """
        base = str(self.chat_conf.get("system_prompt", "")).rstrip()

        if not self.tool_mgr:
            return base

        # Include available resources
        available_resources = await self.get_available_resources()
        if available_resources:
            base += "\n\n**Available Resources:**"
            for uri, contents in available_resources.items():
                resource_info = self.tool_mgr.get_resource_info(uri)
                name = (
                    getattr(getattr(resource_info, "resource", None), "name", None)
                    or uri
                )
                base += f"\n\n**{name}** ({uri}):"

                for content in contents:
                    if isinstance(content, types.TextResourceContents):
                        lines = (content.text or "").strip().split("\n")
                        for line in lines:
                            base += f"\n{line}"
                    elif isinstance(content, types.BlobResourceContents):
                        base += f"\n[Binary content: {len(content.blob)} bytes]"
                    else:
                        base += f"\n[{type(content).__name__} available]"

        # Include prompt catalog
        try:
            prompt_names = self.tool_mgr.list_available_prompts()
        except Exception:
            prompt_names = []

        if prompt_names:
            prompt_list: list[str] = []
            for pname in prompt_names:
                try:
                    pinfo = self.tool_mgr.get_prompt_info(pname)
                    desc = getattr(getattr(pinfo, "prompt", None), "description", None)
                    desc = desc or "No description available"
                except Exception:
                    desc = "No description available"
                prompt_list.append(f"• **{pname}**: {desc}")

            prompts_text = "\n".join(prompt_list)
            base += (
                f"\n\n**Available Prompts** (use apply_prompt method):\n"
                f"{prompts_text}"
            )

        return base
