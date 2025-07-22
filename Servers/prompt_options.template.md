"""
┌──────────────────────────────────────────────────────────────┐
│  MCP PROMPT TEMPLATE – copy‑paste and customise as needed.   │
└──────────────────────────────────────────────────────────────┘
"""

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base as p  # Message/PromptArgument helpers

###############################################################################
# 0.  CREATE THE SERVER INSTANCE - REQUIRED
###############################################################################
mcp = FastMCP(
    name="My Prompt Server",                 # REQUIRED: Displayed in dev‑tools & logs
    instructions="(Optional) server‑wide note for other humans"  # OPTIONAL
)

###############################################################################
# 1.  DEFINE A PROMPT WITH REQUIRED AND OPTIONAL FIELDS
###############################################################################
@mcp.prompt(
    # ───────── REQUIRED FIELDS ─────────
    name="rhyme_joker",                      # REQUIRED: Stable machine‑ID used in prompts/get
    title="Rhyming Jokester",                # REQUIRED: Shown in dropdowns / slash‑command list
    description=(                            # REQUIRED: Describes prompt functionality
        "Tells a short rhyming joke and refuses to help with anything else."
        # Appears in tooltips or a side panel when a user inspects the prompt
    ),

    # ───────── OPTIONAL FIELDS ─────────
    arguments=[                              # OPTIONAL: Define expected input parameters
        p.PromptArgument(
            name="topic",
            description="Subject of the joke (optional)",
            required=False,                  # Client marks field as optional/required
            type="string"                    # *SDK extension*: lets UIs do validation
        ),
        p.PromptArgument(
            name="style",
            description="Rhyme scheme (aa/bb etc.)",
            required=False,
            type="choice"                    # Could drive a dropdown in richer UIs
        )
    ],

    # ───────── OPTIONAL METADATA ─────────
    meta={                                   # OPTIONAL: Additional metadata
        "category": "Fun",                   # UIs can group by category tag
        "author": "Jack",                    # Useful for multi‑author servers
        "version": "1.0.0",                  # Lets clients surface update notices
        "icon": "🤡"                         # Emoji/icon many desktop shells show
    }
)
def rhyme_joker(topic: str = "anything", style: str | None = None) -> list[p.Message]:
    """
    REQUIRED: The decorated function must return one of:
      • A single str  → becomes a *user* message, *or*
      • A list[p.Message] → explicit role‑tagged messages (system/user/assistant)
    """
    # OPTIONAL: Build system message that sets persona
    scheme_hint = f" Use {style} rhyme scheme." if style else ""
    system_text = (
        f"You ONLY respond with a rhyming joke about {topic}. "
        f"Politely refuse any other request.{scheme_hint}"
    )

    # OPTIONAL: Pre-greeting message
    return [
        p.SystemMessage(system_text),
        p.AssistantMessage("Ready to rhyme!")  # Optional "greeting" pre‑message
    ]

###############################################################################
# 2.  ADD MORE PROMPTS BELOW IF NEEDED…
###############################################################################
