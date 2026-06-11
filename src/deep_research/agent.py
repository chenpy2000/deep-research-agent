"""Agent configuration: ReAct system prompt + TritonAI gateway wiring."""

from __future__ import annotations

import os

from claude_agent_sdk import ClaudeAgentOptions

from .tools import research_server

REACT_SYSTEM_PROMPT = """\
You are a deep research agent. Answer the user's research question by running
an explicit ReAct (Reason + Act) loop.

Each step has three parts:
1. Thought: state what you currently know, what is still missing, and what to do next.
2. Action: call exactly one tool —
   - brave_search(query): find candidate sources
   - fetch_page(url): read one promising source in full
3. Observation: summarize what the tool result actually shows before the next Thought.

Research rules:
- Start broad with brave_search, then fetch_page the 2-4 most promising sources.
- Cross-check important claims against at least two independent sources.
- If a search returns nothing useful, reformulate the query rather than giving up.
- Stop researching once you have enough evidence; do not exceed ~8 tool calls.

Final answer (after the loop ends):
- A markdown report that directly answers the question.
- Cite every key claim inline with its source URL.
- Clearly separate well-supported facts from your own inference or uncertainty.
- End with a "Sources" list of all URLs used.

Use only the two tools above. Do not use any other tools.
"""

ALLOWED_TOOLS = [
    "mcp__research__brave_search",
    "mcp__research__fetch_page",
]


def build_options() -> ClaudeAgentOptions:
    """Build agent options, routing the LLM through the TritonAI gateway.

    TRITONAI_BASE_URL / TRITONAI_API_KEY are mapped onto the Anthropic env vars
    the Agent SDK understands. If they are unset, the SDK falls back to your
    regular ANTHROPIC_* credentials so the agent still runs against Anthropic.
    """
    env: dict[str, str] = {}
    base_url = os.environ.get("TRITONAI_BASE_URL")
    api_key = os.environ.get("TRITONAI_API_KEY")
    if base_url:
        env["ANTHROPIC_BASE_URL"] = base_url
    if api_key:
        env["ANTHROPIC_AUTH_TOKEN"] = api_key

    return ClaudeAgentOptions(
        system_prompt=REACT_SYSTEM_PROMPT,
        model=os.environ.get("TRITONAI_MODEL"),  # None -> SDK default
        mcp_servers={"research": research_server},
        allowed_tools=ALLOWED_TOOLS,
        permission_mode="bypassPermissions",  # headless: never prompt for approval
        setting_sources=[],  # don't load user/project Claude settings
        max_turns=20,
        env=env,
    )
