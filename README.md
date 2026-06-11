# Deep Research Agent (MVP)

A minimal deep research agent built on the **Claude Agent SDK** (Python), using a
**ReAct** (Reason + Act) baseline and **Brave Search** as the search engine. The
LLM is served through your **TritonAI** gateway (any Anthropic-compatible
Messages API endpoint). The environment is managed with **uv**.

## How it works

```
question ──> Agent SDK loop (LLM via TritonAI gateway)
                 │  ReAct system prompt: Thought -> Action -> Observation
                 ├── mcp__research__brave_search  (Brave Search API)
                 └── mcp__research__fetch_page    (read a source URL)
             ──> cited markdown report
```

- `src/deep_research/tools.py` — `brave_search` and `fetch_page` as in-process
  MCP tools (`@tool` + `create_sdk_mcp_server`).
- `src/deep_research/agent.py` — ReAct system prompt and `ClaudeAgentOptions`,
  including mapping `TRITONAI_BASE_URL` / `TRITONAI_API_KEY` onto
  `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN`.
- `src/deep_research/main.py` — CLI that streams Thought/Action/Observation
  steps and the final report.

## Prerequisites

- Python 3.10+ and [uv](https://docs.astral.sh/uv/)
- No Node.js needed — `claude-agent-sdk` ≥ 0.2.x ships a bundled Claude Code binary.
- A TritonAI API key (the gateway must expose an Anthropic-compatible Messages API,
  including tool use)
- A [Brave Search API](https://api-dashboard.search.brave.com/) key (free tier works)

## Setup

```sh
uv sync
cp .env.example .env   # then fill in your keys
```

`.env`:

```ini
TRITONAI_BASE_URL=...   # your gateway base URL
TRITONAI_API_KEY=...
TRITONAI_MODEL=...      # model id served by the gateway (optional)
BRAVE_API_KEY=...
```

If `TRITONAI_*` is left unset, the agent falls back to your normal
`ANTHROPIC_API_KEY` credentials so you can smoke-test against Anthropic directly.

## Run

```sh
uv run research "What are the trade-offs between ReAct and plan-and-execute agent architectures?"
```

You'll see the ReAct trace stream by (`>> Action: mcp__research__brave_search(...)`)
followed by a final markdown report with inline citations and a Sources list.

## Next steps (beyond MVP)

- Parallel sub-queries / multi-agent fan-out
- Structured report output (JSON schema) and saving reports to disk
- Better page extraction (e.g. `trafilatura`) and result caching
- Evaluation harness comparing ReAct vs. other baselines
