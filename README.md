# Deep Research Agent

A minimal deep research agent with one direct model loop and three plain Python
tools:

- `brave_search(query)` finds candidate sources with Brave Search.
- `fetch_page(urls)` fetches readable page text for one or more URLs in
  parallel before the agent cites sources.
- `query_user(question)` asks the user a clarifying question in the terminal.

The runtime is just direct model calls plus local Python functions.

## How It Works

```text
question
  -> direct /v1/messages call
  -> model requests brave_search, fetch_page, or query_user when needed
  -> Python runs the requested tool and returns the observation
  -> quality check agent accepts or rejects the candidate final answer
  -> loop repeats until the model writes the final cited report
```

The loop exits only after the quality check accepts the candidate answer for
completeness, comprehensiveness, and conciseness.

## Setup

```sh
uv sync
```

Create a `.env` file:

```ini
TRITONAI_BASE_URL=...
TRITONAI_API_KEY=...
TRITONAI_MODEL=...
BRAVE_API_KEY=...
```

You can also use Anthropic-compatible environment names:

```ini
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=...
BRAVE_API_KEY=...
```

## Run

```sh
uv run research "What are the trade-offs between ReAct and plan-and-execute agent architectures?"
```

The CLI prints each tool action, a compact observation preview, any
`query_user` prompts, the final markdown report, and a small usage summary.
