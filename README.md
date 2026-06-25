# Deep Research Agent

A minimal deep research agent with one direct model loop and two plain Python
tools:

- `brave_search(query)` finds candidate sources with Brave Search.
- `fetch_page(url)` fetches readable page text before the agent cites a source.

The runtime is just direct model calls plus local Python functions.

## How It Works

```text
question
  -> direct /v1/messages call
  -> model requests brave_search or fetch_page when needed
  -> Python runs the requested tool and returns the observation
  -> loop repeats until the model writes the final cited report
```

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

The CLI prints each tool action, a compact observation preview, the final
markdown report, and a small usage summary.
