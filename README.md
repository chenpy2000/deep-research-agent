# deep-research-agent

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
  -> loop repeats on rejected answers
  -> summary agent writes a TL;DR from the accepted final answer
  -> CLI prints the TL;DR and final cited report
```

The loop exits only after the quality check accepts the candidate answer for
completeness and comprehensiveness. After that, a summary agent prints a TL;DR
section before the detailed report.

## Setup

```sh
uv sync
```

Create a `.env` file:

```ini
TRITONAI_BASE_URL=...
TRITONAI_API_KEY=...
TRITONAI_MODEL=...
TRITONAI_CONTEXT_WINDOW=
BRAVE_API_KEY=...
```

You can also use Anthropic-compatible environment names:

```ini
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=...
ANTHROPIC_CONTEXT_WINDOW=
BRAVE_API_KEY=...
```

Set `TRITONAI_CONTEXT_WINDOW` or `ANTHROPIC_CONTEXT_WINDOW` to the model's
context window in tokens, such as `128000` or `128k`, to print per-call context
window usage for each LLM generation. Leave it blank to disable this reporting.

## Run

```sh
uv run research "What are the trade-offs between ReAct and plan-and-execute agent architectures?"
```

The CLI prints each tool action, a compact observation preview, any
`query_user` prompts, a TL;DR section, the final markdown report, and a small
usage summary.
