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
  -> quality check agent audits correctness, directness, and comprehensiveness
     and may call tools while auditing
  -> loop repeats on rejected answers
  -> summary agent writes a TL;DR from the accepted final answer
  -> CLI prints the TL;DR and final cited report
  -> user presses Enter to finish, or types a follow-up to continue researching
  -> question builder rewrites the follow-up into the next research question
```

The loop exits only after the quality check accepts the candidate answer for
correctness, directness, and comprehensiveness. The quality check agent returns
structured metric flags and LLM-generated rejection reasons; if any flag fails,
the generated rejection is returned to the research loop so the model can search
or revise again. After all flags pass, a summary agent prints a TL;DR section
before the detailed report. The CLI then waits for user input: pressing Enter
ends the program, while a follow-up comment or question is passed to a question
builder agent. That agent uses the previous target question, approved answer,
and new user prompt to create the next research question plus comments. The
rewritten question is appended to the conversation and becomes the target
question used by the quality check agent.

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
usage summary after you accept the final answer by pressing Enter.
