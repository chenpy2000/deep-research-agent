"""CLI entry point: `uv run research "your question"`."""

from __future__ import annotations

import argparse
import asyncio
import sys

from claude_agent_sdk import AssistantMessage, ResultMessage, query
from dotenv import load_dotenv

from .agent import build_options


async def run_research(question: str) -> None:
    options = build_options()
    print(f"[deep-research] question: {question}\n")

    async for message in query(prompt=question, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                block_type = getattr(block, "type", None) or type(block).__name__
                if hasattr(block, "text"):  # TextBlock — Thought / Observation / answer
                    print(block.text, flush=True)
                elif hasattr(block, "name"):  # ToolUseBlock — Action
                    print(f"\n>> Action: {block.name}({block.input})\n", flush=True)
                else:
                    print(f"[{block_type}]", flush=True)

        elif isinstance(message, ResultMessage):
            print("\n" + "=" * 60)
            if message.subtype == "success":
                print("[deep-research] done.")
            else:
                print(f"[deep-research] finished with status: {message.subtype}")
            usage = message.usage or {}
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_write = usage.get("cache_creation_input_tokens", 0)
            print(f"  turns: {message.num_turns}")
            print(
                f"  tokens: input={input_tokens}  output={output_tokens}  "
                f"cache_read={cache_read}  cache_write={cache_write}"
            )


def cli() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="research",
        description="Minimal deep research agent (Claude Agent SDK + ReAct + Brave Search).",
    )
    parser.add_argument("question", help="The research question, in quotes.")
    args = parser.parse_args()

    try:
        asyncio.run(run_research(args.question))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    cli()
