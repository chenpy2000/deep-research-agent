"""CLI entry point: `uv run research "your question"`."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

import httpx
from dotenv import load_dotenv

from .agent import (
    QualityCheckAgent,
    QuestionBuilderAgent,
    ResearchAgent,
    SummaryAgent,
)
from .tools import TOOL_DEFINITIONS, run_tool
from .utils import (
    ApiSettings,
    content_text,
    load_api_settings,
    print_context_window_usage,
    print_done,
    response_text,
    tool_observation_preview,
    usage_value,
)

MAX_LOOP_ITERS = 50
FETCH_PAGE_PRINT_CHARS = 1200


def _question_builder_messages(
    old_question: str,
    final_answer: str,
    follow_up: str,
) -> list[dict[str, Any]]:
    builder_input = {
        "old_question": old_question,
        "approved_final_answer": final_answer,
        "user_new_prompt": follow_up,
    }
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(builder_input, ensure_ascii=False),
                }
            ],
        }
    ]


def _format_research_question(question: str, comments: str) -> str:
    if not comments:
        return question
    return f"{question}\n\nComments:\n{comments}"


def _follow_up_message(question: str, comments: str) -> dict[str, Any]:
    text = (
        "The user's follow-up has been rewritten as the next research target. "
        "Steer the next turn to this new question, using the comments as "
        "context.\n\n"
        f"Question:\n{question}"
    )
    if comments:
        text += f"\n\nComments:\n{comments}"

    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
    }


async def _read_follow_up() -> str:
    print(
        "\n[deep-research] Press Enter to finish, or type a follow-up to continue.",
        flush=True,
    )
    try:
        follow_up = await asyncio.to_thread(input, "Follow-up: ")
    except EOFError:
        return ""
    return follow_up.strip()


def _quality_check_messages(
    question: str,
    final_result: str,
    tool_usage_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    quality_input = {
        "user_question": question,
        "previous_tool_call_history": tool_usage_history,
        "deep_research_intended_output": final_result,
        "criteria": QualityCheckAgent.criteria,
    }
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(quality_input, ensure_ascii=False),
                }
            ],
        }
    ]


def _summary_messages(
    question: str,
    final_result: str,
) -> list[dict[str, Any]]:
    summary_input = {
        "user_question": question,
        "approved_detailed_answer": final_result,
    }
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(summary_input, ensure_ascii=False),
                }
            ],
        }
    ]


async def _run_quality_check(
    client: httpx.AsyncClient,
    settings: ApiSettings,
    question: str,
    answer: str,
    tool_usage_history: list[dict[str, Any]],
) -> tuple[list[bool], int, int]:
    input_tokens = 0
    output_tokens = 0

    quality_response = await QualityCheckAgent.call(
        client,
        settings,
        _quality_check_messages(question, answer, tool_usage_history),
    )
    quality_usage = quality_response.get("usage", {})
    if not isinstance(quality_usage, dict):
        quality_usage = {}
    print_context_window_usage(settings, QualityCheckAgent.name, quality_usage)
    input_tokens += usage_value(
        quality_usage, "input_tokens", "prompt_tokens"
    )
    output_tokens += usage_value(
        quality_usage, "output_tokens", "completion_tokens"
    )

    try:
        quality_flags = QualityCheckAgent.parse_flags(
            QualityCheckAgent.output_text(quality_response)
        )
    except RuntimeError:
        print(
            "[deep-research] raw quality check response: "
            f"{json.dumps(quality_response, ensure_ascii=False)}",
            flush=True,
        )
        raise

    return quality_flags, input_tokens, output_tokens


async def _run_summary_agent(
    client: httpx.AsyncClient,
    settings: ApiSettings,
    question: str,
    answer: str,
) -> tuple[str, int, int]:
    summary_response = await SummaryAgent.call(
        client,
        settings,
        _summary_messages(question, answer),
    )
    summary_usage = summary_response.get("usage", {})
    if not isinstance(summary_usage, dict):
        summary_usage = {}
    print_context_window_usage(settings, SummaryAgent.name, summary_usage)
    input_tokens = usage_value(summary_usage, "input_tokens", "prompt_tokens")
    output_tokens = usage_value(summary_usage, "output_tokens", "completion_tokens")

    summary = response_text(summary_response)
    if not summary:
        raise RuntimeError("Summary agent returned an empty TL;DR section.")

    return summary, input_tokens, output_tokens


async def _run_question_builder(
    client: httpx.AsyncClient,
    settings: ApiSettings,
    old_question: str,
    final_answer: str,
    follow_up: str,
) -> tuple[str, str, int, int]:
    builder_response = await QuestionBuilderAgent.call(
        client,
        settings,
        _question_builder_messages(old_question, final_answer, follow_up),
    )
    builder_usage = builder_response.get("usage", {})
    if not isinstance(builder_usage, dict):
        builder_usage = {}
    print_context_window_usage(settings, QuestionBuilderAgent.name, builder_usage)
    input_tokens = usage_value(builder_usage, "input_tokens", "prompt_tokens")
    output_tokens = usage_value(builder_usage, "output_tokens", "completion_tokens")

    question, comments = QuestionBuilderAgent.parse_result(
        response_text(builder_response)
    )
    return question, comments, input_tokens, output_tokens


async def run_research(question: str) -> None:
    settings = load_api_settings()
    review_question = question
    llm_input_tokens = 0
    llm_output_tokens = 0
    tool_counts = {
        str(tool.get("name")): 0
        for tool in TOOL_DEFINITIONS
        if isinstance(tool.get("name"), str)
    }
    tool_usage_history: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": question}]}
    ]

    print(f"[deep-research] question: {question}\n")

    async with httpx.AsyncClient() as client:
        loop_iters = 0
        while True:
            loop_iters += 1
            if loop_iters > MAX_LOOP_ITERS:
                raise RuntimeError(
                    f"Research loop exceeded {MAX_LOOP_ITERS} iterations."
                )

            response = await ResearchAgent.call(client, settings, messages)
            usage = response.get("usage", {})
            if not isinstance(usage, dict):
                usage = {}
            print_context_window_usage(settings, ResearchAgent.name, usage)
            llm_input_tokens += usage_value(
                usage, "input_tokens", "prompt_tokens"
            )
            llm_output_tokens += usage_value(
                usage, "output_tokens", "completion_tokens"
            )

            content = response.get("content", [])
            if not isinstance(content, list):
                raise RuntimeError("Model response did not include a content list.")

            tool_uses = [
                block
                for block in content
                if isinstance(block, dict) and block.get("type") == "tool_use"
            ]
            answer = "\n".join(
                content_text(block).strip()
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip()

            if not tool_uses and not answer:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Your previous response did not call a tool and "
                                    "did not provide an answer. Please either call a "
                                    "tool or provide a final answer."
                                ),
                            }
                        ],
                    }
                )
                continue

            messages.append({"role": "assistant", "content": content})

            if not tool_uses:
                (
                    quality_flags,
                    quality_input_tokens,
                    quality_output_tokens,
                ) = await _run_quality_check(
                    client,
                    settings,
                    review_question,
                    answer,
                    tool_usage_history,
                )
                llm_input_tokens += quality_input_tokens
                llm_output_tokens += quality_output_tokens
                failed_criteria = QualityCheckAgent.failed_criteria(quality_flags)
                if failed_criteria:
                    rejection = QualityCheckAgent.format_rejection(failed_criteria)
                    print(
                        f"\n[deep-research] {rejection}\n"
                        f"*rejected answer: {answer}\n",
                        flush=True,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": rejection}],
                        }
                    )
                    continue

                print("\n[deep-research] quality check: passed\n", flush=True)
                (
                    tldr,
                    summary_input_tokens,
                    summary_output_tokens,
                ) = await _run_summary_agent(
                    client,
                    settings,
                    review_question,
                    answer,
                )
                llm_input_tokens += summary_input_tokens
                llm_output_tokens += summary_output_tokens
                print(tldr, flush=True)
                print("\n---\n", flush=True)
                print(answer, flush=True)
                follow_up = await _read_follow_up()
                if not follow_up:
                    print_done(tool_counts, llm_input_tokens, llm_output_tokens)
                    return

                (
                    new_question,
                    new_comments,
                    builder_input_tokens,
                    builder_output_tokens,
                ) = await _run_question_builder(
                    client,
                    settings,
                    review_question,
                    answer,
                    follow_up,
                )
                llm_input_tokens += builder_input_tokens
                llm_output_tokens += builder_output_tokens
                review_question = _format_research_question(
                    new_question,
                    new_comments,
                )
                messages.append(_follow_up_message(new_question, new_comments))
                loop_iters = 0
                print(
                    "\n[deep-research] new research question:\n"
                    f"{review_question}\n",
                    flush=True,
                )
                print("\n[deep-research] continuing with follow-up...\n", flush=True)
                continue

            tool_results: list[dict[str, Any]] = []
            for tool_use in tool_uses:
                tool_id = str(tool_use.get("id", ""))
                name = str(tool_use.get("name", ""))
                args = tool_use.get("input") or {}
                if not isinstance(args, dict):
                    args = {}

                print(f"\n>> Action: {name}({args})", flush=True)
                tool_usage_history.append({"name": name, "arguments": args})
                result, is_error = await run_tool(name, args)
                tool_counts[name] = tool_counts.get(name, 0) + 1
                observation = tool_observation_preview(
                    name,
                    result,
                    FETCH_PAGE_PRINT_CHARS,
                )
                print(f"\nObservation:\n{observation}\n", flush=True)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result,
                        "is_error": is_error,
                    }
                )

            messages.append({"role": "user", "content": tool_results})


def cli() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="research",
        description="Minimal deep research agent with a direct model loop and Brave Search.",
    )
    parser.add_argument("question", help="The research question, in quotes.")
    args = parser.parse_args()

    try:
        asyncio.run(run_research(args.question))
    except KeyboardInterrupt:
        sys.exit(130)
    except RuntimeError as exc:
        print(f"[deep-research] error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
