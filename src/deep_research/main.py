"""CLI entry point: `uv run research "your question"`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv

from .tools import TOOL_DEFINITIONS, run_tool

QUALITY_CRITERIA: list[dict[str, str]] = [
    {
        "name": "completeness",
        "type": "boolean",
        "explanation": "is the user's question fully answered?",
        "rejection": "user's question is not fully answered",
    },
    {
        "name": "comprehensiveness",
        "type": "boolean",
        "explanation": (
            "are the retrieved context comprehensive enough to make a convincing "
            "answer?"
        ),
        "rejection": "does not collect enough data before making the answer",
    },
    {
        "name": "conciseness",
        "type": "boolean",
        "explanation": "is the final answer concise enough?",
        "rejection": "answer not concise to understand",
    },
]

SYSTEM_PROMPT = """\
You are a deep research agent. Answer the user's question with a compact
research loop.

Use the tools only when they help:
- brave_search(query): find candidate sources.
- fetch_page(urls): read one or more sources before relying on them.
- query_user(question): ask the user for missing context, confirmation, or
  preferences that would steer the search.

Research rules:
- Start with search unless the answer is already obvious.
- Fetch the most useful sources before citing them.
- Cross-check important claims when possible.
- Use query_user only when the user can resolve ambiguity or provide direction
  that search cannot reliably infer. Ask concise, specific questions.
- Stop searching once you have enough evidence.

Final answer:
- Write a clear markdown report that directly answers the question.
- Cite important claims inline with source URLs.
- Separate well-supported facts from uncertainty or inference.
- End with a Sources list of URLs used.
"""

MAX_OUTPUT_TOKENS = 3000
MAX_LOOP_ITERS = 50
FETCH_PAGE_PRINT_CHARS = 1200
QUALITY_CHECK_MAX_OUTPUT_TOKENS = 1000


@dataclass(frozen=True)
class ApiSettings:
    base_url: str
    api_key: str
    model: str
    bearer_auth: bool


def _messages_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1/messages"):
        return base
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def _load_api_settings() -> ApiSettings:
    triton_key = os.environ.get("TRITONAI_API_KEY")
    anthropic_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    api_key = triton_key or anthropic_token or anthropic_key
    model = os.environ.get("TRITONAI_MODEL") or os.environ.get("ANTHROPIC_MODEL")
    base_url = (
        os.environ.get("TRITONAI_BASE_URL")
        or os.environ.get("ANTHROPIC_BASE_URL")
        or "https://api.anthropic.com"
    )

    missing = []
    if not api_key:
        missing.append("TRITONAI_API_KEY, ANTHROPIC_AUTH_TOKEN, or ANTHROPIC_API_KEY")
    if not model:
        missing.append("TRITONAI_MODEL or ANTHROPIC_MODEL")
    if missing:
        raise RuntimeError("Missing required environment value(s): " + ", ".join(missing))

    return ApiSettings(
        base_url=base_url,
        api_key=api_key,
        model=model,
        bearer_auth=bool(triton_key or anthropic_token),
    )


def _headers(settings: ApiSettings) -> dict[str, str]:
    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if settings.bearer_auth:
        headers["authorization"] = f"Bearer {settings.api_key}"
    else:
        headers["x-api-key"] = settings.api_key
    return headers


async def _call_model(
    client: httpx.AsyncClient,
    settings: ApiSettings,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "tools": TOOL_DEFINITIONS,
    }

    resp = await client.post(
        _messages_url(settings.base_url),
        headers=_headers(settings),
        json=payload,
        timeout=90,
    )

    if resp.status_code >= 400:
        raise RuntimeError(f"Model API error {resp.status_code}: {resp.text[:1000]}")

    return resp.json()


def _quality_check_system_prompt() -> str:
    return (
        "You are a quality check agent. Evaluate whether a candidate final "
        "answer satisfies each criterion. Return only a JSON array of booleans "
        "in the same order as the criteria, such as [true, false, true]. Do "
        "not return prose, labels, markdown, or explanations."
    )


async def _call_quality_check_agent(
    client: httpx.AsyncClient,
    settings: ApiSettings,
    question: str,
    final_result: str,
    tool_usage_history: list[dict[str, Any]],
) -> dict[str, Any]:
    quality_input = {
        "user_question": question,
        "previous_tool_call_history": tool_usage_history,
        "deep_research_intended_output": final_result,
        "criteria": QUALITY_CRITERIA,
    }
    payload: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": QUALITY_CHECK_MAX_OUTPUT_TOKENS,
        "system": _quality_check_system_prompt(),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(quality_input, ensure_ascii=False),
                    }
                ],
            }
        ],
    }

    resp = await client.post(
        _messages_url(settings.base_url),
        headers=_headers(settings),
        json=payload,
        timeout=90,
    )

    if resp.status_code >= 400:
        raise RuntimeError(
            f"Quality check API error {resp.status_code}: {resp.text[:1000]}"
        )

    return resp.json()


def _content_text(block: dict[str, Any]) -> str:
    text = block.get("text")
    return text if isinstance(text, str) else ""


def _usage_value(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return 0


def _format_k_tokens(tokens: int) -> str:
    return f"{tokens / 1_000:.3f}K"


def _response_text(response: dict[str, Any]) -> str:
    content = response.get("content", [])
    if not isinstance(content, list):
        raise RuntimeError("Model response did not include a content list.")
    return "\n".join(
        _content_text(block).strip()
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


def _validate_quality_check_flags(values: Any, source: str) -> list[bool]:
    if (
        isinstance(values, list)
        and len(values) == len(QUALITY_CRITERIA)
        and all(isinstance(value, bool) for value in values)
    ):
        return values

    raise RuntimeError(
        f"Quality check agent returned invalid criterion flags: {source}"
    )


def _parse_quality_check_flags(text: str) -> list[bool]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

    try:
        values = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(
                f"Quality check agent did not return a JSON boolean list: {text}"
            ) from None
        values = json.loads(raw[start : end + 1])

    return _validate_quality_check_flags(values, text)


def _failed_quality_criteria(flags: list[bool]) -> list[dict[str, str]]:
    return [
        criterion
        for criterion, passed in zip(QUALITY_CRITERIA, flags)
        if not passed
    ]


def _format_quality_rejection(failed_criteria: list[dict[str, str]]) -> str:
    failures = ", ".join(
        f"{criterion['name']}: {criterion['rejection']}"
        for criterion in failed_criteria
    )
    return f"output rejected for not satisfying: [{failures}]"


async def _run_quality_check(
    client: httpx.AsyncClient,
    settings: ApiSettings,
    question: str,
    answer: str,
    tool_usage_history: list[dict[str, Any]],
) -> tuple[list[bool], int, int]:
    input_tokens = 0
    output_tokens = 0

    quality_response = await _call_quality_check_agent(
        client,
        settings,
        question,
        answer,
        tool_usage_history,
    )
    quality_usage = quality_response.get("usage", {})
    if isinstance(quality_usage, dict):
        input_tokens += _usage_value(
            quality_usage, "input_tokens", "prompt_tokens"
        )
        output_tokens += _usage_value(
            quality_usage, "output_tokens", "completion_tokens"
        )

    try:
        quality_flags = _parse_quality_check_flags(_response_text(quality_response))
    except RuntimeError:
        print(
            "[deep-research] raw quality check response: "
            f"{json.dumps(quality_response, ensure_ascii=False)}",
            flush=True,
        )
        raise

    return quality_flags, input_tokens, output_tokens


def _tool_observation_preview(name: str, result: str) -> str:
    if name != "fetch_page" or len(result) <= FETCH_PAGE_PRINT_CHARS:
        return result
    return result[:FETCH_PAGE_PRINT_CHARS].rstrip() + "\n...[fetch_page truncated]"


def _print_done(
    tool_counts: dict[str, int],
    llm_input_tokens: int,
    llm_output_tokens: int,
) -> None:
    tool_summary = ", ".join(
        f"{name}={count}" for name, count in sorted(tool_counts.items())
    )
    print("\n[deep-research] done.")
    print(f"[deep-research] tool calls: {tool_summary}")
    print(
        "[deep-research] LLM tokens: "
        f"input={_format_k_tokens(llm_input_tokens)}, "
        f"output={_format_k_tokens(llm_output_tokens)}"
    )


async def run_research(question: str) -> None:
    settings = _load_api_settings()
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
        for _ in range(MAX_LOOP_ITERS):
            response = await _call_model(client, settings, messages)
            usage = response.get("usage", {})
            if isinstance(usage, dict):
                llm_input_tokens += _usage_value(
                    usage, "input_tokens", "prompt_tokens"
                )
                llm_output_tokens += _usage_value(
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
                _content_text(block).strip()
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
                    question,
                    answer,
                    tool_usage_history,
                )
                llm_input_tokens += quality_input_tokens
                llm_output_tokens += quality_output_tokens
                failed_criteria = _failed_quality_criteria(quality_flags)
                if failed_criteria:
                    rejection = _format_quality_rejection(failed_criteria)
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
                print(answer, flush=True)
                _print_done(tool_counts, llm_input_tokens, llm_output_tokens)
                return

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
                observation = _tool_observation_preview(name, result)
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

    raise RuntimeError(f"Research loop exceeded {MAX_LOOP_ITERS} iterations.")


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
