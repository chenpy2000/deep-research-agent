"""Shared helpers for configuration, model responses, and CLI output."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ApiSettings:
    base_url: str
    api_key: str
    model: str
    bearer_auth: bool
    context_window_tokens: int | None


def messages_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1/messages"):
        return base
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def parse_context_window_tokens(env_name: str, value: str) -> int:
    token_text = value.strip().lower().replace(",", "").replace("_", "")
    multiplier = 1
    if token_text.endswith("k"):
        multiplier = 1_000
        token_text = token_text[:-1]
    elif token_text.endswith("m"):
        multiplier = 1_000_000
        token_text = token_text[:-1]

    if not token_text.isdigit():
        raise RuntimeError(
            f"{env_name} must be a positive token count, or blank to disable."
        )

    tokens = int(token_text) * multiplier
    if tokens <= 0:
        raise RuntimeError(
            f"{env_name} must be a positive token count, or blank to disable."
        )
    return tokens


def context_window_from_env(name: str) -> int | None:
    value = optional_env(name)
    if value is None:
        return None
    return parse_context_window_tokens(name, value)


def load_api_settings() -> ApiSettings:
    triton_key = optional_env("TRITONAI_API_KEY")
    anthropic_token = optional_env("ANTHROPIC_AUTH_TOKEN")
    anthropic_key = optional_env("ANTHROPIC_API_KEY")
    triton_model = optional_env("TRITONAI_MODEL")
    anthropic_model = optional_env("ANTHROPIC_MODEL")

    api_key = triton_key or anthropic_token or anthropic_key
    model = triton_model or anthropic_model
    base_url = (
        optional_env("TRITONAI_BASE_URL")
        or optional_env("ANTHROPIC_BASE_URL")
        or "https://api.anthropic.com"
    )
    context_window_tokens = None
    if triton_model:
        context_window_tokens = context_window_from_env("TRITONAI_CONTEXT_WINDOW")
    elif anthropic_model:
        context_window_tokens = context_window_from_env("ANTHROPIC_CONTEXT_WINDOW")

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
        context_window_tokens=context_window_tokens,
    )


def headers(settings: ApiSettings) -> dict[str, str]:
    request_headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if settings.bearer_auth:
        request_headers["authorization"] = f"Bearer {settings.api_key}"
    else:
        request_headers["x-api-key"] = settings.api_key
    return request_headers


def content_text(block: dict[str, Any]) -> str:
    text = block.get("text")
    return text if isinstance(text, str) else ""


def response_text(response: dict[str, Any]) -> str:
    content = response.get("content", [])
    if not isinstance(content, list):
        raise RuntimeError("Model response did not include a content list.")
    return "\n".join(
        content_text(block).strip()
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


def usage_value(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return 0


def format_k_tokens(tokens: int) -> str:
    return f"{tokens / 1_000:.3f}K"


def print_context_window_usage(
    settings: ApiSettings,
    label: str,
    usage: dict[str, Any],
) -> None:
    if settings.context_window_tokens is None:
        return

    input_tokens = usage_value(usage, "input_tokens", "prompt_tokens")
    output_tokens = usage_value(usage, "output_tokens", "completion_tokens")
    total_tokens = input_tokens + output_tokens
    if total_tokens <= 0:
        print(
            f"[deep-research] context window ({label}): usage unavailable",
            flush=True,
        )
        return

    percent = total_tokens / settings.context_window_tokens * 100
    print(
        "[deep-research] context window "
        f"({label}): {percent:.2f}% used "
        f"({format_k_tokens(total_tokens)} / "
        f"{format_k_tokens(settings.context_window_tokens)} tokens; "
        f"input={format_k_tokens(input_tokens)}, "
        f"output={format_k_tokens(output_tokens)})",
        flush=True,
    )


def tool_observation_preview(name: str, result: str, max_chars: int) -> str:
    if name != "fetch_page" or len(result) <= max_chars:
        return result
    return result[:max_chars].rstrip() + "\n...[fetch_page truncated]"


def print_done(
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
        f"input={format_k_tokens(llm_input_tokens)}, "
        f"output={format_k_tokens(llm_output_tokens)}"
    )
