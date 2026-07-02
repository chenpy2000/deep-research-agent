"""Model-facing agent definitions used by the research workflow."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import httpx

from .tools import TOOL_DEFINITIONS
from .utils import ApiSettings, headers, messages_url, response_text


class ModelAgent:
    name: ClassVar[str]
    system_prompt: ClassVar[str]
    max_output_tokens: ClassVar[int]
    api_error_label: ClassVar[str] = "Model"
    tools: ClassVar[list[dict[str, Any]] | None] = None

    @classmethod
    def payload(
        cls,
        settings: ApiSettings,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": settings.model,
            "max_tokens": cls.max_output_tokens,
            "system": cls.system_prompt,
            "messages": messages,
        }
        if cls.tools is not None:
            payload["tools"] = cls.tools
        return payload

    @classmethod
    async def call(
        cls,
        client: httpx.AsyncClient,
        settings: ApiSettings,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        resp = await client.post(
            messages_url(settings.base_url),
            headers=headers(settings),
            json=cls.payload(settings, messages),
            timeout=90,
        )

        if resp.status_code >= 400:
            raise RuntimeError(
                f"{cls.api_error_label} API error "
                f"{resp.status_code}: {resp.text[:1000]}"
            )

        return resp.json()


class ResearchAgent(ModelAgent):
    name = "research agent"
    api_error_label = "Model"
    max_output_tokens = 3000
    tools = TOOL_DEFINITIONS
    system_prompt = """\
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


class QuestionBuilderAgent(ModelAgent):
    name = "question builder"
    api_error_label = "Question builder"
    max_output_tokens = 800
    system_prompt = (
        "You are a question building agent. Based on the old user question, "
        "the approved final answer, and the user's new comment or question, "
        "build the next standalone research question that best reflects what "
        "the user wants to ask now. Include brief comments with context, "
        "constraints, or corrections that should steer the next research turn. "
        "Return only a JSON object with string fields: question and comments."
    )

    @classmethod
    def parse_result(cls, text: str) -> tuple[str, str]:
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise RuntimeError(
                    "Question builder did not return a JSON object: "
                    f"{text}"
                ) from None
            value = json.loads(raw[start : end + 1])

        if not isinstance(value, dict):
            raise RuntimeError(
                f"Question builder returned invalid JSON: {text}"
            )

        question = value.get("question")
        comments = value.get("comments", "")
        if not isinstance(question, str) or not question.strip():
            raise RuntimeError(
                f"Question builder returned an empty question: {text}"
            )
        if not isinstance(comments, str):
            raise RuntimeError(
                f"Question builder returned invalid comments: {text}"
            )

        return question.strip(), comments.strip()


class QualityCheckAgent(ModelAgent):
    name = "quality check"
    api_error_label = "Quality check"
    max_output_tokens = 1000
    criteria: ClassVar[list[dict[str, str]]] = [
        {
            "name": "answers_user_question",
            "type": "boolean",
            "explanation": "does it answer the user's question?",
            "rejection": "does not answer the user's question",
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
    ]
    system_prompt = (
        "You are a quality check agent. Evaluate whether a candidate final "
        "answer satisfies each criterion. Return only a JSON array of booleans "
        "in the same order as the criteria, such as [true, false]. Do "
        "not return prose, labels, markdown, or explanations."
    )

    @classmethod
    def output_text(cls, response: dict[str, Any]) -> str:
        text = response_text(response)
        if text:
            return text

        content = response.get("content", [])
        if not isinstance(content, list):
            return ""
        return "\n".join(
            str(block.get("thinking", "")).strip()
            for block in content
            if isinstance(block, dict) and block.get("type") == "thinking"
        ).strip()

    @classmethod
    def validate_flags(cls, values: Any, source: str) -> list[bool]:
        if (
            isinstance(values, list)
            and len(values) == len(cls.criteria)
            and all(isinstance(value, bool) for value in values)
        ):
            return values

        raise RuntimeError(
            f"Quality check agent returned invalid criterion flags: {source}"
        )

    @classmethod
    def parse_flags(cls, text: str) -> list[bool]:
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
                    "Quality check agent did not return a JSON boolean list: "
                    f"{text}"
                ) from None
            values = json.loads(raw[start : end + 1])

        return cls.validate_flags(values, text)

    @classmethod
    def failed_criteria(cls, flags: list[bool]) -> list[dict[str, str]]:
        return [
            criterion
            for criterion, passed in zip(cls.criteria, flags)
            if not passed
        ]

    @classmethod
    def format_rejection(cls, failed_criteria: list[dict[str, str]]) -> str:
        failures = ", ".join(
            f"{criterion['name']}: {criterion['rejection']}"
            for criterion in failed_criteria
        )
        return f"output rejected for not satisfying: [{failures}]"


class SummaryAgent(ModelAgent):
    name = "summary agent"
    api_error_label = "Summary agent"
    max_output_tokens = 600
    system_prompt = (
        "You are a summary agent. Given the user's original question and an "
        "approved detailed answer, write only a markdown TL;DR section. Start "
        "with '## TL;DR'. First give the most straightforward short answer to "
        "the question, then briefly summarize the key supporting points. Do "
        "not add facts that are not supported by the detailed answer. Do not "
        "include any other section."
    )
