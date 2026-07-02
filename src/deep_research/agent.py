"""Model-facing agent definitions used by the research workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, ClassVar

import httpx

from .tools import TOOL_DEFINITIONS
from .utils import ApiSettings, headers, messages_url, response_text


@dataclass(frozen=True)
class QualityMetricResult:
    name: str
    passed: bool
    reason: str


@dataclass(frozen=True)
class QualityCheckResult:
    metrics: list[QualityMetricResult]
    flags: list[bool]
    rejection: str


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
    max_output_tokens = 2000
    tools = TOOL_DEFINITIONS
    criteria: ClassVar[list[dict[str, str]]] = [
        {
            "name": "correctness",
            "type": "boolean",
            "explanation": (
                "are important claims supported by reliable/current sources, "
                "is freshness checked when relevant, and does the reasoning "
                "follow from gathered resources?"
            ),
        },
        {
            "name": "answers_user_question",
            "type": "boolean",
            "explanation": "does it answer the user's question?",
        },
        {
            "name": "comprehensiveness",
            "type": "boolean",
            "explanation": (
                "is the retrieved context comprehensive enough to make a convincing "
                "answer?"
            ),
        },
    ]
    system_prompt = """\
You are a quality check agent. Audit a candidate final answer before it is
shown to the user.

You may call tools when they help you verify facts, source reliability,
freshness, reasoning, or coverage:
- brave_search(query): search for current or corroborating sources.
- fetch_page(urls): read source pages before relying on them.
- query_user(question): ask the user only if human clarification is necessary.

Evaluate the metrics in this exact order:
1. correctness: important claims are supported by reliable/current sources,
   freshness is checked when relevant, and reasoning follows from gathered
   resources.
2. answers_user_question: the candidate directly answers the user's question.
3. comprehensiveness: the retrieved context is broad and deep enough for a
   convincing answer.

When you are done auditing, return only this JSON object shape:
{
  "metrics": [
    {"name": "correctness", "passed": true, "reason": "..."},
    {"name": "answers_user_question", "passed": true, "reason": "..."},
    {"name": "comprehensiveness", "passed": true, "reason": "..."}
  ],
  "flags": [true, true, true],
  "rejection": ""
}

The flags list must match the metric passed values. If any metric fails,
write a concise actionable rejection using your generated reasons. Do not
return markdown, labels, or prose outside the JSON object.
"""

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
    def metric_names(cls) -> list[str]:
        return [criterion["name"] for criterion in cls.criteria]

    @classmethod
    def _strip_json_markdown(cls, text: str) -> str:
        raw = text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        return raw

    @classmethod
    def parse_result(cls, text: str) -> QualityCheckResult:
        raw = cls._strip_json_markdown(text)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise RuntimeError(
                    "Quality check agent did not return a JSON object: "
                    f"{text}"
                ) from None
            value = json.loads(raw[start : end + 1])

        return cls.validate_result(value, text)

    @classmethod
    def validate_result(cls, value: Any, source: str) -> QualityCheckResult:
        if not isinstance(value, dict):
            raise RuntimeError(
                f"Quality check agent returned invalid JSON: {source}"
            )

        raw_metrics = value.get("metrics")
        flags = value.get("flags")
        rejection = value.get("rejection", "")
        expected_names = cls.metric_names()

        if not isinstance(raw_metrics, list) or len(raw_metrics) != len(expected_names):
            raise RuntimeError(
                f"Quality check agent returned invalid metrics: {source}"
            )
        if (
            not isinstance(flags, list)
            or len(flags) != len(expected_names)
            or not all(isinstance(flag, bool) for flag in flags)
        ):
            raise RuntimeError(
                f"Quality check agent returned invalid flags: {source}"
            )
        if not isinstance(rejection, str):
            raise RuntimeError(
                f"Quality check agent returned invalid rejection: {source}"
            )

        metrics: list[QualityMetricResult] = []
        for index, raw_metric in enumerate(raw_metrics):
            if not isinstance(raw_metric, dict):
                raise RuntimeError(
                    f"Quality check agent returned invalid metric: {source}"
                )
            name = raw_metric.get("name")
            passed = raw_metric.get("passed")
            reason = raw_metric.get("reason")
            if name != expected_names[index]:
                raise RuntimeError(
                    f"Quality check agent returned metrics out of order: {source}"
                )
            if not isinstance(passed, bool) or not isinstance(reason, str):
                raise RuntimeError(
                    f"Quality check agent returned invalid metric fields: {source}"
                )
            if not reason.strip():
                raise RuntimeError(
                    f"Quality check agent returned an empty metric reason: {source}"
                )
            metrics.append(
                QualityMetricResult(name=name, passed=passed, reason=reason.strip())
            )

        metric_flags = [metric.passed for metric in metrics]
        if flags != metric_flags:
            raise RuntimeError(
                f"Quality check agent returned mismatched flags: {source}"
            )

        return QualityCheckResult(
            metrics=metrics,
            flags=flags,
            rejection=rejection.strip(),
        )

    @classmethod
    def failed_metrics(
        cls,
        result: QualityCheckResult,
    ) -> list[QualityMetricResult]:
        return [metric for metric in result.metrics if not metric.passed]

    @classmethod
    def format_rejection(cls, result: QualityCheckResult) -> str:
        if result.rejection:
            return result.rejection

        failures = "; ".join(
            f"{metric.name}: {metric.reason}"
            for metric in cls.failed_metrics(result)
        )
        return f"Quality check rejected the answer: {failures}"


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
