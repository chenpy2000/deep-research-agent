"""Plain Python tools used by the research loop."""

from __future__ import annotations

import html
import os
import re
from typing import Any

import httpx

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "brave_search",
        "description": (
            "Search the web with Brave Search. Use this to find sources, facts, "
            "and current information. Returns numbered results with title, URL, "
            "and snippet."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "fetch_page",
        "description": (
            "Fetch one web page by URL and return its readable text content. "
            "Use this to read a source before citing it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
]


def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    return re.sub(r"\s+", " ", raw).strip()


async def brave_search(query: str) -> tuple[str, bool]:
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return "BRAVE_API_KEY is not set in the environment.", True

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                BRAVE_ENDPOINT,
                params={"q": query, "count": 6},
                headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            )
    except httpx.HTTPError as exc:
        return f"Brave Search request failed: {exc}", True

    if resp.status_code != 200:
        return f"Brave Search API error {resp.status_code}: {resp.text[:300]}", True

    results = resp.json().get("web", {}).get("results", [])
    if not results:
        return f"No results for query: {query!r}", False

    lines = []
    for i, result in enumerate(results, 1):
        snippet = _strip_html(result.get("description") or "")
        lines.append(
            f"{i}. {result.get('title', '(no title)')}\n"
            f"   URL: {result.get('url')}\n"
            f"   {snippet}"
        )
    return "\n\n".join(lines), False


async def fetch_page(url: str) -> tuple[str, bool]:
    try:
        async with httpx.AsyncClient(
            timeout=25,
            follow_redirects=True,
            headers={"User-Agent": "deep-research-agent/0.1"},
        ) as client:
            resp = await client.get(url)
    except httpx.HTTPError as exc:
        return f"Failed to fetch {url}: {exc}", True

    if resp.status_code != 200:
        return f"HTTP {resp.status_code} when fetching {url}", True

    content_type = resp.headers.get("content-type", "")
    if "html" in content_type:
        text = _strip_html(resp.text)
    elif any(kind in content_type for kind in ("text", "json", "xml")):
        text = resp.text
    else:
        return f"Fetched {url}, but content type is not readable text: {content_type}", True

    return f"Content of {url}:\n\n{text}", False


async def run_tool(name: str, args: dict[str, Any]) -> tuple[str, bool]:
    if name == "brave_search":
        query = str(args.get("query", "")).strip()
        if not query:
            return "brave_search requires a non-empty query.", True
        return await brave_search(query)

    if name == "fetch_page":
        url = str(args.get("url", "")).strip()
        if not url:
            return "fetch_page requires a non-empty url.", True
        result, is_error = await fetch_page(url)
        return result, is_error

    return f"Unknown tool: {name}", True
