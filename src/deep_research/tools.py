"""Research tools exposed to the agent as an in-process MCP server.

Two tools:
- brave_search: query the Brave Search API for candidate sources
- fetch_page:   download one URL and return its readable text
"""

from __future__ import annotations

import html
import os
import re
from typing import Any

import httpx
from claude_agent_sdk import create_sdk_mcp_server, tool

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
MAX_PAGE_CHARS = 8_000


def _text_result(text: str, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": is_error}


def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    return re.sub(r"\s+", " ", raw).strip()


@tool(
    "brave_search",
    "Search the web with Brave Search. Call this whenever you need to find "
    "sources, facts, or current information. Returns the top results as "
    "numbered entries with title, URL, and snippet.",
    {"query": str},
)
async def brave_search(args: dict[str, Any]) -> dict[str, Any]:
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return _text_result("BRAVE_API_KEY is not set in the environment.", is_error=True)

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                BRAVE_ENDPOINT,
                params={"q": args["query"], "count": 6},
                headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            )
    except httpx.HTTPError as exc:
        return _text_result(f"Brave Search request failed: {exc}", is_error=True)

    if resp.status_code != 200:
        return _text_result(
            f"Brave Search API error {resp.status_code}: {resp.text[:300]}",
            is_error=True,
        )

    results = resp.json().get("web", {}).get("results", [])
    if not results:
        return _text_result(f"No results for query: {args['query']!r}")

    lines = []
    for i, r in enumerate(results, 1):
        snippet = _strip_html(r.get("description") or "")
        lines.append(f"{i}. {r.get('title', '(no title)')}\n   URL: {r.get('url')}\n   {snippet}")
    return _text_result("\n\n".join(lines))


@tool(
    "fetch_page",
    "Fetch one web page by URL and return its readable text content "
    f"(truncated to {MAX_PAGE_CHARS} characters). Use this to read a source "
    "found via brave_search before citing it.",
    {"url": str},
)
async def fetch_page(args: dict[str, Any]) -> dict[str, Any]:
    url = args["url"]
    try:
        async with httpx.AsyncClient(
            timeout=25, follow_redirects=True, headers={"User-Agent": "deep-research-agent/0.1"}
        ) as client:
            resp = await client.get(url)
    except httpx.HTTPError as exc:
        return _text_result(f"Failed to fetch {url}: {exc}", is_error=True)

    if resp.status_code != 200:
        return _text_result(f"HTTP {resp.status_code} when fetching {url}", is_error=True)

    content_type = resp.headers.get("content-type", "")
    if "html" in content_type:
        text = _strip_html(resp.text)
    else:
        text = resp.text

    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + " ...[truncated]"
    return _text_result(f"Content of {url}:\n\n{text}")


research_server = create_sdk_mcp_server(
    name="research",
    version="0.1.0",
    tools=[brave_search, fetch_page],
)
