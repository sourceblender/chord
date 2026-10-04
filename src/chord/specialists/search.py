"""Web search (#81): one query, what the search backend returns, an answer from it.

2026-09-13: "Model says it cant search the web either." It couldn't: there
was no search capability. This one writes a short query from the conversation,
asks Brave's API (keyed; DuckDuckGo's HTML page when no key is set or Brave
fails), and hands the results to the voice node as untrusted text to answer
from. It reads result snippets only, never whole pages, and never follows
anything the results say.
"""
from __future__ import annotations

import asyncio
import html
import re
import time
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from ..contract import Job, Outcome, Result
from ..registry import load
from ..web_search import parse_options
from . import SpecialistContext, specialist

MAX_RESULTS = 5
TIMEOUT_S = 10.0     # per HTTP operation
DEADLINE_S = 25.0    # the whole backend stage, fallback included (#95): a drip-fed body can't hold the turn
QUERY_SYSTEM = ("Write ONE web search query for the user's last message: the key words a search engine "
                "needs, at most 12 words. Reply with the query only, no quotes, no explanation.")


@dataclass
class Hit:
    title: str
    url: str
    snippet: str


def _text(s: str) -> str:
    """HTML fragment to plain text on one line."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()


def _web_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


async def brave(query: str, key: str, client: httpx.AsyncClient, *, max_results: int = MAX_RESULTS,
                country: str | None = None) -> list[Hit]:
    params = {"q": query, "count": max_results, "extra_snippets": "true"}
    if country:
        params["country"] = country
    r = await client.get("https://api.search.brave.com/res/v1/web/search",
                         params=params,
                         headers={"X-Subscription-Token": key, "Accept": "application/json"})
    if r.status_code != 200:
        raise httpx.HTTPStatusError(f"status {r.status_code}", request=r.request, response=r)
    data = r.json()
    web = data.get("web") if isinstance(data, dict) else None
    results = web.get("results") if isinstance(web, dict) else None
    if not isinstance(results, list):
        raise ValueError("unexpected Brave response shape")  # [] / null / {"web": null}: fall back (#95)
    hits = []
    for x in results:
        if not isinstance(x, dict) or not isinstance(x.get("url"), str) or not _web_url(x["url"]):
            continue
        extra_raw = x.get("extra_snippets")
        extra = extra_raw if isinstance(extra_raw, list) else []
        parts = [x.get("description"), *extra[:2]]
        snippet = " … ".join(_text(p) for p in parts if isinstance(p, str) and p)
        title = x.get("title")
        hits.append(Hit(_text(title) if isinstance(title, str) else "", x["url"], snippet[:600]))
    return hits[:max_results]


_DDG_LINK = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_DDG_SNIPPET = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.S)


async def duckduckgo(query: str, client: httpx.AsyncClient, *, max_results: int = MAX_RESULTS) -> list[Hit]:
    # A GET returned a 202 anti-bot page; the form POST returned results (2026-09-13).
    r = await client.post("https://html.duckduckgo.com/html/", data={"q": query},
                          headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                                 "(KHTML, like Gecko) Chrome/126 Safari/537.36"})
    if r.status_code != 200:  # 202 is its anti-bot page ("bots use DuckDuckGo too")
        raise httpx.HTTPStatusError(f"status {r.status_code}", request=r.request, response=r)
    links = list(_DDG_LINK.finditer(r.text))
    hits = []
    for i, m in enumerate(links):
        href, title = m.group(1), m.group(2)
        # A snippet belongs to a result only if it sits before the NEXT result's
        # link; a result with no snippet must not borrow its neighbour's (#95).
        end = links[i + 1].start() if i + 1 < len(links) else len(r.text)
        own = _DDG_SNIPPET.search(r.text, m.end(), end)
        if href.startswith("//"):
            href = "https:" + href
        target = parse_qs(urlparse(href).query).get("uddg", [None])[0]
        url = unquote(target) if target else href
        if "duckduckgo.com/y.js" in url or not _web_url(url):  # an ad or an unsafe/non-web target
            continue
        hits.append(Hit(_text(title), url, _text(own.group(1)) if own else ""))
    return hits[:max_results]


async def search(query: str, brave_key: str, transport: httpx.AsyncBaseTransport | None = None,
                 *, max_results: int = MAX_RESULTS, country: str | None = None,
                 ) -> tuple[str, list[Hit], list[str]]:
    """(backend used, hits, errors). Brave first when keyed; DuckDuckGo otherwise
    or when Brave fails or finds nothing."""
    errors = []
    # No redirects: httpx drops Authorization across origins but not Brave's
    # X-Subscription-Token (#95). A 3xx is an error like any other non-200.
    async with httpx.AsyncClient(timeout=TIMEOUT_S, follow_redirects=False, transport=transport) as client:
        if brave_key:
            try:
                hits = await brave(query, brave_key, client, max_results=max_results, country=country)
                if hits:
                    return "brave", hits, errors
                errors.append("brave: no results")
            except (httpx.HTTPError, ValueError) as exc:
                errors.append(f"brave: {_why(exc)}")
        try:
            return "duckduckgo", await duckduckgo(query, client, max_results=max_results), errors
        except httpx.HTTPError as exc:
            errors.append(f"duckduckgo: {_why(exc)}")
            return "none", [], errors


def _why(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


def sources_text(hits: list[Hit]) -> str:
    return "\n".join(f"[{i}] {h.title} ({urlparse(h.url).netloc}) {h.url}\n    {h.snippet}"
                     for i, h in enumerate(hits, 1))


@specialist("search")
async def run(job: Job, ctx: SpecialistContext) -> Result:
    capability = load()["search"]
    last_user = next((m["text"] for m in reversed(job.conversation) if m["role"] == "user"), job.intent)
    started = time.monotonic()
    query, query_error = "", None
    # Who writes the query. A version 2 config names it: the helper, with the helper's
    # thinking switch, whatever the registry record says. Version 1 and env-only
    # installs keep the registry's model, as before.
    try:  # bounded like the router; on any failure the user's own words are the query (#95)
        if ctx.settings.config_version == 2:
            from ..graph import router_client  # deferred: graph imports the specialists
            writer = router_client(ctx.model(ctx.settings.router_model), ctx.settings.router_thinking_mode)
        else:
            writer = ctx.model(capability.model)
        reply = await asyncio.wait_for(writer.ainvoke(
            [SystemMessage(QUERY_SYSTEM), HumanMessage(f"Last message: {last_user}\nWhat is wanted: {job.intent}")]),
            ctx.settings.router_timeout_s)
        query = _text(reply.content if isinstance(reply.content, str) else "")[:200].strip("\"' ")
    except Exception as exc:  # cancellation (BaseException) still propagates
        query_error = "timed out" if isinstance(exc, TimeoutError) else type(exc).__name__
    query = query or _text(last_user)[:200]
    options = parse_options(job.web_search_options or {})
    query = options.localized_query(query)
    ctx.progress("searching")
    try:
        backend, hits, errors = await asyncio.wait_for(
            search(query, ctx.settings.brave_api_key, max_results=options.max_results, country=options.country),
            DEADLINE_S,
        )
    except TimeoutError:  # cancellation (BaseException) still propagates
        backend, hits, errors = "none", [], [f"search stage timed out after {DEADLINE_S:g} s"]
    ctx.trace.tool_calls.append({"tool": "web.search", "backend": backend, "query": query, "errors": errors,
                                 "query_model_error": query_error,
                                 "results": len(hits), "urls": [h.url for h in hits],
                                 "search_context_size": options.context_size,
                                 "country": options.country,
                                 "ms": round((time.monotonic() - started) * 1000)})
    if not hits:
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.failed,
                      summary="The web search didn't return any results this time.")
    sources = [{"id": i, "title": h.title, "url": h.url} for i, h in enumerate(hits, 1)]
    return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed,
                  summary=sources_text(hits),
                  provenance={"kind": "search", "query": query, "backend": backend,
                              "urls": [h.url for h in hits], "sources": sources})
