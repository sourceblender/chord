"""Stored Chat Completions (S13e, Phase 3): `store: true` on POST
/v1/chat/completions keeps the completion, and the five stored operations read
it back: list (model and metadata filters, after/limit/order), retrieve,
update metadata, delete, and the request's messages.

Stored in the Responses SQLite file (responses_store.py), 30-day floor, real
deletes. Only the Chat door stores; the Responses door has its own store. A
streamed completion is stored as the completion object its chunks add up to.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .responses_store import ResponseStore


def _error(status: int, message: str, code: str | None, param: str | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error", "param": param, "code": code}},
                        status_code=status)


def metadata_error(metadata) -> str | None:
    if metadata is None:
        return None
    if not (isinstance(metadata, dict) and len(metadata) <= 16 and all(
            isinstance(k, str) and len(k) <= 64 and isinstance(v, str) and len(v) <= 512 for k, v in metadata.items())):
        return "metadata must be at most 16 string pairs (keys <= 64, values <= 512 chars)"
    return None


def stored_messages(completion_id: str, messages: list[dict]) -> list[dict]:
    """The request's messages as ChatCompletionMessageList items: string content
    as content, a parts array as content_parts (text joined into content)."""
    out = []
    for i, m in enumerate(messages):
        content = m.get("content")
        parts = content if isinstance(content, list) else None
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text") if parts else content
        item = {"id": f"{completion_id}-{i}", "role": m.get("role"), "content": text if isinstance(text, str) else None,
                "refusal": None, "content_parts": [p for p in parts if isinstance(p, dict) and p.get("type") in ("text", "image_url")] if parts else None}
        if m.get("name") is not None:
            item["name"] = m["name"]
        if m.get("tool_calls"):
            item["tool_calls"] = m["tool_calls"]
        out.append(item)
    return out


def completion_from_chunks(chunks: list[dict]) -> dict | None:
    """The chat.completion object a stream's chunks add up to: the same fields the
    non-stream store keeps -- text, refusal, tool calls, the legacy function_call,
    annotations, logprobs, finish, usage, service_tier and system_fingerprint
    (review 2026-09-24 B15). The chunks are already wire-shaped."""
    if not chunks:
        return None
    first = chunks[0]
    text, refusal, calls, finish, usage, tier = "", None, {}, None, None, None
    fingerprint, function_call, annotations, logprobs = None, None, [], None
    for chunk in chunks:
        usage = chunk.get("usage") or usage
        tier = chunk.get("service_tier") or tier
        if chunk.get("system_fingerprint") is not None:
            fingerprint = chunk["system_fingerprint"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text += delta.get("content") or ""
            if delta.get("refusal"):
                refusal = (refusal or "") + delta["refusal"]
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                slot["id"] = slot["id"] or tc.get("id") or ""
                fn = tc.get("function") or {}
                slot["function"]["name"] += fn.get("name") or ""
                slot["function"]["arguments"] += fn.get("arguments") or ""
            if isinstance(delta.get("function_call"), dict):
                function_call = function_call or {"name": "", "arguments": ""}
                function_call["name"] += delta["function_call"].get("name") or ""
                function_call["arguments"] += delta["function_call"].get("arguments") or ""
            if isinstance(delta.get("annotations"), list):
                annotations += delta["annotations"]
            if isinstance(choice.get("logprobs"), dict):
                logprobs = logprobs or {"content": None, "refusal": None}
                for key in ("content", "refusal"):
                    if isinstance(choice["logprobs"].get(key), list):
                        logprobs[key] = (logprobs[key] or []) + choice["logprobs"][key]
            finish = choice.get("finish_reason") or finish
    message = {"role": "assistant", "content": text or None, "refusal": refusal}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    if function_call:
        message["function_call"] = function_call
    if annotations:
        message["annotations"] = annotations
    completion = {"id": first["id"], "object": "chat.completion", "created": first["created"], "model": first["model"],
                  "choices": [{"index": 0, "message": message, "logprobs": logprobs, "finish_reason": finish or "stop"}]}
    if usage:
        completion["usage"] = usage
    if tier:
        completion["service_tier"] = tier
    if fingerprint is not None:
        completion["system_fingerprint"] = fingerprint
    return completion


def register(app: FastAPI, store: ResponseStore) -> None:
    def not_found(completion_id: str) -> JSONResponse:
        return _error(404, f"Chat completion with id '{completion_id}' not found.", None, None)

    def paging(request: Request, default_order: str = "asc",
               ) -> tuple[int, str, str | None] | JSONResponse:
        q = request.query_params
        try:
            limit = int(q.get("limit") or 20)
        except ValueError:
            limit = 0
        if not 1 <= limit <= 100:
            return _error(400, "limit must be between 1 and 100", "invalid_value", "limit")
        order = q.get("order") or default_order
        if order not in ("asc", "desc"):
            return _error(400, "order must be asc or desc", "invalid_value", "order")
        return limit, order, q.get("after")

    def page(items: list[dict], limit: int, order: str, after: str | None) -> JSONResponse:
        # Items arrive from the store already in `order`, paged by `after`, and
        # trimmed to `limit + 1` (so `has_more` is exact). The route that built
        # `items` is responsible for cursor validation; `page` is dumb on
        # purpose (review 2026-09-23).
        chunk = items[:limit]
        return JSONResponse({"object": "list", "data": chunk, "first_id": chunk[0]["id"] if chunk else None,
                             "last_id": chunk[-1]["id"] if chunk else None, "has_more": len(items) > limit})

    def with_metadata(completion: dict, metadata: dict) -> dict:
        """The stored object as the pinned CreateChatCompletionResponse carries it: `metadata`
        is one of its properties. It was kept and filterable but never returned, so an update
        answered 200 with no sign of what it had changed (the S13e gate, 2026-09-17)."""
        return {**completion, "metadata": metadata}

    @app.get("/v1/chat/completions")
    async def list_completions(request: Request):
        opts = paging(request)
        if isinstance(opts, JSONResponse):
            return opts
        limit, order, after = opts
        # An empty `after=` query string is `""` here, not None. Normalize so
        # the store sees no cursor and the route treats `after=` like omitting
        # the parameter, which is what callers expect. Reject explicitly empty
        # whitespace too: a non-empty string of spaces is not a valid cursor
        # and the store would 400 anyway (Copilot review 2026-09-23,
        # third round).
        if after is not None:
            after = after.strip() or None
        q = request.query_params
        wanted = {k[len("metadata["):-1]: v for k, v in q.multi_items() if k.startswith("metadata[") and k.endswith("]")}
        model = q.get("model")
        # Pull limit+1 so `has_more` is exact without a second COUNT(*) query
        # (review 2026-09-23). One extra row of work beats a second pass over
        # the index every time.
        rows, ok = store.list_chat(after_id=after, limit=limit + 1, order=order,
                                   model=model, metadata=wanted or None)
        if after and not ok:
            return _error(400, f"no item '{after}' in this list", "invalid_value", "after")
        items = [with_metadata(c, meta) for c, meta in rows]
        return page(items, limit, order, after)

    @app.get("/v1/chat/completions/{completion_id}")
    async def retrieve_completion(completion_id: str):
        found = store.get_chat(completion_id)
        return JSONResponse(with_metadata(found[0], found[1])) if found else not_found(completion_id)

    @app.post("/v1/chat/completions/{completion_id}")
    async def update_completion(completion_id: str, request: Request):
        if store.get_chat(completion_id) is None:
            return not_found(completion_id)
        try:
            body = await request.json()
        except (ValueError, RecursionError):
            return _error(400, "body is not JSON", "invalid_json")
        if not isinstance(body, dict) or "metadata" not in body:
            return _error(400, "metadata is required", "missing_required_parameter", "metadata")
        unknown = sorted(set(body) - {"metadata"})
        if unknown:
            return _error(400, f"Unrecognized request argument supplied: {unknown[0]}", "unknown_parameter", unknown[0])
        if (bad := metadata_error(body["metadata"])) is not None:
            return _error(400, bad, "invalid_value", "metadata")
        store.update_chat_metadata(completion_id, body["metadata"] or {})
        updated = store.get_chat(completion_id)
        if updated is None:
            # Deleted (or aged past retention) between the two reads: the answer
            # is 404, never a TypeError on None (review 2026-09-22).
            return not_found(completion_id)
        return JSONResponse(with_metadata(updated[0], updated[1]))

    @app.delete("/v1/chat/completions/{completion_id}")
    async def delete_completion(completion_id: str):
        if not store.delete_chat(completion_id):
            return not_found(completion_id)
        return JSONResponse({"object": "chat.completion.deleted", "id": completion_id, "deleted": True})

    @app.get("/v1/chat/completions/{completion_id}/messages")
    async def completion_messages(completion_id: str, request: Request):
        found = store.get_chat(completion_id)
        if found is None:
            return not_found(completion_id)
        opts = paging(request)
        if isinstance(opts, JSONResponse):
            return opts
        limit, order, after = opts
        if after is not None:
            after = after.strip() or None
        items = stored_messages(completion_id, found[2])
        # The messages endpoint reads the whole list (it is bounded by the
        # request's `messages` length, not by 30-day retention), so cursor
        # and order are applied here. Cursor existence is a local id lookup;
        # the order/after walk is in Python because the data is already
        # resident (review 2026-09-23, Copilot high-severity fix).
        if after and after not in {i["id"] for i in items}:
            return _error(400, f"no item '{after}' in this list", "invalid_value", "after")
        ordered = list(reversed(items)) if order == "desc" else items
        if after:
            ids = [i["id"] for i in ordered]
            ordered = ordered[ids.index(after) + 1:]
        # Ask for limit + 1 so `has_more` is exact without a second query.
        page_items = ordered[:limit + 1]
        return page(page_items, limit, order, after)

