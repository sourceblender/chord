"""The Responses API (#146, Phase 3a): POST /v1/responses (non-stream and
stream), GET and DELETE /v1/responses/{id}, GET /v1/responses/{id}/input_items,
and multi-turn chaining through `previous_response_id`, in the pinned spec's
shapes.

One core, two doors. A Responses request is translated into the Chat
Completions core (chat_api's run_chat), so validation, the persona base, service
tiers, forced tool calls, the router and the specialists behave exactly as
they do on Chat. What differs is the edge:

- Specialists are offered as the spec's built-in tools. `web_search` lets the
  router send a turn to search; `image_generation` lets it make a picture.
  Without them no specialist runs, as with OpenAI, where a model only calls
  tools it was given. A request that mixes function tools with a built-in is
  refused by name: function tools own the turn (S04), so a built-in beside
  them would be silently dropped, and the rest of the project refuses that
  shape rather than emit a response that echoes a tool the system never ran.
- Output is typed items: `message` (output_text with url_citation
  annotations, which Chat could only stream as a profile extension),
  `function_call`, `web_search_call` and `image_generation_call` (the image as
  base64, never markdown).
- Streaming is the Responses event vocabulary with sequence numbers.
- `store` defaults to true (the spec's default): the response, its input items
  and the conversation so far are kept (responses_store.py) for retrieval,
  deletion, input_items and chaining.
- `background: true` answers at once with a queued Response and runs the turn
  as a task (it needs `store`, and cannot stream). `conversation` continues a
  stored Conversation (never together with `previous_response_id`).

Refused by name, not silently ignored: prompt, max_tool_calls, top_logprobs, context_management, moderation, include values,
truncation "auto", reasoning summaries, tools and tool_choice types Chord
can't honour, input item references and files.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
from pathlib import Path
import hashlib
import logging
import hmac
import json
import os
import secrets
import time
from collections.abc import AsyncGenerator
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from ulid import ULID

from . import graph as graph_mod
from .http_transport import ClosingStreamingResponse
from .registry import PROMPTS_DIR
from .responses_store import DuplicateItemId, ResponseStore

logger = logging.getLogger(__name__)

CREATE_FIELDS = {
    "background", "context_management", "conversation", "include", "input", "instructions", "max_output_tokens",
    "max_tool_calls", "metadata", "model", "moderation", "parallel_tool_calls", "previous_response_id", "prompt",
    "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention", "reasoning", "safety_identifier",
    "service_tier", "store", "stream", "stream_options", "temperature", "text", "tool_choice", "tools",
    "top_logprobs", "top_p", "truncation", "user",
}
COUNT_FIELDS = {"model", "input", "previous_response_id", "tools", "text", "reasoning", "truncation", "instructions",
                "personality", "conversation", "tool_choice", "parallel_tool_calls"}
# Present and not null (or not false/empty) requests unsupported behavior.
REFUSED_WHEN_SET = ("prompt", "max_tool_calls", "context_management", "moderation")
PASSTHROUGH = ("temperature", "top_p", "service_tier", "safety_identifier", "prompt_cache_key",
               "prompt_cache_options", "prompt_cache_retention", "user", "metadata")
WEB_SEARCH_TOOLS = {"web_search", "web_search_preview", "web_search_2025_08_26", "web_search_preview_2025_03_11"}
# Chat-core error params, renamed to the field the Responses caller sent.
PARAM_NAMES = {"max_completion_tokens": "max_output_tokens", "max_tokens": "max_output_tokens",
               "response_format": "text.format", "reasoning_effort": "reasoning.effort", "verbosity": "text.verbosity",
               "messages": "input", "functions": "tools", "function_call": "tool_choice"}
# ImageGenToolCall's optional fields in the pinned spec: None = any string, else its enum.
IMAGE_ITEM_FIELDS: dict[str, tuple[str, ...] | None] = {
    "size": None, "quality": ("low", "medium", "high", "xhigh", "max", "auto"),
    "action": ("generate", "edit", "auto"), "background": ("transparent", "opaque", "auto"),
    "output_format": ("png", "webp", "jpeg"), "revised_prompt": None,
}
IMAGE_MARKER = "[image you generated earlier]"
COMPACT_FIELDS = {"model", "input", "previous_response_id", "instructions", "prompt_cache_key", "prompt_cache_retention",
                  "prompt_cache_options", "service_tier"}
COMPACTION_LEAD = "Summary of the conversation so far (compacted):\n"
_COMPACTION_KEY = b""


def compaction_token(summary: str) -> str:
    """Opaque to the caller and signed, NOT encrypted: tamper-proof, but a caller
    who decodes it can read the summary."""
    payload = base64.urlsafe_b64encode(json.dumps({"v": 1, "summary": summary}).encode()).decode()
    sig = hmac.new(_COMPACTION_KEY, payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def compaction_summary(token) -> str | None:
    if not isinstance(token, str) or "." not in token:
        return None
    payload, sig = token.rsplit(".", 1)
    # Bytes, never strs -- the token is client-supplied, so compare_digest on a
    # non-ASCII sig was a crafted-request 500 (the second site beside
    # artifact_links). errors="replace" on BOTH sides:
    # a lone surrogate from json.loads would make a bare .encode() raise too,
    # and replaced bytes can never equal a legitimate token.
    expected = hmac.new(_COMPACTION_KEY, payload.encode("utf-8", "replace"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig.encode("utf-8", "replace"), expected.encode()):
        return None
    try:
        return json.loads(base64.urlsafe_b64decode(payload.encode()))["summary"]
    except (ValueError, KeyError):
        return None


def _id(prefix: str) -> str:
    return f"{prefix}_{str(ULID()).lower()}"


def _error(status: int, message: str, code: str | None, param: str | None = None,
           kind: str = "invalid_request_error", headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind, "param": param, "code": code}},
                        status_code=status, headers=headers)


class Refusal(Exception):
    def __init__(self, message: str, param: str | None, code: str = "unsupported_value", status: int = 400):
        super().__init__(message)
        self.param, self.code, self.status = param, code, status


# --- request translation -----------------------------------------------------------

def _text_of(parts, param: str) -> str:
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        raise Refusal("content must be a string or an array of content parts", param, "invalid_value")
    texts = []
    for j, part in enumerate(parts):
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in ("output_text", "input_text") and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif kind == "refusal" and isinstance(part.get("refusal"), str):
            texts.append(part["refusal"])
        else:
            raise Refusal(f"unsupported content part {kind!r}", f"{param}[{j}].type")
    return "".join(texts)


def _user_parts(parts, param: str):
    """Responses input content -> Chat content (text, images, audio)."""
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        raise Refusal("content must be a string or an array of content parts", param, "invalid_value")
    out = []
    for j, part in enumerate(parts):
        kind = part.get("type") if isinstance(part, dict) else None
        where = f"{param}[{j}]"
        if kind == "input_text" and isinstance(part.get("text"), str):
            out.append({"type": "text", "text": part["text"]})
        elif kind == "input_image":
            if part.get("file_id") or not isinstance(part.get("image_url"), str):
                raise Refusal("input_image needs an image_url; file ids are not supported", f"{where}.image_url")
            out.append({"type": "image_url", "image_url": {"url": part["image_url"], "detail": part.get("detail") or "auto"}})
        elif kind == "input_audio" and isinstance(part.get("input_audio"), dict):
            out.append({"type": "input_audio", "input_audio": part["input_audio"]})
        else:
            raise Refusal(f"unsupported content part {kind!r}", f"{where}.type")
    if out and all(p["type"] == "text" for p in out):
        return "".join(p["text"] for p in out)   # text-only content is plain text, as a Chat caller would send it
    return out


def input_to_messages(value) -> tuple[list[dict], list[dict]]:
    """(chat messages, input items as stored for input_items with our ids)."""
    if isinstance(value, str):
        return [{"role": "user", "content": value}], [
            {"id": _id("msg"), "type": "message", "role": "user", "status": "completed",
             "content": [{"type": "input_text", "text": value}]}]
    if not isinstance(value, list) or not value:
        raise Refusal("input must be a string or a non-empty array of items", "input", "invalid_value")
    messages: list[dict] = []
    items: list[dict] = []
    for i, item in enumerate(value):
        where = f"input[{i}]"
        if not isinstance(item, dict):
            raise Refusal("each input item must be an object", where, "invalid_value")
        kind = item.get("type") or ("message" if "role" in item else None)
        if kind == "message":
            role = item.get("role")
            if role not in ("user", "assistant", "system", "developer"):
                raise Refusal("role must be user, assistant, system or developer", f"{where}.role", "invalid_value")
            content = item.get("content")
            if role == "assistant":
                text = _text_of(content, f"{where}.content")
                messages.append({"role": "assistant", "content": text})
                stored = [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}]
            else:
                parts = _user_parts(content, f"{where}.content")
                messages.append({"role": role, "content": parts})
                stored = ([{"type": "input_text", "text": parts}] if isinstance(parts, str) else
                          [({"type": "input_text", "text": p["text"]} if p["type"] == "text" else
                            {"type": "input_image", "image_url": p["image_url"]["url"], "detail": p["image_url"]["detail"]}
                            if p["type"] == "image_url" else {"type": "input_audio", "input_audio": p["input_audio"]})
                           for p in parts])
            items.append({"id": item.get("id") or _id("msg"), "type": "message", "role": role,
                          "status": "completed", "content": stored})
        elif kind == "function_call":
            call_id, name, args = item.get("call_id"), item.get("name"), item.get("arguments")
            if not all(isinstance(x, str) for x in (call_id, name, args)):
                raise Refusal("function_call needs string call_id, name and arguments", where, "invalid_value")
            call = {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}
            if messages and messages[-1]["role"] == "assistant" and messages[-1].get("tool_calls") is not None:
                messages[-1]["tool_calls"].append(call)
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            items.append({"id": item.get("id") or _id("fc"), "type": "function_call", "call_id": call_id,
                          "name": name, "arguments": args, "status": "completed"})
        elif kind == "function_call_output":
            call_id, output = item.get("call_id"), item.get("output")
            if not isinstance(call_id, str):
                raise Refusal("function_call_output needs a string call_id", f"{where}.call_id", "invalid_value")
            text = output if isinstance(output, str) else _text_of(output, f"{where}.output")
            messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            items.append({"id": item.get("id") or _id("fco"), "type": "function_call_output", "call_id": call_id,
                          "output": text, "status": "completed"})
        # Both kinds are stored as items too, like every kind the model reads:
        # conversation mode rebuilds context from the stored items only, so an
        # item kept out of `items` was gone from the second turn on while
        # previous_response_id chaining kept it (review 2026-09-24 B4).
        elif kind == "image_generation_call":
            messages.append({"role": "assistant", "content": IMAGE_MARKER})
            status = item.get("status") if item.get("status") in ("in_progress", "completed", "generating", "failed") else "completed"
            stored_image = {"id": item.get("id") or _id("ig"), "type": "image_generation_call", "status": status,
                            "result": item.get("result") if isinstance(item.get("result"), str) else None}
            # The other fields the pinned ImageGenToolCall declares, kept when
            # given and checked against its types and enums (Copilot on #332:
            # they were dropped). null is the spec's "not given".
            for field, allowed in IMAGE_ITEM_FIELDS.items():
                value = item.get(field)
                if value is None:
                    continue
                if not isinstance(value, str) or (allowed is not None and value not in allowed):
                    expected = "a string" if allowed is None else "one of " + ", ".join(allowed)
                    raise Refusal(f"{field} must be {expected}", f"{where}.{field}", "invalid_value")
                stored_image[field] = value
            items.append(stored_image)
        elif kind == "compaction":
            summary = compaction_summary(item.get("encrypted_content"))
            if summary is None:
                raise Refusal("this compaction item was not produced here or was altered", f"{where}.encrypted_content",
                              "invalid_value")
            messages.append({"role": "system", "content": COMPACTION_LEAD + summary})
            items.append({"id": item.get("id") or _id("cmp"), "type": "compaction",
                          "encrypted_content": item["encrypted_content"]})
        elif kind in ("web_search_call", "reasoning"):
            continue   # our own earlier output, replayed by a caller managing state: nothing for the model
        else:
            raise Refusal(f"input item type {kind!r} is not supported", f"{where}.type")
    return messages, items


def translate(body: dict, previous: list[dict] | None) -> tuple[dict, frozenset, dict]:
    """A validated Responses body -> (chat body, offered specialist routes, echo fields)."""
    for key in REFUSED_WHEN_SET:
        if body.get(key) not in (None, False, [], {}):
            raise Refusal(f"{key} is not supported", key, "unsupported_parameter")
    if body.get("top_logprobs") not in (None, 0):
        raise Refusal("top_logprobs is not supported", "top_logprobs", "unsupported_parameter")
    if body.get("include") not in (None, []):
        raise Refusal("include is not supported", "include", "unsupported_parameter")
    if body.get("truncation") not in (None, "disabled"):
        raise Refusal("only truncation 'disabled' is supported", "truncation")
    if "input" not in body:
        raise Refusal("input is required", "input", "missing_required_parameter")
    messages, items = input_to_messages(body["input"])

    chat: dict[str, Any] = {"model": body.get("model")}
    for key in PASSTHROUGH:
        if key in body:
            chat[key] = body[key]
    if body.get("max_output_tokens") is not None:
        chat["max_completion_tokens"] = body["max_output_tokens"]

    reasoning = body.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, dict):
            raise Refusal("reasoning must be an object", "reasoning", "invalid_value")
        if reasoning.get("summary") is not None or reasoning.get("generate_summary") is not None:
            raise Refusal("reasoning summaries are not supported", "reasoning.summary")
        if reasoning.get("effort") is not None:
            chat["reasoning_effort"] = reasoning["effort"]

    text = body.get("text")
    fmt = {"type": "text"}
    verbosity = None
    if text is not None:
        if not isinstance(text, dict):
            raise Refusal("text must be an object", "text", "invalid_value")
        fmt = text.get("format") or {"type": "text"}
        kind = fmt.get("type") if isinstance(fmt, dict) else None
        if kind == "json_object":
            chat["response_format"] = {"type": "json_object"}
        elif kind == "json_schema":
            spec = {k: fmt[k] for k in ("name", "schema", "strict", "description") if k in fmt}
            chat["response_format"] = {"type": "json_schema", "json_schema": spec}
        elif kind != "text":
            raise Refusal("text.format.type must be text, json_object or json_schema", "text.format.type")
        verbosity = text.get("verbosity")
        if verbosity is not None:
            chat["verbosity"] = verbosity

    offered: set[str] = set()
    functions = []
    tools_echo = []
    tools = body.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise Refusal("tools must be an array", "tools", "invalid_value")
        for i, tool in enumerate(tools):
            kind = tool.get("type") if isinstance(tool, dict) else None
            if kind == "function":
                if not isinstance(tool.get("name"), str):
                    raise Refusal("a function tool needs a name", f"tools[{i}].name", "invalid_value")
                fn = {"name": tool["name"]}
                for key in ("description", "parameters", "strict"):
                    if tool.get(key) is not None:
                        fn[key] = tool[key]
                functions.append({"type": "function", "function": fn})
                tools_echo.append({"type": "function", "name": tool["name"], "description": tool.get("description"),
                                   "parameters": tool.get("parameters"), "strict": tool.get("strict")})
            elif kind in WEB_SEARCH_TOOLS:
                if set(tool) - {"type"}:
                    raise Refusal("web search options are not supported", f"tools[{i}]")
                offered.add("search")
                tools_echo.append({"type": kind})
            elif kind == "image_generation":
                if set(tool) - {"type"}:
                    raise Refusal("image generation options are not supported", f"tools[{i}]")
                offered.add("image")
                tools_echo.append({"type": "image_generation"})
            else:
                raise Refusal(f"tool type {kind!r} is not supported", f"tools[{i}].type")
        # A request that mixes function tools with a built-in would be a silent
        # drop: the function tools would route the turn to chat
        # (skipped_client_tools) and the built-in would never run, but the
        # response object echoes the input tool list and would report it as if
        # it had. Refuse by name so the caller knows to send one or the other.
        if functions and offered:
            raise Refusal("function tools and built-in tools (web_search, image_generation) cannot be combined in one request",
                          "tools", "unsupported_value")
    if functions:
        chat["tools"] = functions
        if body.get("parallel_tool_calls") is not None:
            chat["parallel_tool_calls"] = body["parallel_tool_calls"]

    choice = body.get("tool_choice")
    if choice is not None:
        if choice in ("auto", "none", "required"):
            if choice == "none":
                offered.clear()
            if functions:
                chat["tool_choice"] = choice
            elif choice == "required":
                raise Refusal("tool_choice 'required' needs a function tool", "tool_choice")
        elif isinstance(choice, dict) and choice.get("type") == "function" and isinstance(choice.get("name"), str):
            chat["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        elif isinstance(choice, dict) and choice.get("type") == "allowed_tools":
            allowed = [t for t in choice.get("tools") or [] if isinstance(t, dict)]
            if any(t.get("type") != "function" for t in allowed):
                raise Refusal("allowed_tools may list only function tools", "tool_choice.tools")
            chat["tool_choice"] = {"type": "allowed_tools", "allowed_tools": {
                "mode": choice.get("mode") or "auto",
                "tools": [{"type": "function", "function": {"name": t.get("name")}} for t in allowed]}}
        else:
            raise Refusal("this tool_choice is not supported", "tool_choice")

    instructions = body.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise Refusal("instructions must be a string", "instructions", "invalid_value")
    # Instructions apply to this response only; they are never carried into the
    # stored conversation a later previous_response_id continues (spec).
    chat["messages"] = ([{"role": "system", "content": instructions}] if instructions else []) + (previous or []) + messages
    echo = {"instructions": instructions, "tools": tools_echo, "tool_choice": choice if choice is not None else "auto",
            "text": {"format": fmt, "verbosity": verbosity or "medium"}, "input_items": items, "input_messages": messages,
            "reasoning": {"effort": (reasoning or {}).get("effort"), "summary": None}}
    return chat, frozenset(offered), echo


def _param(param: str | None) -> str | None:
    if not param:
        return param
    head = param.split("[", 1)[0].split(".", 1)[0]
    return PARAM_NAMES.get(head, param) if head in PARAM_NAMES else param


# --- building the Response ------------------------------------------------------------

def _usage(chat_usage: dict | None) -> dict | None:
    if not chat_usage:
        return None
    return {
        "input_tokens": chat_usage.get("prompt_tokens") or 0,
        "input_tokens_details": {"cached_tokens": ((chat_usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0,
                                 "cache_write_tokens": 0},
        "output_tokens": chat_usage.get("completion_tokens") or 0,
        "output_tokens_details": {"reasoning_tokens": ((chat_usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0},
        "total_tokens": chat_usage.get("total_tokens") or 0,
    }


def _annotation(a: dict) -> dict | None:
    c = a.get("url_citation") if isinstance(a, dict) else None
    if not isinstance(c, dict):
        return None
    return {"type": "url_citation", "url": c.get("url", ""), "title": c.get("title", ""),
            "start_index": c.get("start_index", 0), "end_index": c.get("end_index", 0)}


def message_item(item_id: str, text: str, annotations: list[dict], status: str = "completed", refusal: str | None = None) -> dict:
    content = [{"type": "output_text", "text": text, "annotations": annotations, "logprobs": []}]
    if refusal:
        content = [{"type": "refusal", "refusal": refusal}]
    return {"id": item_id, "type": "message", "role": "assistant", "status": status, "content": content}


def image_item(item_id: str, mime: str, data: bytes | None, status: str = "completed") -> dict:
    return {"id": item_id, "type": "image_generation_call", "status": status,
            "result": base64.b64encode(data).decode() if data is not None else None,
            "output_format": mime.split("/", 1)[1] if data is not None else None}


def reasoning_item(item_id: str, text: str) -> dict:
    return {"id": item_id, "type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": text}]}


def _reasoning_text(part: dict) -> str:
    value = part.get("reasoning_content") if part.get("reasoning_content") is not None else part.get("reasoning")
    return value if isinstance(value, str) else ""


def wants_reasoning(body: dict) -> bool:
    effort = (body.get("reasoning") or {}).get("effort") if isinstance(body.get("reasoning"), dict) else None
    return effort not in (None, "none")


def search_item(item_id: str, query: str, status: str = "completed") -> dict:
    return {"id": item_id, "type": "web_search_call", "status": status, "action": {"type": "search", "query": query}}


def base_response(response_id: str, body: dict, echo: dict, created_at: int, store: bool,
                  conversation_id: str | None = None) -> dict:
    return {
        **({"conversation": {"id": conversation_id}} if conversation_id else {}),
        "id": response_id, "object": "response", "created_at": created_at, "status": "in_progress",
        "background": body.get("background") is True, "error": None, "incomplete_details": None,
        "instructions": echo["instructions"], "max_output_tokens": body.get("max_output_tokens"),
        "model": body["model"], "output": [], "parallel_tool_calls": body.get("parallel_tool_calls") is not False,
        "previous_response_id": body.get("previous_response_id"), "reasoning": echo["reasoning"],
        "temperature": body.get("temperature") if body.get("temperature") is not None else 1.0,
        "text": echo["text"], "tool_choice": echo["tool_choice"], "tools": echo["tools"],
        "top_p": body.get("top_p") if body.get("top_p") is not None else 1.0, "truncation": "disabled",
        "metadata": body.get("metadata") or {},
        **{k: body[k] for k in ("safety_identifier", "prompt_cache_key", "user") if body.get(k) is not None},
    }


def finish(response: dict, finish_reason: str | None, usage: dict | None, service_tier: str | None) -> dict:
    if finish_reason == "length":
        response.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    elif finish_reason == "content_filter":
        response.update(status="incomplete", incomplete_details={"reason": "content_filter"})
    else:
        response.update(status="completed", completed_at=int(time.time()))
    if usage is not None:
        response["usage"] = usage
    if service_tier:
        response["service_tier"] = service_tier
    return response


def conversation_after(previous: list[dict] | None, echo: dict, output: list[dict]) -> list[dict]:
    """What a later previous_response_id continues: the earlier turns, this
    input, and this response's output, as chat messages (never instructions)."""
    turn = [*(previous or []), *echo["input_messages"]]
    text = "".join(p.get("text", "") for item in output if item["type"] == "message"
                   for p in item["content"] if p["type"] == "output_text")
    calls = [{"id": item["call_id"], "type": "function", "function": {"name": item["name"], "arguments": item["arguments"]}}
             for item in output if item["type"] == "function_call"]
    if any(item["type"] == "image_generation_call" for item in output):
        text = (text + "\n\n" + IMAGE_MARKER).strip()
    assistant: dict = {"role": "assistant", "content": text or None}
    if calls:
        assistant["tool_calls"] = calls
    if text or calls:
        turn.append(assistant)
    return turn


# --- the doors -------------------------------------------------------------------------

def _start_background(app, store, request, body, echo, chat, offered, previous, conversation_id) -> JSONResponse:
    """background: true (spec): answer at once with a queued Response, run the
    turn as a task, and let retrieve report its progress; cancel stops it."""
    response_id = _id("resp")
    response = base_response(response_id, body, echo, int(time.time()), True, conversation_id)
    response["status"] = "queued"
    store.put(response, echo["input_items"], previous or [])

    async def run():
        # Every write from here is put_if_present: a row DELETEd mid-turn is a
        # tombstone. The unconditional puts used here resurrected a deleted
        # response as `completed`, input items and all, once the turn finished
        # (review 2026-09-22, #1, reproduced).
        if not store.put_if_present({**response, "status": "in_progress"}, echo["input_items"], previous or []):
            return
        sink: dict = {"trace_fields": {"operation": "POST /v1/responses (background)", "response_id": response_id},
                      "keep_reasoning": wants_reasoning(body)}
        try:
            result = await app.state.run_chat(chat, request, allowed_routes=offered, sink=sink,
                                              watch_client=False)  # the client already has its answer
            if result.status_code == 499 and getattr(result, "body", None) == b"":
                store.put_if_present({**response, "status": "failed", "error": {
                    "code": "server_error", "message": "The response failed.",
                }}, echo["input_items"], previous or [])
                return
            payload = json.loads(result.body)
            if result.status_code >= 400:
                err = payload.get("error") or {}
                failed = {**response, "status": "failed",
                          "error": {"code": "server_error" if result.status_code >= 500 else "invalid_prompt",
                                    "message": err.get("message") or "the response failed"}}
                store.put_if_present(failed, echo["input_items"], previous or [])
                return
            done = _from_chat(response_id, body, echo, payload, sink, True, conversation_id)
            done["created_at"] = response["created_at"]
            if not store.put_if_present(done, echo["input_items"], conversation_after(previous, echo, done["output"])):
                return                      # deleted mid-turn: nothing to append anywhere
            if conversation_id:
                try:
                    store.append_items(conversation_id, echo["input_items"] + done["output"])
                except DuplicateItemId as dup:
                    # Another request put this id in first, after our preflight
                    # (Copilot on #332): the turn cannot join the conversation.
                    failed = {k: v for k, v in done.items() if k != "completed_at"}
                    failed.update(status="failed", error={"code": "invalid_prompt", "message": str(dup)})
                    store.put_if_present(failed, echo["input_items"], previous or [])
        except asyncio.CancelledError:
            store.put_if_present({**response, "status": "cancelled"}, echo["input_items"], previous or [])
            raise
        except Exception:  # noqa: BLE001 - a crash is a failed response, never a stuck one
            logger.exception("background response %s failed", response_id)
            store.put_if_present({**response, "status": "failed", "error": {
                "code": "server_error", "message": "The response failed.",
            }}, echo["input_items"], previous or [])
    lease = getattr(request.state, "admission_lease", None)
    task = asyncio.get_running_loop().create_task(run())
    app.state.background[response_id] = task
    if lease is not None:
        lease.transfer_to(task)
    # A task cancelled before its first step skips run()'s finally entirely.
    # Remove it from the registry on every terminal path, including that one.
    task.add_done_callback(lambda _task: app.state.background.pop(response_id, None))
    return JSONResponse(construct_response(response))


def _transcript(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "[image]" if p.get("type") == "image_url" else "[audio]") for p in content)
        for call in m.get("tool_calls") or []:
            lines.append(f"assistant called {call['function']['name']}({call['function']['arguments']})")
        if role == "tool":
            lines.append(f"tool result ({m.get('tool_call_id')}): {content}")
        elif content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def register(app: FastAPI, deps, store: ResponseStore) -> None:
    # Wrapped, not rebound: the parameter stays the ResponseStore it is,
    # and every route reads the constructing wrapper under its own name.
    cstore = ConstructingStore(store)
    global _COMPACTION_KEY
    key_file = deps.settings.data_dir / "compaction.key"
    key_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        # O_CREAT|O_EXCL with mode 0600: never world-readable even briefly
        # (write_text creates 0644&umask and chmods after), and never
        # overwriting a key that exists -- an overwrite silently invalidates
        # every outstanding signed compaction token (review 2026-09-22).
        fd = os.open(key_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(secrets.token_hex(32))
    except FileExistsError:
        pass
    _COMPACTION_KEY = key_file.read_text().strip().encode()
    if not _COMPACTION_KEY:
        # A crash between the O_EXCL create and the write leaves an empty file
        # that every later start would load without complaint -- and HMAC
        # under an empty key is public knowledge, so the "tamper-proof"
        # compaction channel becomes forgeable system-prompt text. Fail closed,
        # loudly, with the remedy.
        raise RuntimeError(f"{key_file} is empty; delete it to regenerate the compaction key")
    app.state.background = {}
    cstore.abandon_interrupted()
    compact_prompt = (PROMPTS_DIR / "compact.md").read_text()

    @app.post("/v1/responses/compact")
    async def compact(request: Request):
        """Summarize the context into one compaction item a later request can use
        in place of the turns it replaces."""
        try:
            body = await request.json()
        except (ValueError, RecursionError):
            return _error(400, "body is not JSON", "invalid_json")
        if not isinstance(body, dict):
            return _error(400, "body must be a JSON object", "invalid_json")
        unknown = sorted(set(body) - COMPACT_FIELDS)
        if unknown:
            return _error(400, f"Unrecognized request argument supplied: {unknown[0]}", "unknown_parameter", unknown[0])
        model = body.get("model")
        # Resolved through graph_mod, like every other route. This check carried the
        # prefix as an inlined literal until 2026-09-17 and was the one place the
        # rename to chord-1-poly did not reach by changing the constant.
        if graph_mod.persona_for(model) is None:
            return _error(404, f"The model '{model}' does not exist", "model_not_found", "model")
        previous = []
        if body.get("previous_response_id") is not None:
            previous = cstore.conversation(body["previous_response_id"]) if isinstance(body["previous_response_id"], str) else None
            if previous is None:
                return _error(400, f"Previous response with id '{body['previous_response_id']}' not found.",
                              "previous_response_not_found", "previous_response_id")
            chained = cstore.get(body["previous_response_id"])
            if chained is not None and chained.get("status") not in TERMINAL:
                # A non-terminal row's stored conversation is missing its own
                # turn: chaining from it silently loses that context (#7).
                # Refused by name; terminal chains on.
                return _error(400, f"Previous response with id '{body['previous_response_id']}' is "
                                   f"{chained.get('status')}; it must finish before it can be chained.",
                              "previous_response_incomplete", "previous_response_id")
        try:
            current = input_to_messages(body["input"])[0] if body.get("input") is not None else []
        except Refusal as r:
            return _error(r.status, str(r), r.code, r.param)
        conversation = previous + current
        if not conversation:
            return _error(400, "nothing to compact: give input or previous_response_id", "missing_required_parameter", "input")
        system = compact_prompt + (f"\n\nAlso follow: {body['instructions']}" if isinstance(body.get("instructions"), str) else "")
        call = graph_mod.thinking_switch({   # as translations (review 2026-09-24 A6)
            "model": deps.settings.persona_model, "max_completion_tokens": 2048,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": _transcript(conversation)}]},
            deps.settings.persona_thinking_mode)
        try:
            data, _ = await deps.upstream.complete(call)
            summary = (data["choices"][0]["message"].get("content") or "").strip()
        except Exception:  # noqa: BLE001 - the backend failing is ours, a 502
            return _error(502, "the conversation could not be compacted right now", "compaction_failed", None, "server_error")
        if not summary:
            return _error(502, "the model returned an empty summary", "compaction_failed", None, "server_error")
        usage = _usage(data.get("usage")) or {"input_tokens": 0, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                                              "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 0}
        return JSONResponse({"id": _id("resp"), "object": "response.compaction", "created_at": int(time.time()),
                             "output": [{"id": _id("cmp"), "type": "compaction", "encrypted_content": compaction_token(summary)}],
                             "usage": usage})

    @app.post("/v1/responses/input_tokens")
    async def input_tokens(request: Request):
        """The exact prompt token count for a would-be response: the same
        translation, base layer and template as create, read from the model's
        own usage for a one-token prefill (a tokenizer beside the model would
        guess: the gateway's counter said 21 where the model counted 27)."""
        try:
            body = await request.json()
        except (ValueError, RecursionError):
            return _error(400, "body is not JSON", "invalid_json")
        if not isinstance(body, dict):
            return _error(400, "body must be a JSON object", "invalid_json")
        unknown = sorted(set(body) - COUNT_FIELDS)
        if unknown:
            return _error(400, f"Unrecognized request argument supplied: {unknown[0]}", "unknown_parameter", unknown[0])
        if body.get("personality") is not None:
            return _error(400, "personality is not supported", "unsupported_parameter", "personality")
        previous = None
        prev_id, conv = body.get("previous_response_id"), body.get("conversation")
        conversation_id = conv.get("id") if isinstance(conv, dict) else conv
        if conv is not None and prev_id is not None:
            return _error(400, "previous_response_id and conversation cannot be used together", "invalid_request", "conversation")
        if prev_id is not None:
            previous = cstore.conversation(prev_id) if isinstance(prev_id, str) else None
            if previous is None:
                return _error(400, f"Previous response with id '{prev_id}' not found.", "previous_response_not_found",
                              "previous_response_id")
            chained = cstore.get(prev_id)
            if chained is not None and chained.get("status") not in TERMINAL:
                return _error(400, f"Previous response with id '{prev_id}' is {chained.get('status')}; "
                                   "it must finish before it can be chained.",
                              "previous_response_incomplete", "previous_response_id")
        if conv is not None:
            if not isinstance(conversation_id, str) or cstore.conversation_resource(conversation_id) is None:
                return _error(404, f"Conversation with id '{conversation_id}' not found.", None, "conversation")
            try:
                previous = input_to_messages(cstore.conversation_items(conversation_id))[0] or None
            except Refusal:
                previous = None
        try:
            chat, _, _ = translate({k: v for k, v in body.items() if k not in ("previous_response_id", "conversation")}, previous)
        except Refusal as r:
            return _error(r.status, str(r), r.code, r.param)
        sink: dict = {"count_only": True, "trace_fields": {"operation": "POST /v1/responses/input_tokens"}}
        result = await app.state.run_chat(chat, request, allowed_routes=frozenset(), sink=sink)
        if result.status_code == 499 and getattr(result, "body", None) == b"":
            return Response(status_code=499)
        payload = json.loads(result.body)
        if result.status_code >= 400:
            err = payload.get("error") or {}
            return _error(result.status_code, err.get("message") or "request failed", err.get("code"),
                          _param(err.get("param")), err.get("type") or "invalid_request_error")
        count = (payload.get("usage") or {}).get("prompt_tokens")
        if not isinstance(count, int):
            return _error(502, "the model did not report a token count", "token_count_unavailable", None, "server_error")
        return JSONResponse({"object": "response.input_tokens", "input_tokens": count})

    @app.post("/v1/responses/{response_id}/cancel")
    async def cancel(response_id: str):
        response = cstore.get(response_id)
        if response is None:
            return _error(404, f"Response with id '{response_id}' not found.", None, None)
        if not response.get("background"):
            return _error(400, "Only responses created with background can be cancelled.", "invalid_request", None)
        task = app.state.background.get(response_id)
        if task is not None and not task.done():
            task.cancel()
            try:
                # The shield separates the two cancellations this await can
                # see: the TASK's own, which is the point of this route (run()
                # stores 'cancelled'), from OURS -- a client that left
                # mid-request -- which must propagate instead of being
                # swallowed so this handler keeps answering a dead connection
                # (review 2026-09-22).
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
            except Exception:  # noqa: BLE001 - run() already stored the failure
                pass
        elif response.get("status") in ("queued", "in_progress"):
            # The task is gone (a restart already failed these; this is the
            # same-process case where the handle is missing). Cancel must still
            # return status cancelled, not the stale in-progress object.
            response = {**response, "status": "cancelled"}
            cstore.replace_response(response)
        final = cstore.get(response_id)
        if final is None:
            # Deployment verification (2026-09-22) once watched a cancel of a
            # completed background response answer 200 with a bare `null`
            # body: this closing re-read came back empty and
            # JSONResponse(None) serialized it -- a failure shaped like a
            # success, the worst shape there is. An empty re-read is answered
            # as what it is: an unknown id, a 404.
            return _error(404, f"Response with id '{response_id}' not found.", None, None)
        return JSONResponse(final)

    @app.post("/v1/responses")
    async def create(request: Request):
        try:
            body = await request.json()
        except (ValueError, RecursionError):
            return _error(400, "body is not JSON", "invalid_json")
        if not isinstance(body, dict):
            return _error(400, "body must be a JSON object", "invalid_json")
        unknown = sorted(set(body) - CREATE_FIELDS)
        if unknown:
            return _error(400, f"Unrecognized request argument supplied: {unknown[0]}", "unknown_parameter", unknown[0])
        if body.get("store") is not None and not isinstance(body["store"], bool):
            return _error(400, "store must be a boolean", "invalid_type", "store")
        keep = body.get("store") is not False
        previous = None
        prev_id = body.get("previous_response_id")
        conv = body.get("conversation")
        conversation_id = conv.get("id") if isinstance(conv, dict) else conv
        if conv is not None:
            if prev_id is not None:
                return _error(400, "previous_response_id and conversation cannot be used together", "invalid_request",
                              "conversation")
            if not isinstance(conversation_id, str) or cstore.conversation_resource(conversation_id) is None:
                return _error(404, f"Conversation with id '{conversation_id}' not found.", None, "conversation")
            try:
                previous = input_to_messages(cstore.conversation_items(conversation_id))[0] or None
            except Refusal:
                previous = None
        if prev_id is not None:
            previous = cstore.conversation(prev_id) if isinstance(prev_id, str) else None
            if previous is None:
                return _error(400, f"Previous response with id '{prev_id}' not found.", "previous_response_not_found",
                              "previous_response_id")
            chained = cstore.get(prev_id)
            if chained is not None and chained.get("status") not in TERMINAL:
                return _error(400, f"Previous response with id '{prev_id}' is {chained.get('status')}; "
                                   "it must finish before it can be chained.",
                              "previous_response_incomplete", "previous_response_id")
        try:
            chat, offered, echo = translate(body, previous)
            if conversation_id is not None:
                dup = duplicate_item_id(echo["input_items"], cstore.conversation_items(conversation_id))
                if dup is not None:
                    raise Refusal(str(dup), "input", dup.code)
        except Refusal as r:
            return _error(r.status, str(r), r.code, r.param)
        stream = body.get("stream") is True
        if body.get("background") is not None and not isinstance(body["background"], bool):
            return _error(400, "background must be a boolean", "invalid_type", "background")
        if body.get("background") is True:
            if not keep:
                return _error(400, "background mode requires store", "invalid_request", "store")
            if stream:
                return _error(400, "streaming a background response is not supported", "unsupported_parameter", "stream")
            # base_response indexes body["model"]. Without this check a missing
            # model is a KeyError here, while the same body without background
            # reaches the chat door and returns 404 model_not_found.
            if graph_mod.persona_for(body.get("model")) is None:
                return _error(404, f"model {body.get('model')!r} not served; use one of /v1/models",
                              "model_not_found", "model")
            return _start_background(app, cstore, request, body, echo, chat, offered, previous, conversation_id)
        if stream:
            chat["stream"] = True
            chat["stream_options"] = {"include_usage": True}
        response_id = _id("resp")
        sink: dict = {"trace_fields": {"operation": "POST /v1/responses", "response_id": response_id},
                      "keep_reasoning": wants_reasoning(body)}
        result = await app.state.run_chat(chat, request, allowed_routes=offered, sink=sink)
        headers = {k: v for k, v in result.headers.items() if k in ("x-request-id", "x-chord-trace-id")}
        # Chat can stop before its first streamed byte when the caller leaves.
        # Its 499 deliberately has no JSON body and no announced response id.
        if result.status_code == 499 and getattr(result, "body", None) == b"":
            return Response(status_code=499, headers=headers)
        if result.status_code >= 400 or not stream:
            payload = json.loads(result.body)
            if result.status_code >= 400:
                err = payload.get("error") or {}
                return _error(result.status_code, err.get("message") or "request failed", err.get("code"),
                              _param(err.get("param")), err.get("type") or "invalid_request_error", headers)
            response = _from_chat(response_id, body, echo, payload, sink, keep, conversation_id)
            if keep:
                cstore.put(response, echo["input_items"], conversation_after(previous, echo, response["output"]))
            if conversation_id:
                try:
                    cstore.append_items(conversation_id, echo["input_items"] + response["output"])
                except DuplicateItemId as dup:
                    # The locked check caught what the preflight could not: a
                    # concurrent request appended this id while our model ran
                    # (Copilot on #332). The 400 is the answer, so the row goes.
                    if keep:
                        cstore.delete(response_id)
                    return _error(400, str(dup), "invalid_value", "input", headers=headers)
            return JSONResponse(construct_response(response), headers=headers)
        events = _stream(response_id, body, echo, result, sink, keep, cstore, previous, conversation_id)
        return ClosingStreamingResponse(events, media_type="text/event-stream", headers=headers)

    @app.get("/v1/responses/{response_id}")
    async def retrieve(response_id: str, request: Request):
        q = request.query_params
        if q.get("include") or q.getlist("include[]"):
            return _error(400, "include is not supported", "unsupported_parameter", "include")
        response = cstore.get(response_id)
        if response is None:
            return _error(404, f"Response with id '{response_id}' not found.", None, None)
        if q.get("stream") not in ("true", "True", "1"):
            if q.get("starting_after") is not None:
                return _error(400, "starting_after needs stream=true", "invalid_request", "starting_after")
            return JSONResponse(response)
        try:
            after = int(q["starting_after"]) if q.get("starting_after") is not None else -1
        except ValueError:
            return _error(400, "starting_after must be an integer", "invalid_type", "starting_after")
        # include_obfuscation is accepted: nothing here is ever obfuscated.
        def task_alive(rid: str) -> bool:
            task = app.state.background.get(rid)
            return task is not None and not task.done()

        return ClosingStreamingResponse(_replay(cstore, response_id, after, task_alive),
                                        media_type="text/event-stream")

    @app.delete("/v1/responses/{response_id}")
    async def delete(response_id: str):
        # Cancel the running turn BEFORE removing the row: the tombstone puts
        # stop the resurrection, and the cancel stops the WORK -- a deleted
        # response must not keep burning a model call nobody can retrieve
        # (2026-09-22, #1). The shield pattern is the cancel
        # route's: the task's own cancellation is expected; ours (a client
        # leaving mid-delete) must propagate.
        task = app.state.background.pop(response_id, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
            except Exception:  # noqa: BLE001 - run() already stored the failure
                pass
        if not cstore.delete(response_id):
            return _error(404, f"Response with id '{response_id}' not found.", None, None)
        return JSONResponse({"id": response_id, "object": "response.deleted", "deleted": True})

    @app.get("/v1/responses/{response_id}/input_items")
    async def input_items(response_id: str, request: Request):
        items = cstore.input_items(response_id)
        if items is None:
            return _error(404, f"Response with id '{response_id}' not found.", None, None)
        return _page(items, request)

    register_conversations(app, cstore)


CONVERSATION_ITEM_TYPES = {"message", "function_call", "function_call_output"}


def _conversation_items(raw) -> list[dict]:
    if not isinstance(raw, list):
        raise Refusal("items must be an array", "items", "invalid_value")
    if len(raw) > 20:
        raise Refusal("at most 20 items can be added at a time", "items", "array_above_max_length")
    for i, item in enumerate(raw):
        kind = item.get("type") or ("message" if isinstance(item, dict) and "role" in item else None) if isinstance(item, dict) else None
        if kind not in CONVERSATION_ITEM_TYPES:
            raise Refusal(f"item type {kind!r} is not supported", f"items[{i}].type")
    return input_to_messages(raw)[1] if raw else []


def duplicate_item_id(items: list[dict], existing: list[dict]) -> Refusal | None:
    """The refusal for an item id already in the conversation, or twice in
    this request; None when every id is new. Caller-supplied ids were taken as
    given, so two items could share msg_dup and deleting one deleted both
    (review 2026-09-24 B21)."""
    seen = {i["id"] for i in existing}
    for item in items:
        if item["id"] in seen:
            return Refusal(f"an item with id '{item['id']}' is already in this conversation", "items", "invalid_value")
        seen.add(item["id"])
    return None


def register_conversations(app: FastAPI, store: ResponseStore | ConstructingStore) -> None:
    async def json_body(request: Request) -> dict | JSONResponse:
        """The parsed JSON object body, or the ready error envelope.

        One value, narrowed by isinstance: pyright drops a union tuple's
        element correlation the moment it is destructured from a call, so the
        old (body, err) shape left every conversations route arguing with the
        checker about whether body could be None (pyright batch B, review
        2026-09-22)."""
        try:
            body = await request.json()
        except (ValueError, RecursionError):
            return _error(400, "body is not JSON", "invalid_json")
        if not isinstance(body, dict):
            return _error(400, "body must be a JSON object", "invalid_json")
        return body

    def not_found(conversation_id: str) -> JSONResponse:
        return _error(404, f"Conversation with id '{conversation_id}' not found.", None, None)

    def check_metadata(body: dict):
        metadata = body.get("metadata")
        if metadata is not None and not (isinstance(metadata, dict) and len(metadata) <= 16 and all(
                isinstance(k, str) and len(k) <= 64 and isinstance(v, str) and len(v) <= 512 for k, v in metadata.items())):
            raise Refusal("metadata must be at most 16 string pairs (keys <= 64, values <= 512 chars)", "metadata", "invalid_value")
        return metadata or {}

    @app.post("/v1/conversations")
    async def create_conversation(request: Request):
        body = await json_body(request)
        if isinstance(body, JSONResponse):
            return body
        unknown = sorted(set(body) - {"items", "metadata"})
        if unknown:
            return _error(400, f"Unrecognized request argument supplied: {unknown[0]}", "unknown_parameter", unknown[0])
        try:
            metadata = check_metadata(body)
            items = _conversation_items(body.get("items") or [])
            if (dup := duplicate_item_id(items, [])) is not None:
                raise dup
        except Refusal as r:
            return _error(r.status, str(r), r.code, r.param)
        return JSONResponse(store.create_conversation(_id("conv"), metadata, items))

    @app.get("/v1/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str):
        resource = store.conversation_resource(conversation_id)
        return JSONResponse(resource) if resource else not_found(conversation_id)

    @app.post("/v1/conversations/{conversation_id}")
    async def update_conversation(conversation_id: str, request: Request):
        if store.conversation_resource(conversation_id) is None:
            return not_found(conversation_id)
        body = await json_body(request)
        if isinstance(body, JSONResponse):
            return body
        unknown = sorted(set(body) - {"metadata"})
        if unknown:
            return _error(400, f"Unrecognized request argument supplied: {unknown[0]}", "unknown_parameter", unknown[0])
        if "metadata" not in body:
            return _error(400, "metadata is required", "missing_required_parameter", "metadata")
        try:
            metadata = check_metadata(body)
        except Refusal as r:
            return _error(r.status, str(r), r.code, r.param)
        resource = store.update_conversation(conversation_id, metadata)
        return JSONResponse(resource) if resource else not_found(conversation_id)

    @app.delete("/v1/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str):
        if not store.delete_conversation(conversation_id):
            return not_found(conversation_id)
        return JSONResponse({"id": conversation_id, "object": "conversation.deleted", "deleted": True})

    @app.post("/v1/conversations/{conversation_id}/items")
    async def add_items(conversation_id: str, request: Request):
        if store.conversation_resource(conversation_id) is None:
            return not_found(conversation_id)
        if request.query_params.getlist("include") or request.query_params.getlist("include[]"):
            return _error(400, "include is not supported", "unsupported_parameter", "include")
        body = await json_body(request)
        if isinstance(body, JSONResponse):
            return body
        unknown = sorted(set(body) - {"items"})
        if unknown:
            return _error(400, f"Unrecognized request argument supplied: {unknown[0]}", "unknown_parameter", unknown[0])
        if "items" not in body:
            return _error(400, "items is required", "missing_required_parameter", "items")
        try:
            items = _conversation_items(body["items"])
            if (dup := duplicate_item_id(items, store.conversation_items(conversation_id))) is not None:
                raise dup
        except Refusal as r:
            return _error(r.status, str(r), r.code, r.param)
        try:
            if not store.append_items(conversation_id, items):
                return not_found(conversation_id)
        except DuplicateItemId as dup:   # a concurrent add won the race past the preflight (Copilot on #332)
            return _error(400, str(dup), "invalid_value", "items")
        return JSONResponse({"object": "list", "data": items, "first_id": items[0]["id"] if items else None,
                             "last_id": items[-1]["id"] if items else None, "has_more": False})

    @app.get("/v1/conversations/{conversation_id}/items")
    async def list_items(conversation_id: str, request: Request):
        if store.conversation_resource(conversation_id) is None:
            return not_found(conversation_id)
        return _page(store.conversation_items(conversation_id), request)

    @app.get("/v1/conversations/{conversation_id}/items/{item_id}")
    async def get_item(conversation_id: str, item_id: str, request: Request):
        if store.conversation_resource(conversation_id) is None:
            return not_found(conversation_id)
        if request.query_params.getlist("include") or request.query_params.getlist("include[]"):
            return _error(400, "include is not supported", "unsupported_parameter", "include")
        item = next((i for i in store.conversation_items(conversation_id) if i["id"] == item_id), None)
        return JSONResponse(item) if item else _error(404, f"Item with id '{item_id}' not found.", None, None)

    @app.delete("/v1/conversations/{conversation_id}/items/{item_id}")
    async def delete_item(conversation_id: str, item_id: str):
        if store.conversation_resource(conversation_id) is None:
            return not_found(conversation_id)
        if not store.delete_item(conversation_id, item_id):
            return _error(404, f"Item with id '{item_id}' not found.", None, None)
        return JSONResponse(store.conversation_resource(conversation_id))


async def _replay(store: ResponseStore | ConstructingStore, response_id: str, after: int,
                  task_alive=None) -> AsyncIterator[str]:
    """A stored response as its event stream from sequence `after`+1: the events
    a streamed response really sent, or the deterministic replay of one that
    wasn't. A background response still running sends its opening events and
    waits for the rest -- bounded: a task that died on a BaseException leaves
    its row non-terminal forever, and every resumed client then polled this
    loop (a retention sweep inside every store.get) until process restart
    (review 2026-09-22). Past the deadline the row is failed the way
    abandon_interrupted fails a row whose process is gone; a task somehow still
    alive replaces it with the true outcome when it finishes."""
    sent_up_to = after
    deadline = asyncio.get_running_loop().time() + REPLAY_WAIT_S
    while True:
        response = store.get(response_id)
        if response is None:
            return
        events = store.events(response_id) or replay_events(response)
        for e in events:
            if e["sequence_number"] > sent_up_to:
                sent_up_to = e["sequence_number"]
                yield sse_frame(e)
        if response["status"] in TERMINAL:
            return
        if asyncio.get_running_loop().time() >= deadline:
            if task_alive is not None and task_alive(response_id):
                # The worker is still running: a passive observer must not
                # execute a live response. Re-arm and keep
                # waiting; the deadline exists for rows whose worker is GONE.
                deadline = asyncio.get_running_loop().time() + REPLAY_WAIT_S
            else:
                store.replace_response({**response, "status": "failed", "error": {
                    "code": "server_error", "message": "The response did not finish."}})
        await asyncio.sleep(0.25)


def _page(items: list[dict], request: Request) -> JSONResponse:
    """limit (1-100, default 20), order (default desc), after: the spec's cursor list."""
    q = request.query_params
    if q.getlist("include") or q.getlist("include[]"):
        return _error(400, "include is not supported", "unsupported_parameter", "include")
    try:
        limit = int(q.get("limit") or 20)
    except ValueError:
        limit = 0
    if not 1 <= limit <= 100:
        return _error(400, "limit must be between 1 and 100", "invalid_value", "limit")
    order = q.get("order") or "desc"
    if order not in ("asc", "desc"):
        return _error(400, "order must be asc or desc", "invalid_value", "order")
    ordered = items if order == "asc" else list(reversed(items))
    after = q.get("after")
    if after:
        ids = [i["id"] for i in ordered]
        if after not in ids:
            return _error(400, f"no item '{after}' in this list", "invalid_value", "after")
        ordered = ordered[ids.index(after) + 1:]
    page = ordered[:limit]
    return JSONResponse({"object": "list", "data": page, "first_id": page[0]["id"] if page else None,
                         "last_id": page[-1]["id"] if page else None, "has_more": len(ordered) > limit})


def _from_chat(response_id: str, body: dict, echo: dict, payload: dict, sink: dict, keep: bool,
               conversation_id: str | None = None) -> dict:
    response = base_response(response_id, body, echo, payload.get("created") or int(time.time()), keep, conversation_id)
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output = []
    trace = sink.get("trace")
    query = trace.fields.get("search_query") if trace is not None else None
    if query is not None:
        output.append(search_item(_id("ws"), query))
    thought = _reasoning_text(message)
    if thought:
        output.append(reasoning_item(_id("rs"), thought))
    text = message.get("content") or ""
    if text or message.get("refusal") or not message.get("tool_calls"):
        annotations = [a for a in (_annotation(x) for x in message.get("annotations") or []) if a]
        output.append(message_item(_id("msg"), text, annotations, refusal=message.get("refusal")))
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        output.append({"id": _id("fc"), "type": "function_call", "call_id": call.get("id"), "name": fn.get("name"),
                       "arguments": fn.get("arguments") or "", "status": "completed"})
    for mime, data in sink.get("images") or []:
        output.append(image_item(_id("ig"), mime, data))
    response["output"] = output
    return finish(response, choice.get("finish_reason"), _usage(payload.get("usage")), payload.get("service_tier"))


async def _chat_chunks(result) -> AsyncIterator[dict]:
    buffer = ""
    iterator = result.body_iterator
    try:
        async for piece in iterator:
            buffer += piece.decode() if isinstance(piece, bytes) else piece
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                for line in frame.splitlines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        yield json.loads(line[6:])
    finally:
        await iterator.aclose()


def sse_frame(payload: dict) -> str:
    payload = construct_event(payload)
    return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n"


# --- construction from the pinned spec's field lists (#189) ------------------------
# Every Response, item, content part and stream event leaves built from the field
# lists derived from the pinned spec (response_fields.json, drift-tested), never
# hand-assembled: #196 found an undeclared `name` on function_call_arguments.done
# that no test exercised. A field we set that the spec doesn't declare is dropped,
# and an event type the spec doesn't know is a bug that raises.
FIELDS = json.loads((Path(__file__).resolve().parent / "response_fields.json").read_text())


def _keep(obj: dict, fields) -> dict:
    return {k: v for k, v in obj.items() if k in fields}


def construct_item(item):
    if not isinstance(item, dict) or item.get("type") not in FIELDS["items"]:
        return item
    built = _keep(item, FIELDS["items"][item["type"]])
    if isinstance(built.get("content"), list):
        built["content"] = [_keep(part, FIELDS["parts"][part["type"]])
                            if isinstance(part, dict) and part.get("type") in FIELDS["parts"] else part
                            for part in built["content"]]
    return built


def construct_response(response: dict) -> dict:
    built = _keep(response, FIELDS["response"])
    if isinstance(built.get("output"), list):
        built["output"] = [construct_item(i) for i in built["output"]]
    return built


def construct_event(event: dict) -> dict:
    kind = event.get("type")
    if kind not in FIELDS["events"]:
        raise ValueError(f"event type {kind!r} is not in the pinned spec")
    built = _keep(event, FIELDS["events"][kind])
    if isinstance(built.get("response"), dict):
        built["response"] = construct_response(built["response"])
    if isinstance(built.get("item"), dict):
        built["item"] = construct_item(built["item"])
    if isinstance(built.get("part"), dict) and built["part"].get("type") in FIELDS["parts"]:
        built["part"] = _keep(built["part"], FIELDS["parts"][built["part"]["type"]])
    return built


class ConstructingStore:
    """The store as Responses sees it: what is kept is what is served, built."""
    def __init__(self, inner):
        self._inner = inner

    def put(self, response, input_items, conversation, events=None):
        return self._inner.put(construct_response(response), input_items, conversation,
                               [construct_event(e) for e in events] if events is not None else None)

    def put_if_present(self, response, input_items, conversation, events=None):
        return self._inner.put_if_present(construct_response(response), input_items, conversation,
                                          [construct_event(e) for e in events] if events is not None else None)

    def append_items(self, conversation_id, items):
        return self._inner.append_items(conversation_id, [construct_item(i) for i in items])

    def __getattr__(self, name):
        return getattr(self._inner, name)


TERMINAL = ("completed", "incomplete", "failed", "cancelled")

# How long one resumed stream waits for a non-terminal row before failing it
# (see _replay). Generous: a background turn can carry an image specialist
# render of a few minutes; an hour without a terminal status means the task is
# gone, not slow.
REPLAY_WAIT_S = 3600.0


def replay_events(response: dict) -> list[dict]:
    """The event sequence a stored response that was NOT streamed would have
    sent: deterministic, so a resume by sequence number is stable across reads."""
    events: list[dict] = []

    def add(kind: str, **fields) -> None:
        events.append(json.loads(json.dumps({"type": kind, "sequence_number": len(events), **fields})))

    opening = {**response, "status": "in_progress", "output": []}
    for key in ("usage", "completed_at"):
        opening.pop(key, None)
    opening.update(error=None, incomplete_details=None)
    add("response.created", response=opening)
    add("response.in_progress", response=opening)
    if response["status"] not in TERMINAL:
        return events
    for index, item in enumerate(response.get("output") or []):
        kind = item["type"]
        if kind == "message":
            add("response.output_item.added", output_index=index, item={**item, "status": "in_progress", "content": []})
            for ci, part in enumerate(item["content"]):
                if part["type"] == "output_text":
                    add("response.content_part.added", item_id=item["id"], output_index=index, content_index=ci,
                        part={"type": "output_text", "text": "", "annotations": [], "logprobs": []})
                    if part["text"]:
                        add("response.output_text.delta", item_id=item["id"], output_index=index, content_index=ci,
                            delta=part["text"], logprobs=[])
                    for ai, ann in enumerate(part.get("annotations") or []):
                        add("response.output_text.annotation.added", item_id=item["id"], output_index=index,
                            content_index=ci, annotation_index=ai, annotation=ann)
                    add("response.output_text.done", item_id=item["id"], output_index=index, content_index=ci,
                        text=part["text"], logprobs=[])
                else:
                    add("response.content_part.added", item_id=item["id"], output_index=index, content_index=ci,
                        part={"type": "refusal", "refusal": ""})
                    add("response.refusal.delta", item_id=item["id"], output_index=index, content_index=ci, delta=part["refusal"])
                    add("response.refusal.done", item_id=item["id"], output_index=index, content_index=ci, refusal=part["refusal"])
                add("response.content_part.done", item_id=item["id"], output_index=index, content_index=ci, part=part)
            add("response.output_item.done", output_index=index, item=item)
        elif kind == "reasoning":
            add("response.output_item.added", output_index=index, item={**item, "content": []})
            for ci, part in enumerate(item.get("content") or []):
                add("response.reasoning_text.delta", item_id=item["id"], output_index=index, content_index=ci, delta=part["text"])
                add("response.reasoning_text.done", item_id=item["id"], output_index=index, content_index=ci, text=part["text"])
            add("response.output_item.done", output_index=index, item=item)
        elif kind == "function_call":
            add("response.output_item.added", output_index=index, item={**item, "arguments": "", "status": "in_progress"})
            if item["arguments"]:
                add("response.function_call_arguments.delta", item_id=item["id"], output_index=index, delta=item["arguments"])
            add("response.function_call_arguments.done", item_id=item["id"], output_index=index,
                arguments=item["arguments"])
            add("response.output_item.done", output_index=index, item=item)
        elif kind == "web_search_call":
            add("response.output_item.added", output_index=index, item={**item, "status": "in_progress"})
            for stage in ("in_progress", "searching", "completed"):
                add(f"response.web_search_call.{stage}", output_index=index, item_id=item["id"])
            add("response.output_item.done", output_index=index, item=item)
        elif kind == "image_generation_call":
            add("response.output_item.added", output_index=index,
                item={**item, "status": "in_progress", "result": None, "output_format": None})
            for stage in ("in_progress", "generating", "completed"):
                add(f"response.image_generation_call.{stage}", output_index=index, item_id=item["id"])
            add("response.output_item.done", output_index=index, item=item)
    # `cancelled` has no entry, on purpose: the pinned spec's ResponseStreamEvent
    # ends only in completed, incomplete or failed (`response.cancelled` exists
    # only as a webhook). A cancelled replay is its opening pair and then the
    # stream closes; the status is on GET. Inventing an event, or relabelling a
    # cancel as response.failed, would be off-spec or untrue (review 2026-09-24
    # B21, stream replay).
    final = {"completed": "response.completed", "incomplete": "response.incomplete", "failed": "response.failed"}
    if response["status"] in final:
        add(final[response["status"]], response=response)
    return events


async def _stream(response_id, body, echo, result, sink, keep, store, previous,
                  conversation_id=None) -> AsyncIterator[str]:
    """The event stream, wrapped in the guarantee that the row it announces
    ends terminal.

    A disconnect closes this generator with the row still in_progress, and
    nothing else ever ends it: a streamed response is not in
    app.state.background, the _replay deadline only runs for a client that
    resumes, and the restart sweep used to skip non-background rows -- so the
    orphan waited forever, a poller polled forever, and the chaining refusal
    made it permanently unchainable (batch 2). The
    finally is SYNC work only: it runs under GeneratorExit, where yielding is
    illegal."""
    live: dict = {}
    inner = _stream_events(response_id, body, echo, result, sink, keep, store,
                           previous, conversation_id, live)
    try:
        async with contextlib.aclosing(inner):
            async for frame in inner:
                yield frame
    finally:
        response = live.get("response")
        if keep and response is not None:
            row = store.get(response_id)
            if row is not None and row.get("status") not in TERMINAL:
                # failed, in Chord's interruption vocabulary (the restart
                # sweep and the replay deadline say the same sentence); the
                # spec's `incomplete` is finish()'s, reserved for the length
                # and content_filter reasons. events=None on purpose: the
                # terminal event never crossed the wire, so a resume replays
                # the deterministic sequence -- which ends in response.failed
                # -- rather than stored partials with no ending.
                partial = live.get("partial")
                store.put_if_present(
                    {**response, "output": partial() if partial else [], "status": "failed",
                     "error": {"code": "server_error",
                               "message": "The response was interrupted before it finished."}},
                    echo["input_items"], previous or [], None)


async def _stream_events(response_id, body, echo, result, sink, keep, store, previous,
                         conversation_id, live: dict) -> AsyncGenerator[str, None]:
    seq = 0
    sent: list[dict] = []   # what went out, stored for GET ?stream=true resumes

    def event(kind: str, **fields) -> str:
        nonlocal seq
        payload = json.loads(json.dumps({"type": kind, "sequence_number": seq, **fields}))
        seq += 1
        sent.append(payload)
        return sse_frame(payload)

    response = base_response(response_id, body, echo, int(time.time()), keep, conversation_id)
    live["response"] = response
    if keep:
        # The row exists BEFORE the first event goes out: response.created
        # hands the caller an id, and a client that disconnected mid-stream
        # used to be left holding a resp_X that 404d forever -- the turn's
        # input lost in conversation mode with it (2026-09-22, #6).
        # What ends an orphaned row: the wrapper's finally on any generator
        # exit, abandon_interrupted on restart after a hard crash, and the
        # _replay deadline as the backstop for a worker that died silently.
        store.put(response, echo["input_items"], previous or [])
    yield event("response.created", response=json.loads(json.dumps(response)))
    yield event("response.in_progress", response=json.loads(json.dumps(response)))

    output: list[dict] = []
    live["output"] = output
    reasoning = None        # the open reasoning item, while the model thinks
    message = None          # the open message item

    def close_reasoning():
        # Only ever called with an open reasoning item; the assert is the
        # closure's narrowing that pyright cannot infer across scopes.
        assert reasoning is not None
        reasoning["done"] = True
        done = reasoning_item(reasoning["id"], reasoning["text"])
        output[reasoning["index"]] = done
        return [event("response.reasoning_text.done", item_id=reasoning["id"], output_index=reasoning["index"],
                      content_index=0, text=reasoning["text"]),
                event("response.output_item.done", output_index=reasoning["index"], item=done)]
    text, annotations, refusal = "", [], ""
    # The message's content parts in the order they opened: "text" and/or
    # "refusal". The content_index of a part is its place here. A refusal is a
    # part of its own, as non-stream and replay_events carry it (review
    # 2026-09-24 B3: `delta.refusal` was never read, so a declined turn went
    # out as an empty output_text).
    parts: list[str] = []
    calls: dict[int, dict] = {}
    call_index: dict[int, int] = {}
    finish_reason, usage, tier, searched = None, None, None, False

    def message_content() -> list[dict]:
        return [{"type": "output_text", "text": text, "annotations": annotations, "logprobs": []} if kind == "text"
                else {"type": "refusal", "refusal": refusal} for kind in parts]

    def open_part(kind: str):
        """The message item (on its first part) and the part's opening events."""
        nonlocal message
        out = []
        if message is None:
            message = {"id": _id("msg"), "type": "message", "role": "assistant", "status": "in_progress",
                       "content": [], "_index": len(output)}
            output.append(message)
            out.append(event("response.output_item.added", output_index=message["_index"],
                             item={k: v for k, v in message.items() if k != "_index"}))
        parts.append(kind)
        out.append(event("response.content_part.added", item_id=message["id"], output_index=message["_index"],
                         content_index=len(parts) - 1,
                         part={"type": "output_text", "text": "", "annotations": [], "logprobs": []} if kind == "text"
                         else {"type": "refusal", "refusal": ""}))
        return out

    def partial() -> list[dict]:
        """The output as far as it got, for a turn that ends early: the text and
        reasoning already sent to the client, not the empty placeholders the
        items opened with (review 2026-09-24 B21: an interrupted stream stored
        an empty in_progress message although its deltas had reached the
        client)."""
        now = list(output)
        if reasoning is not None and not reasoning.get("done"):
            now[reasoning["index"]] = reasoning_item(reasoning["id"], reasoning["text"])
        if message is not None and "_index" in message:
            now[message["_index"]] = {"id": message["id"], "type": "message", "role": "assistant",
                                      "status": "incomplete", "content": message_content()}
        for i, item in calls.items():
            now[call_index[i]] = {**item, "status": "incomplete"}
        return now

    live["partial"] = partial

    def open_search():
        nonlocal searched
        trace = sink.get("trace")
        query = trace.fields.get("search_query") if trace is not None else None
        if searched or query is None:
            return []
        searched = True
        item = search_item(_id("ws"), query, "in_progress")
        index = len(output)
        output.append(item)
        done = search_item(item["id"], query)
        output[index] = done
        return [event("response.output_item.added", output_index=index, item=item),
                event("response.web_search_call.in_progress", output_index=index, item_id=item["id"]),
                event("response.web_search_call.searching", output_index=index, item_id=item["id"]),
                event("response.web_search_call.completed", output_index=index, item_id=item["id"]),
                event("response.output_item.done", output_index=index, item=done)]

    async for chunk in _chat_chunks(result):
        if "error" in chunk:
            err = chunk["error"]
            response.update(status="failed", error={"code": "server_error", "message": err.get("message") or "failed"})
            output[:] = partial()
            response["output"] = output
            frame = event("response.failed", response=response)
            if keep:
                store.put_if_present(response, echo["input_items"], previous or [], sent)
            yield frame
            return
        tier = chunk.get("service_tier") or tier
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            thought = _reasoning_text(delta)
            if thought and message is None and not calls:
                if reasoning is None:
                    reasoning = {"id": _id("rs"), "index": len(output), "text": ""}
                    output.append(reasoning_item(reasoning["id"], ""))
                    yield event("response.output_item.added", output_index=reasoning["index"],
                                item={"id": reasoning["id"], "type": "reasoning", "summary": [], "content": []})
                reasoning["text"] += thought
                yield event("response.reasoning_text.delta", item_id=reasoning["id"], output_index=reasoning["index"],
                            content_index=0, delta=thought)
            for kind, piece in (("text", delta.get("content")), ("refusal", delta.get("refusal"))):
                if not piece or not isinstance(piece, str):
                    continue
                if reasoning is not None and not reasoning.get("done"):
                    for out in close_reasoning():
                        yield out
                for out in open_search():
                    yield out
                if kind not in parts:
                    for out in open_part(kind):
                        yield out
                assert message is not None   # open_part opened it; pyright cannot see through the closure
                if kind == "text":
                    text += piece
                    yield event("response.output_text.delta", item_id=message["id"], output_index=message["_index"],
                                content_index=parts.index("text"), delta=piece, logprobs=[])
                else:
                    refusal += piece
                    yield event("response.refusal.delta", item_id=message["id"], output_index=message["_index"],
                                content_index=parts.index("refusal"), delta=piece)
            for note in delta.get("annotations") or []:
                mapped = _annotation(note)
                if mapped and message is not None and "text" in parts:
                    annotations.append(mapped)
                    yield event("response.output_text.annotation.added", item_id=message["id"],
                                output_index=message["_index"], content_index=parts.index("text"),
                                annotation_index=len(annotations) - 1, annotation=mapped)
            for tc in delta.get("tool_calls") or []:
                i = tc.get("index", 0)
                fn = tc.get("function") or {}
                if i not in calls:
                    if reasoning is not None and not reasoning.get("done"):
                        for out in close_reasoning():
                            yield out
                    for out in open_search():
                        yield out
                    item = {"id": _id("fc"), "type": "function_call", "call_id": tc.get("id") or "",
                            "name": fn.get("name") or "", "arguments": "", "status": "in_progress"}
                    calls[i] = item
                    call_index[i] = len(output)
                    output.append(item)
                    yield event("response.output_item.added", output_index=call_index[i], item=dict(item))
                item = calls[i]
                if tc.get("id") and not item["call_id"]:
                    item["call_id"] = tc["id"]
                if fn.get("name") and not item["name"]:
                    item["name"] = fn["name"]
                if fn.get("arguments"):
                    item["arguments"] += fn["arguments"]
                    yield event("response.function_call_arguments.delta", item_id=item["id"],
                                output_index=call_index[i], delta=fn["arguments"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    if reasoning is not None and not reasoning.get("done"):
        for out in close_reasoning():
            yield out
    for out in open_search():
        yield out
    if message is None and not calls:
        for out in open_part("text"):
            yield out
    if message is not None:
        index = message["_index"]
        content = message_content()
        for ci, part in enumerate(content):
            if part["type"] == "output_text":
                yield event("response.output_text.done", item_id=message["id"], output_index=index, content_index=ci,
                            text=text, logprobs=[])
            else:
                yield event("response.refusal.done", item_id=message["id"], output_index=index, content_index=ci,
                            refusal=refusal)
            yield event("response.content_part.done", item_id=message["id"], output_index=index, content_index=ci, part=part)
        done = {"id": message["id"], "type": "message", "role": "assistant", "status": "completed", "content": content}
        del message["_index"]
        output[index] = done
        yield event("response.output_item.done", output_index=index, item=done)
    for i, item in sorted(calls.items()):
        item["status"] = "completed"
        yield event("response.function_call_arguments.done", item_id=item["id"], output_index=call_index[i],
                    arguments=item["arguments"])
        yield event("response.output_item.done", output_index=call_index[i], item=dict(item))
    for mime, data in sink.get("images") or []:
        pending = image_item(_id("ig"), mime, None, "in_progress")
        index = len(output)
        output.append(pending)
        yield event("response.output_item.added", output_index=index, item=pending)
        yield event("response.image_generation_call.in_progress", output_index=index, item_id=pending["id"])
        yield event("response.image_generation_call.generating", output_index=index, item_id=pending["id"])
        yield event("response.image_generation_call.completed", output_index=index, item_id=pending["id"])
        done = image_item(pending["id"], mime, data)
        output[index] = done
        yield event("response.output_item.done", output_index=index, item=done)

    response["output"] = output
    finish(response, finish_reason, _usage(usage), tier)
    kind = "response.incomplete" if response["status"] == "incomplete" else "response.completed"
    frame = event(kind, response=response)
    stored = True
    if keep:
        # put_if_present: a DELETE that landed mid-stream is a tombstone the
        # final write must not walk over -- the early store above reopens
        # exactly the resurrection the batch-1 tombstones closed.
        stored = store.put_if_present(response, echo["input_items"],
                                      conversation_after(previous, echo, output), sent)
    if conversation_id and stored:
        try:
            store.append_items(conversation_id, echo["input_items"] + output)
        except DuplicateItemId as dup:
            # A concurrent request appended this id after our preflight
            # (Copilot on #332). The completed frame has not gone out yet: take
            # it back (its sequence number too) and end the turn failed.
            sent.pop()
            seq -= 1
            response.update(status="failed", error={"code": "invalid_prompt", "message": str(dup)})
            response.pop("completed_at", None)
            frame = event("response.failed", response=response)
            if keep:
                store.put_if_present(response, echo["input_items"], previous or [], sent)
    yield frame
