"""Closed-default Chat Completions wire shaping."""

from __future__ import annotations

from .fingerprint import opaque as opaque_fingerprint


REASONING_FIELDS = ("reasoning_content", "reasoning")


def _drop_reasoning(payload: dict, stream: bool) -> dict:
    """OpenAI's Chat Completions never returns the model's reasoning text. We pass
    it through only when the client explicitly turned thinking on with our
    declared chat_template_kwargs extension (#143); otherwise it is dropped."""
    for choice in payload.get("choices") or []:
        part = choice.get("delta" if stream else "message")
        if isinstance(part, dict):
            for key in REASONING_FIELDS:
                part.pop(key, None)
    return payload


def _conform_choices(payload: dict, stream: bool) -> dict:
    """Fields the spec requires but allows to be null must be PRESENT, not
    absent (pinned openai-openapi 4bb21ba, qa/conformance schema lane):
    non-stream choice.logprobs and message.refusal; stream choice.finish_reason.
    A client that reads `choice["logprobs"]` gets a KeyError from us and None
    from OpenAI, which is the difference this closes."""
    for choice in payload.get("choices") or []:
        choice.setdefault("index", 0)
        if stream:
            choice.setdefault("finish_reason", None)
            continue
        choice.setdefault("logprobs", None)
        message = choice.get("message")
        if isinstance(message, dict):
            upstream_block = message.pop("provider_specific_fields", None)
            if not isinstance(upstream_block, dict):
                upstream_block = {}
            if "refusal" not in message:
                # Our upstream (LiteLLM) moves the model's refusal into its
                # extension block; restore it to where the spec puts it.
                message["refusal"] = upstream_block.get("refusal")
            # The rest of that block (reasoning, refusal copies) is the inner
            # hop's, not our contract: never forwarded to clients (#142).
    return _construct(payload, stream)


# The wire is BUILT from the pinned spec's field lists (openai-openapi 4bb21ba),
# never by subtracting known extras from what upstream sent: subtraction leaked
# every novel upstream field (prompt_logprobs, routed_experts, kv_transfer_params,
# stop_reason, prompt_token_ids...), #187. Declared extensions
# (manifest response_extensions) are the only additions: reasoning_content, which
# only survives _drop_reasoning for a caller who turned thinking on, and streamed
# annotations for web_search_options.
_BODY_FIELDS = frozenset({"id", "object", "created", "model", "choices", "usage", "service_tier", "system_fingerprint",
                          "metadata", "moderation"})
_CHUNK_FIELDS = frozenset({"id", "object", "created", "model", "choices", "usage", "service_tier", "system_fingerprint",
                           "moderation", "obfuscation"})
_CHOICE_FIELDS = frozenset({"index", "message", "finish_reason", "logprobs"})
_STREAM_CHOICE_FIELDS = frozenset({"index", "delta", "finish_reason", "logprobs"})
_MESSAGE_FIELDS = frozenset({"role", "content", "refusal", "tool_calls", "annotations", "audio", "function_call",
                             "reasoning_content"})
_DELTA_FIELDS = frozenset({"role", "content", "refusal", "tool_calls", "function_call", "reasoning_content", "annotations"})
_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details", "completion_tokens_details")
_PROMPT_DETAIL_FIELDS = ("audio_tokens", "cached_tokens", "text_tokens", "image_tokens", "cache_write_tokens")
_COMPLETION_DETAIL_FIELDS = (
    "accepted_prediction_tokens",
    "audio_tokens",
    "reasoning_tokens",
    "text_tokens",
    "rejected_prediction_tokens",
)


def _pick(obj: dict, allowed: frozenset) -> dict:
    if "reasoning" in obj and "reasoning_content" not in obj:
        obj = {**obj, "reasoning_content": obj["reasoning"]}   # the upstream alias, under the one declared name
    return {k: v for k, v in obj.items() if k in allowed}


def _only(obj, fields: tuple) -> dict | None:
    return {k: obj[k] for k in fields if k in obj} if isinstance(obj, dict) else obj


def _tool_call(call, stream: bool):
    if not isinstance(call, dict):
        return call
    if call.get("type") == "custom":
        built = _only(call, ("id", "type"))
        assert built is not None
        built["custom"] = _only(call.get("custom"), ("name", "input"))
        return built
    built = _only(call, ("index", "id", "type") if stream else ("id", "type"))
    assert built is not None
    if "function" in call:
        built["function"] = _only(call["function"], ("name", "arguments"))
    return built


def _annotation(note):
    if not isinstance(note, dict):
        return note
    built = _only(note, ("type",))
    assert built is not None
    if "url_citation" in note:
        built["url_citation"] = _only(note["url_citation"], ("start_index", "end_index", "url", "title"))
    return built


def _nested(part: dict, stream: bool) -> dict:
    """One level down from _pick: objects inside the message or delta are rebuilt
    from their own spec field lists too (a novel key inside a tool call
    is still a novel key on the wire)."""
    if isinstance(part.get("tool_calls"), list):
        part["tool_calls"] = [_tool_call(c, stream) for c in part["tool_calls"]]
    if isinstance(part.get("annotations"), list):
        part["annotations"] = [_annotation(a) for a in part["annotations"]]
    if isinstance(part.get("audio"), dict):
        part["audio"] = _only(part["audio"], ("id", "expires_at", "data", "transcript"))
    if isinstance(part.get("function_call"), dict):
        part["function_call"] = _only(part["function_call"], ("name", "arguments"))
    # Unlike content, refusal, and audio, these optional fields are not nullable
    # in the OpenAI schema. Upstreams commonly materialize absent fields as null.
    for key in ("tool_calls", "annotations", "function_call"):
        if part.get(key) is None:
            part.pop(key, None)
    return part


def _usage(value: dict) -> dict:
    built = _only(value, _USAGE_FIELDS)
    assert built is not None
    for key, fields in (
        ("prompt_tokens_details", _PROMPT_DETAIL_FIELDS),
        ("completion_tokens_details", _COMPLETION_DETAIL_FIELDS),
    ):
        details = value.get(key)
        if isinstance(details, dict):
            built[key] = _only(details, fields)
        elif details is None:
            built.pop(key, None)
    return built


def _construct(payload: dict, stream: bool) -> dict:
    out = _pick(payload, _CHUNK_FIELDS if stream else _BODY_FIELDS)
    usage = out.get("usage")
    if isinstance(usage, dict):
        out["usage"] = _usage(usage)
    elif usage is None and not stream:
        out.pop("usage", None)
    if "system_fingerprint" in out:
        out["system_fingerprint"] = opaque_fingerprint(out["system_fingerprint"])
    part_key, part_fields = ("delta", _DELTA_FIELDS) if stream else ("message", _MESSAGE_FIELDS)
    choices = []
    for choice in payload.get("choices") or []:
        built = _pick(choice, _STREAM_CHOICE_FIELDS if stream else _CHOICE_FIELDS)
        if isinstance(choice.get(part_key), dict):
            built[part_key] = _nested(_pick(choice[part_key], part_fields), stream)
        choices.append(built)
    if "choices" in payload:
        out["choices"] = choices
    return out
