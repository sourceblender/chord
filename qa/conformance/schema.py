#!/usr/bin/env python3
"""Validate Chord responses against the pinned OpenAI OpenAPI schemas.

Every emitted line has the shared conformance result shape. Payload content is
never included in evidence: failures report only JSON paths and schema errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import httpx
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012


SPEC_PATH = Path(__file__).parent / "spec" / "openapi.json"
SPEC_SHA256 = "3d6223349eadfd937624b9e6b8abf596ec2f680a1a367889cf6a6f924e568127"
SPEC_COMMIT = "4bb21ba8e9213c3d955b69dc3f76dd7537439828"

SCHEMAS = {
    "chat": "CreateChatCompletionResponse",
    "chat-stream": "CreateChatCompletionStreamResponse",
    "completion": "CreateCompletionResponse",
    "images": "ImagesResponse",
    "models": "ListModelsResponse",
    # GET /v1/models/{model} and DELETE /v1/models/{model} had NO kind at all, so
    # neither object could be validated by the strict lane and a regression on
    # either was invisible to it. S13f required checking
    # DeleteModelResponse against the pinned schema by hand; `Model` was the same
    # hole one route over, found reading the map.
    "model": "Model",
    "model-deleted": "DeleteModelResponse",
    "error": "ErrorResponse",
    "transcription": "CreateTranscriptionResponseJson",
    "response": "Response",
    "response-event": "ResponseStreamEvent",
    "response-items": "ResponseItemList",
    "conversation": "ConversationResource",
    "conversation-deleted": "DeletedConversationResource",
    "conversation-items": "ConversationItemList",
    "chat-list": "ChatCompletionList",
    "chat-messages": "ChatCompletionMessageList",
    "response-input-tokens": "TokenCountsResource",
    "response-compaction": "CompactResource",
    "translation": "CreateTranslationResponseJson",
    "file": "OpenAIFile",
    "file-list": "ListFilesResponse",
    "file-deleted": "DeleteFileResponse",
    "embedding": "CreateEmbeddingResponse",
    "beta-response": "BetaResponse",
    "beta-response-event": "BetaResponseStreamEvent",
    "beta-response-items": "BetaResponseItemList",
    "beta-response-input-tokens": "BetaTokenCountsResource",
    "beta-response-compaction": "BetaCompactResource",
}

PATHS = {
    "chat": "/v1/chat/completions",
    "chat-stream": "/v1/chat/completions",
    "completion": "/v1/completions",
    "images": "/v1/images/generations",
    "models": "/v1/models",
    "model": "/v1/models/{model}",
    "model-deleted": "/v1/models/{model}",
    "error": "/v1/chat/completions",
    "transcription": "/v1/audio/transcriptions",
    "response": "/v1/responses",
    "response-event": "/v1/responses",
    "response-items": "/v1/responses/{response_id}/input_items",
    "conversation": "/v1/conversations/{conversation_id}",
    "conversation-deleted": "/v1/conversations/{conversation_id}",
    "conversation-items": "/v1/conversations/{conversation_id}/items",
    "chat-list": "/v1/chat/completions",
    "chat-messages": "/v1/chat/completions/{completion_id}/messages",
    "response-input-tokens": "/v1/responses/input_tokens",
    "response-compaction": "/v1/responses/compact",
    "translation": "/v1/audio/translations",
    "file": "/v1/files/{file_id}",
    "file-list": "/v1/files",
    "file-deleted": "/v1/files/{file_id}",
    "embedding": "/v1/embeddings",
    "beta-response": "/v1/responses?beta=true",
    "beta-response-event": "/v1/responses?beta=true",
    "beta-response-items": "/v1/responses/{response_id}/input_items?beta=true",
    "beta-response-input-tokens": "/v1/responses/input_tokens?beta=true",
    "beta-response-compaction": "/v1/responses/compact?beta=true",
}


def result(check: str, path: str, verdict: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"check": check, "path": path, "verdict": verdict, "evidence": evidence}


def _json_path(parts: Iterable[Any]) -> str:
    out = "$"
    for part in parts:
        out += f"[{part}]" if isinstance(part, int) else f".{part}"
    return out


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _oas_nullable(value: Any) -> Any:
    """Interpret the nullable keyword retained in the pinned OpenAPI 3.1 file.

    The document uses both JSON Schema null unions and the older OAS nullable
    keyword. The API examples explicitly contain those nulls. Normalizing that
    keyword prevents the validator from rejecting values the published schema
    describes as nullable.
    """
    if isinstance(value, list):
        return [_oas_nullable(item) for item in value]
    if not isinstance(value, dict):
        return value
    obj = {key: _oas_nullable(item) for key, item in value.items() if key != "nullable"}
    if value.get("nullable") is True:
        if isinstance(obj.get("type"), str):
            obj["type"] = [obj["type"], "null"]
            if isinstance(obj.get("enum"), list) and None not in obj["enum"]:
                obj["enum"] = [*obj["enum"], None]
        elif isinstance(obj.get("type"), list):
            if "null" not in obj["type"]:
                obj["type"] = [*obj["type"], "null"]
        else:
            obj = {"anyOf": [obj, {"type": "null"}]}
    return obj


class Spec:
    def __init__(self, path: Path = SPEC_PATH):
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != SPEC_SHA256:
            raise ValueError(f"OpenAI spec hash mismatch: expected {SPEC_SHA256}, got {actual}")
        self.document = _oas_nullable(json.loads(raw))
        self.registry = Registry().with_resource(
            "urn:openai:api", Resource(contents=self.document, specification=DRAFT202012)
        )

    def validator(self, schema_name: str) -> Draft202012Validator:
        if schema_name not in self.document["components"]["schemas"]:
            raise KeyError(f"schema absent from pinned spec: {schema_name}")
        return Draft202012Validator(
            {"$ref": f"urn:openai:api#/components/schemas/{schema_name}"},
            registry=self.registry,
            format_checker=FormatChecker(),
        )


# --- Undeclared fields (#142) -------------------------------------------------
# JSON Schema objects are open unless the schema closes them, so a payload can be
# schema-valid while carrying fields the spec never declared. This walks the
# payload beside the pinned schema and names every key no applicable branch
# declares. A schema-sanctioned open map (additionalProperties / patternProperties,
# or an object schema with no declared properties at all) is open, not extra.

def _absolute_refs(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: ("urn:openai:api" + v if k == "$ref" and isinstance(v, str) and v.startswith("#") else _absolute_refs(v))
                for k, v in node.items()}
    if isinstance(node, list):
        return [_absolute_refs(v) for v in node]
    return node


def _resolve(spec: "Spec", node: Any) -> Any:
    while isinstance(node, dict) and isinstance(node.get("$ref"), str):
        node = spec.document["components"]["schemas"][node["$ref"].rsplit("/", 1)[-1]]
    return node


def _valid(spec: "Spec", node: Any, instance: Any) -> bool:
    return Draft202012Validator(_absolute_refs(node), registry=spec.registry).is_valid(instance)


def _branches(spec: "Spec", node: Any, instance: Any) -> list[dict]:
    """Every concrete subschema that applies to this instance."""
    node = _resolve(spec, node)
    if not isinstance(node, dict):
        return []
    out = [node]
    for sub in node.get("allOf", []):
        out += _branches(spec, sub, instance)
    for key in ("anyOf", "oneOf"):
        for sub in node.get(key, []):
            if _valid(spec, sub, instance):
                out += _branches(spec, sub, instance)
    return out


def undeclared_paths(payload: Any, schema_name: str, spec: "Spec", values: dict[str, list[Any]] | None = None) -> list[str]:
    """JSON paths ([*] for array elements) of keys the pinned schema never declares.
    If `values` is given, it collects every value seen at each undeclared path, so a
    declared extension's shape can be checked too, not only its name (#143)."""
    found: list[str] = []
    seen = values if values is not None else {}

    def undeclared_below(value: Any, path: str) -> None:
        # Everything under an undeclared key is undeclared too, so an extension
        # allowlist has to name each nested path exactly (no silent subtrees).
        if isinstance(value, dict):
            for key, child in value.items():
                found.append(f"{path}.{key}")
                seen.setdefault(f"{path}.{key}", []).append(child)
                undeclared_below(child, f"{path}.{key}")
        elif isinstance(value, list):
            for child in value:
                undeclared_below(child, f"{path}[*]")

    def walk(instance: Any, nodes: list[Any], path: str) -> None:
        branches = [b for n in nodes for b in _branches(spec, n, instance)]
        if isinstance(instance, dict):
            declared: dict[str, list[Any]] = {}
            open_map, any_properties = [], False
            for b in branches:
                props = b.get("properties")
                if isinstance(props, dict):
                    any_properties = True
                    for k, v in props.items():
                        declared.setdefault(k, []).append(v)
                extra = b.get("additionalProperties")
                if extra not in (None, False) or b.get("patternProperties"):
                    open_map.append(extra if isinstance(extra, dict) else {})
            for key, value in instance.items():
                child = f"{path}.{key}"
                if key in declared:
                    walk(value, declared[key], child)
                elif open_map:
                    walk(value, open_map, child)
                elif any_properties:
                    found.append(child)  # closed-by-intent object: an extra
                    seen.setdefault(child, []).append(value)
                    undeclared_below(value, child)
                # else: a free-form object schema; anything goes
        elif isinstance(instance, list):
            items = [b["items"] for b in branches if isinstance(b.get("items"), dict)]
            for value in instance:
                walk(value, items, f"{path}[*]")

    walk(payload, [{"$ref": f"#/components/schemas/{schema_name}"}], "$")
    return sorted(set(found))


def response_extensions(kind: str) -> dict[str, dict]:
    """Exact declared extension paths for this payload kind, from the manifest."""
    from chord import manifest  # the product declares its own extensions
    return {e["path"]: e for e in manifest.load().get("response_extensions", []) if kind in e.get("modes", [])}


def extension_shape_errors(values: dict[str, list[Any]], allowed: dict[str, dict]) -> list[dict[str, Any]]:
    """A declared extension must also have its declared shape: type, required keys.
    Paths and rules only, never the offending value."""
    errors = []
    for path, entry in allowed.items():
        schema = entry.get("schema")
        if not schema:
            continue
        for value in values.get(path, []):
            for err in Draft202012Validator(schema).iter_errors(value):
                detail = {"json_path": path + _json_path(err.absolute_path)[1:], "validator": err.validator,
                          "instance_type": _json_type(err.instance)}
                if err.validator in {"type", "enum", "const", "required"}:
                    detail["expectation"] = err.validator_value
                errors.append(detail)
    return errors


def _covered(path: str, allowed: dict[str, str]) -> bool:
    return path in allowed


def validate_payload(
    payload: Any,
    *,
    kind: str,
    check: str | None = None,
    path: str | None = None,
    spec: Spec | None = None,
    fields: str | None = None,
) -> dict[str, Any]:
    """fields=None: the schema alone. fields="strict": also fail on ANY field outside
    the pinned spec, our own extensions included. fields="profile": also fail on
    undeclared fields, but pass the manifest's exact extension paths, which are
    reported, never hidden (#142)."""
    spec = spec or Spec()
    schema_name = SCHEMAS[kind]
    errors = sorted(spec.validator(schema_name).iter_errors(payload), key=lambda e: list(e.absolute_path))
    evidence: dict[str, Any] = {"schema": schema_name, "spec_version": "2.3.0", "spec_commit": SPEC_COMMIT}
    if errors:
        evidence["errors"] = []
        for error in errors[:20]:
            detail = {
                "json_path": _json_path(error.absolute_path),
                "schema_path": _json_path(error.absolute_schema_path),
                "validator": error.validator,
                "instance_type": _json_type(error.instance),
            }
            if error.validator in {"type", "enum", "const", "required"}:
                detail["expectation"] = error.validator_value
            evidence["errors"].append(detail)
        evidence["error_count"] = len(errors)
    else:
        evidence["validated"] = True
    extras_fail = False
    if fields in ("strict", "profile") and isinstance(payload, (dict, list)):
        values: dict[str, list[Any]] = {}
        extras = undeclared_paths(payload, schema_name, spec, values)
        allowed = response_extensions(kind) if fields == "profile" else {}
        evidence["undeclared_fields"] = [p for p in extras if not _covered(p, allowed)]
        if fields == "profile":
            evidence["declared_extensions"] = [p for p in extras if _covered(p, allowed)]
            evidence["extension_shape_errors"] = extension_shape_errors(values, allowed)
        extras_fail = bool(evidence["undeclared_fields"] or evidence.get("extension_shape_errors"))
        check = check or f"schema.{kind}.{fields}"
    return result(check or f"schema.{kind}", path or PATHS[kind], "fail" if errors or extras_fail else "pass", evidence)


def validate_http_response(
    response: httpx.Response,
    *,
    kind: str,
    check: str,
    expected_status: int,
    spec: Spec,
) -> dict[str, Any]:
    """Validate both the expected HTTP outcome and its published JSON shape."""
    schema_kind = kind if response.status_code == expected_status else "error"
    try:
        payload = response.json()
    except (ValueError, UnicodeDecodeError):
        row = result(check, PATHS[kind], "fail", {
            "schema": SCHEMAS[schema_kind],
            "spec_version": "2.3.0",
            "spec_commit": SPEC_COMMIT,
            "error_type": "non-json-response",
        })
    else:
        row = validate_payload(payload, kind=schema_kind, check=check, spec=spec)
    row["evidence"]["status"] = response.status_code
    row["evidence"]["expected_status"] = expected_status
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    row["evidence"]["content_type"] = content_type
    if response.status_code != expected_status or content_type != "application/json":
        row["verdict"] = "fail"
    return row


def parse_sse(raw: bytes | str) -> tuple[list[Any], bool, list[str]]:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    chunks, done, framing_errors, data_lines = [], False, [], []

    def dispatch(line_no: int) -> None:
        nonlocal done, data_lines
        if not data_lines:
            return
        data = "\n".join(data_lines)
        data_lines = []
        if data == "[DONE]":
            if done:
                framing_errors.append(f"line {line_no}: duplicate [DONE]")
            done = True
            return
        if done:
            framing_errors.append(f"line {line_no}: data after [DONE]")
            return
        try:
            chunks.append(json.loads(data))
        except json.JSONDecodeError as exc:
            framing_errors.append(f"line {line_no}: invalid JSON ({exc.msg})")

    for line_no, line in enumerate([*text.splitlines(), ""], 1):
        if not line:
            dispatch(line_no)
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            continue
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_lines.append(value)
        elif field in {"event", "id", "retry"}:
            continue
    return chunks, done, framing_errors


def validate_stream(raw: bytes | str, *, spec: Spec | None = None) -> list[dict[str, Any]]:
    spec = spec or Spec()
    chunks, done, framing_errors = parse_sse(raw)
    rows = [
        validate_payload(chunk, kind="chat-stream", check=f"schema.chat.stream.chunk[{index}]", spec=spec)
        for index, chunk in enumerate(chunks)
    ]
    evidence = {"chunks": len(chunks), "done": done}
    if framing_errors:
        evidence["errors"] = framing_errors[:20]
    framing_pass = bool(chunks) and done and not framing_errors
    rows.append(result("schema.chat.stream.framing", PATHS["chat-stream"],
                       "pass" if framing_pass else "fail", evidence))
    return rows


def field_rows(payloads: list[Any], kind: str, check: str, spec: Spec) -> list[dict[str, Any]]:
    """Two rows over one response (all its stream chunks together): `.strict`
    fails on any field outside the pinned spec; `.profile` fails only on fields
    that are neither in the spec nor an exact declared extension (#142)."""
    rows = []
    for mode in ("strict", "profile"):
        allowed = response_extensions(kind) if mode == "profile" else {}
        undeclared, declared, values = set(), set(), {}
        for payload in payloads:
            if isinstance(payload, (dict, list)):
                for path in undeclared_paths(payload, SCHEMAS[kind], spec, values):
                    (declared if _covered(path, allowed) else undeclared).add(path)
        evidence = {"schema": SCHEMAS[kind], "payloads": len(payloads), "undeclared_fields": sorted(undeclared)}
        shape = extension_shape_errors(values, allowed) if mode == "profile" else []
        if mode == "profile":
            evidence["declared_extensions"] = sorted(declared)
            evidence["extension_shape_errors"] = shape
        rows.append(result(f"{check}.{mode}", PATHS[kind], "fail" if undeclared or shape else "pass", evidence))
    return rows


def _headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _buffered_decoded_response(response: httpx.Response, content: bytes) -> httpx.Response:
    """Rebuild an iterated response without headers for the encoded wire body."""
    headers = [(key, value) for key, value in response.headers.multi_items()
               if key.lower() not in {"content-encoding", "content-length"}]
    return httpx.Response(response.status_code, headers=headers, content=content)


def _parsed(response: httpx.Response) -> Any:
    """The body as JSON, or None. Parsed once, never raised: a 200 that is not JSON
    is already a failing schema row and must not abort the rest of the run (#143)."""
    try:
        return response.json()
    except (ValueError, UnicodeDecodeError):
        return None


def _with_fields(rows: list[dict[str, Any]], response: httpx.Response, kind: str, check: str, spec: Spec) -> None:
    """Strict and profile rows for any response whose body parsed, success or error."""
    body = _parsed(response)
    if isinstance(body, (dict, list)):
        family = kind if response.status_code < 400 else "error"
        rows.extend(field_rows([body], family, f"{check}.fields", spec))


def run_live(base_url: str, model: str, api_key: str | None, include_images: bool, timeout: float,
             transport: httpx.BaseTransport | None = None) -> list[dict[str, Any]]:
    spec = Spec()
    base = base_url.rstrip("/")
    rows: list[dict[str, Any]] = []
    headers = _headers(api_key)
    with httpx.Client(timeout=timeout, headers=headers, transport=transport) as client:
        response = client.get(f"{base}/models")
        rows.append(validate_http_response(response, kind="models", check="schema.live.models",
                                           expected_status=200, spec=spec))
        _with_fields(rows, response, "models", "schema.live.models", spec)

        body = {"model": model, "messages": [{"role": "user", "content": "Reply with exactly: hello"}]}
        response = client.post(f"{base}/chat/completions", json=body)
        rows.append(validate_http_response(response, kind="chat", check="schema.live.chat",
                                           expected_status=200, spec=spec))
        _with_fields(rows, response, "chat", "schema.live.chat", spec)

        with client.stream("POST", f"{base}/chat/completions", json={**body, "stream": True}) as response:
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            raw = b"".join(response.iter_bytes())
        if response.status_code == 200:
            rows.append(result("schema.live.chat-stream.content-type", PATHS["chat-stream"],
                               "pass" if content_type == "text/event-stream" else "fail",
                               {"expected": "text/event-stream", "actual": content_type}))
            rows.extend(validate_stream(raw, spec=spec))
            rows.extend(field_rows(parse_sse(raw)[0], "chat-stream", "schema.live.chat-stream.fields", spec))
        else:
            buffered = _buffered_decoded_response(response, raw)
            rows.append(validate_http_response(buffered, kind="chat-stream",
                                               check="schema.live.chat-stream.error",
                                               expected_status=200, spec=spec))
            _with_fields(rows, buffered, "error", "schema.live.chat-stream.error", spec)

        bad = client.post(f"{base}/chat/completions", json={"model": model, "messages": []})
        rows.append(validate_http_response(bad, kind="error", check="schema.live.chat.error",
                                           expected_status=400, spec=spec))
        _with_fields(rows, bad, "error", "schema.live.chat.error", spec)

        if include_images:
            image = client.post(f"{base}/images/generations", json={
                "model": model, "prompt": "A yellow ceramic mug on a blue table.",
                "n": 1, "size": "256x256", "response_format": "b64_json",
            })
            rows.append(validate_http_response(image, kind="images", check="schema.live.images",
                                               expected_status=200, spec=spec))
            _with_fields(rows, image, "images", "schema.live.images", spec)
    return rows


def _api_key(name: str) -> str | None:
    """Read the key from ``name``; the retired SEVEN_DS_API_KEY still works, with a warning."""
    key = os.environ.get(name)
    if not key and name == "CHORD_API_KEY" and os.environ.get("SEVEN_DS_API_KEY"):
        sys.stderr.write("warning: SEVEN_DS_API_KEY is deprecated; set CHORD_API_KEY\n")
        key = os.environ["SEVEN_DS_API_KEY"]
    return key


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {name: sum(row["verdict"] == name for row in rows)
              for name in ("pass", "fail", "unsupported-rejected", "declared-noop", "SILENT")}
    return result("schema.summary", "*", "fail" if counts["fail"] or counts["SILENT"] else "pass", counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base ending in /v1")
    parser.add_argument("--model", default="chord-1-poly")
    parser.add_argument("--api-key-env", default="CHORD_API_KEY")
    parser.add_argument("--include-images", action="store_true", help="make one real image request")
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args(argv)
    try:
        rows = run_live(args.base_url, args.model, _api_key(args.api_key_env), args.include_images, args.timeout)
    except Exception as exc:
        rows = [result("schema.runner", "*", "fail", {"error_type": type(exc).__name__, "message": str(exc)[:300]})]
    rows.append(_summary(rows))
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    return 1 if rows[-1]["verdict"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
