#!/usr/bin/env python3
"""Derive the Responses wire field lists from the pinned spec, for
src/chord/response_fields.json (the service can't import qa at runtime).
tests/test_response_fields_drift.py keeps the vendored file equal to this output."""
import json
import sys
from pathlib import Path

SPEC = Path(__file__).resolve().parent / "spec" / "openapi.json"
ITEMS = {"message": "OutputMessage", "function_call": "FunctionToolCall", "web_search_call": "WebSearchToolCall",
         "image_generation_call": "ImageGenToolCall", "reasoning": "ReasoningItem", "compaction": "CompactionBody"}
PARTS = {"output_text": "OutputTextContent", "refusal": "RefusalContent", "reasoning_text": "ReasoningTextContent"}


def derive() -> dict:
    schemas = json.loads(SPEC.read_text())["components"]["schemas"]

    def props(name: str) -> list[str]:
        s = schemas[name]
        out = dict(s.get("properties", {}))
        for part in s.get("allOf", []):
            ref = part.get("$ref")
            out.update((schemas[ref.rsplit("/", 1)[1]] if ref else part).get("properties", {}))
        return sorted(out)

    events = {}
    for ref in schemas["ResponseStreamEvent"]["anyOf"]:
        name = ref["$ref"].rsplit("/", 1)[1]
        enum = schemas[name].get("properties", {}).get("type", {}).get("enum") or []
        for kind in enum:
            events[kind] = props(name)
    return {"spec": "openai-openapi 4bb21ba (pinned)", "response": props("Response"), "events": dict(sorted(events.items())),
            "items": {k: props(v) for k, v in ITEMS.items()}, "parts": {k: props(v) for k, v in PARTS.items()}}


if __name__ == "__main__":
    json.dump(derive(), sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
