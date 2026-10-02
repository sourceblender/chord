"""Responses objects and events are built from src/chord/response_fields.json,
which must equal what the pinned spec derives (qa/conformance/derive_response_fields.py)."""
import json
from pathlib import Path

import pytest

from qa.conformance.derive_response_fields import derive
from chord import responses as R

ROOT = Path(__file__).resolve().parents[1]


def test_the_vendored_field_lists_equal_the_pinned_spec():
    assert json.loads((ROOT / "src" / "chord" / "response_fields.json").read_text()) == derive()


def test_an_undeclared_field_is_dropped_at_every_level():
    event = {"type": "response.completed", "sequence_number": 3, "debug": 1, "response": {
        "id": "resp_1", "object": "response", "status": "completed", "internal_route": "image", "output": [
            {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed", "_index": 0,
             "content": [{"type": "output_text", "text": "hi", "annotations": [], "logprobs": [], "tokens": 5}]},
            {"id": "fc_1", "type": "function_call", "call_id": "c", "name": "f", "arguments": "{}", "status": "completed", "parsed": {}}]}}
    built = R.construct_event(event)
    dumped = json.dumps(built)
    assert not any(k in dumped for k in ("debug", "internal_route", "_index", "tokens", "parsed"))
    assert built["response"]["output"][0]["content"][0]["text"] == "hi"
    assert built["response"]["output"][1]["arguments"] == "{}"


def test_the_196_leak_cannot_come_back():
    built = R.construct_event({"type": "response.function_call_arguments.done", "sequence_number": 1, "item_id": "fc",
                               "output_index": 0, "arguments": "{}", "name": "get_weather"})
    assert "name" not in built


def test_an_event_type_the_spec_does_not_know_is_a_bug():
    with pytest.raises(ValueError, match="not in the pinned spec"):
        R.construct_event({"type": "response.vendor_progress", "sequence_number": 0})


def test_a_part_inside_an_event_is_built_too():
    built = R.construct_event({"type": "response.content_part.done", "sequence_number": 2, "item_id": "m", "output_index": 0,
                               "content_index": 0, "part": {"type": "output_text", "text": "x", "annotations": [], "logprobs": [], "extra": 1}})
    assert "extra" not in built["part"]
