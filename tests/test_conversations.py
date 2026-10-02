"""The Conversations API (#146/#142, Phase 4) and `conversation` on Responses:
a conversation is ordered items kept until deleted; a response in it reads them
as context and appends its own input and output. Official SDK, strict schema."""
import json

import openai
import pytest

from test_responses import MODEL, make, strict
from test_client_tools_own_the_turn import Calling


def test_crud_through_the_sdk_strict(tmp_path):
    deps, client, sdk = make(tmp_path)
    conv = sdk.conversations.create(metadata={"topic": "tea"}, items=[{"type": "message", "role": "user", "content": "hi"}])
    raw = client.get(f"/v1/conversations/{conv.id}").json()
    strict(raw, "conversation")
    assert conv.id.startswith("conv_") and raw["metadata"] == {"topic": "tea"}
    updated = sdk.conversations.update(conv.id, metadata={"topic": "coffee"})
    assert updated.metadata == {"topic": "coffee"}
    deleted = client.delete(f"/v1/conversations/{conv.id}").json()
    strict(deleted, "conversation-deleted")
    with pytest.raises(openai.NotFoundError):
        sdk.conversations.retrieve(conv.id)


def test_items_add_list_get_delete(tmp_path):
    deps, client, sdk = make(tmp_path)
    conv = sdk.conversations.create()
    added = sdk.conversations.items.create(conv.id, items=[{"type": "message", "role": "user", "content": t} for t in ("a", "b", "c")])
    assert [i.content[0].text for i in added.data] == ["a", "b", "c"]
    page = client.get(f"/v1/conversations/{conv.id}/items?limit=2").json()
    strict(page, "conversation-items")
    assert [i["content"][0]["text"] for i in page["data"]] == ["c", "b"] and page["has_more"] is True
    one = client.get(f"/v1/conversations/{conv.id}/items/{page['data'][0]['id']}").json()
    assert one["content"][0]["text"] == "c"
    after = client.delete(f"/v1/conversations/{conv.id}/items/{one['id']}").json()
    strict(after, "conversation")
    left = sdk.conversations.items.list(conv.id, order="asc")
    assert [i.content[0].text for i in left.data] == ["a", "b"]
    assert client.get(f"/v1/conversations/{conv.id}/items/{one['id']}").status_code == 404


def test_a_response_in_a_conversation_reads_it_and_appends_to_it(tmp_path):
    deps, client, sdk = make(tmp_path)
    conv = sdk.conversations.create(items=[{"type": "message", "role": "user", "content": "my name is Ava"}])
    first = sdk.responses.create(model=MODEL, conversation=conv.id, input="hello", instructions="Be brief.")
    assert first.conversation.id == conv.id
    strict(client.get(f"/v1/responses/{first.id}").json(), "response")
    second = sdk.responses.create(model=MODEL, conversation={"id": conv.id}, input="what is my name?")
    sent = deps.upstream.bodies[1]["messages"]
    assert [(m["role"], m["content"]) for m in sent if m["role"] != "system"] == [
        ("user", "my name is Ava"), ("user", "hello"), ("assistant", "hi there"), ("user", "what is my name?")]
    assert "Be brief." not in json.dumps(sent)
    items = sdk.conversations.items.list(conv.id, order="asc").data
    assert [(i.type, getattr(i, "role", None)) for i in items] == [
        ("message", "user"), ("message", "user"), ("message", "assistant"), ("message", "user"), ("message", "assistant")]
    assert second.output_text == "hi there"


def test_a_streamed_response_appends_too(tmp_path):
    deps, client, sdk = make(tmp_path)
    conv = sdk.conversations.create()
    with sdk.responses.stream(model=MODEL, conversation=conv.id, input="hello") as s:
        for _ in s:
            pass
    assert len(sdk.conversations.items.list(conv.id).data) == 2


def test_function_calls_land_in_the_conversation(tmp_path):
    from test_responses import WEATHER
    deps, client, sdk = make(tmp_path, Calling())
    conv = sdk.conversations.create()
    sdk.responses.create(model=MODEL, conversation=conv.id, input="weather?", tools=[WEATHER])
    types = [i.type for i in sdk.conversations.items.list(conv.id, order="asc").data]
    assert types == ["message", "function_call", "function_call"]


@pytest.mark.parametrize("body,status,param", [
    ({"conversation": "conv_missing"}, 404, "conversation"),
    ({"conversation": "CONV", "previous_response_id": "resp_x"}, 400, "conversation"),
])
def test_conversation_param_errors(tmp_path, body, status, param):
    deps, client, sdk = make(tmp_path)
    conv = sdk.conversations.create()
    body = {k: (conv.id if v == "CONV" else v) for k, v in body.items()}
    r = client.post("/v1/responses", json={"model": MODEL, "input": "x", **body})
    assert r.status_code == status and r.json()["error"]["param"] == param
    assert deps.upstream.bodies == []


@pytest.mark.parametrize("path,body,param", [
    ("/v1/conversations", {"items": [{"type": "item_reference", "id": "x"}]}, "items[0].type"),
    ("/v1/conversations", {"items": [{"type": "message", "role": "user", "content": "x"}] * 21}, "items"),
    ("/v1/conversations", {"metadata": {"k": 1}}, "metadata"),
    ("/v1/conversations", {"bogus": 1}, "bogus"),
])
def test_conversation_refusals(tmp_path, path, body, param):
    deps, client, sdk = make(tmp_path)
    r = client.post(path, json=body)
    assert r.status_code == 400 and r.json()["error"]["param"] == param
    strict(r.json(), "error")


def test_unknown_conversation_everywhere_is_404(tmp_path):
    deps, client, sdk = make(tmp_path)
    for method, path in (("GET", "/v1/conversations/conv_x"), ("POST", "/v1/conversations/conv_x"),
                         ("DELETE", "/v1/conversations/conv_x"), ("GET", "/v1/conversations/conv_x/items"),
                         ("POST", "/v1/conversations/conv_x/items"), ("GET", "/v1/conversations/conv_x/items/msg_x"),
                         ("DELETE", "/v1/conversations/conv_x/items/msg_x")):
        r = client.request(method, path, json={"metadata": {}, "items": []})
        assert r.status_code == 404, (method, path, r.text)


def test_an_empty_page_has_null_cursors_and_nothing_else_off_spec(tmp_path):
    """Conversation item rule (the T03 companions, 2026-09-17): no item, no id; never an invented string."""
    from qa.conformance.schema import Spec, validate_payload
    deps, client, sdk = make(tmp_path)
    conv = sdk.conversations.create()
    page = client.get(f"/v1/conversations/{conv.id}/items").json()
    assert page["data"] == [] and page["first_id"] is None and page["last_id"] is None and page["has_more"] is False
    assert list(sdk.conversations.items.list(conv.id)) == []                     # the official SDK parses it
    row = validate_payload({**page, "first_id": "x", "last_id": "x"}, kind="conversation-items", spec=Spec(), fields="strict")
    assert row["verdict"] == "pass"                                              # the two cursors are the only deviation


def test_compaction_and_image_items_stay_in_the_conversation(tmp_path):
    """Review 2026-09-24 B4: input_to_messages turned a compaction item and an
    image_generation_call into model context but never into stored items, and
    conversation mode appends only the stored items -- so both were gone from
    the second turn on, while previous_response_id chaining kept them."""
    from chord.responses import IMAGE_MARKER
    from test_responses_compact import Summarizing
    deps, client, sdk = make(tmp_path, Summarizing())
    token = sdk.responses.compact(model=MODEL, input="My name is Ava.").output[0].encrypted_content
    conv = sdk.conversations.create()
    image = {"id": "ig_caller", "type": "image_generation_call", "status": "completed", "result": "aGk="}
    first = client.post("/v1/responses", json={"model": MODEL, "conversation": conv.id, "input": [
        {"type": "compaction", "encrypted_content": token}, image, {"role": "user", "content": "hello"}]}).json()
    items = client.get(f"/v1/responses/{first['id']}/input_items?order=asc").json()
    strict(items, "response-items")
    assert [i["type"] for i in items["data"]] == ["compaction", "image_generation_call", "message"]
    listed = client.get(f"/v1/conversations/{conv.id}/items?order=asc").json()
    strict(listed, "conversation-items")
    assert [i["type"] for i in listed["data"]] == ["compaction", "image_generation_call", "message", "message"]

    sdk.responses.create(model=MODEL, conversation=conv.id, input="what is my name?")
    sent = deps.upstream.bodies[-1]["messages"]
    assert any(m["role"] == "system" and "The user's name is Ava" in m["content"] for m in sent), sent
    assert ("assistant", IMAGE_MARKER) in [(m["role"], m["content"]) for m in sent], sent


def test_items_never_outlive_a_deleted_conversation(tmp_path):
    """Review 2026-09-24 B21: append_items did not check the conversation
    exists, so a turn finishing after DELETE left rows no route can reach."""
    from chord.responses_store import ResponseStore
    store = ResponseStore(tmp_path / "responses.sqlite3")
    store.create_conversation("conv_a", {}, [])
    assert store.delete_conversation("conv_a")
    assert store.append_items("conv_a", [{"id": "msg_late", "type": "message"}]) is False
    assert store.conversation_items("conv_a") == []


@pytest.mark.parametrize("second", ["same-request", "later-request", "response-input"])
def test_a_duplicate_item_id_in_a_conversation_is_refused(tmp_path, second):
    """Review 2026-09-24 B21: caller-supplied ids were not unique within a
    conversation; two items with id msg_dup, and deleting one deleted both."""
    deps, client, sdk = make(tmp_path)
    conv = sdk.conversations.create().id
    dup = {"id": "msg_dup", "type": "message", "role": "user", "content": "x"}
    if second == "same-request":
        r = client.post(f"/v1/conversations/{conv}/items", json={"items": [dup, dup]})
    else:
        assert client.post(f"/v1/conversations/{conv}/items", json={"items": [dup]}).status_code == 200
        if second == "later-request":
            r = client.post(f"/v1/conversations/{conv}/items", json={"items": [dup]})
        else:
            r = client.post("/v1/responses", json={"model": MODEL, "conversation": conv, "input": [dup]})
    assert r.status_code == 400, r.text
    strict(r.json(), "error")
    assert "msg_dup" in r.json()["error"]["message"]
    assert deps.upstream.bodies == []
    assert [i["id"] for i in client.get(f"/v1/conversations/{conv}/items").json()["data"]].count("msg_dup") <= 1


def test_an_image_input_item_keeps_the_fields_the_spec_declares(tmp_path):
    """Copilot on #332: the stored image_generation_call kept only id, status,
    result and output_format. The pinned ImageGenToolCall also declares size,
    quality, action, background and revised_prompt; each is kept, and a value
    outside the spec's type or enum is a 400 naming it."""
    deps, client, sdk = make(tmp_path)
    conv = sdk.conversations.create().id
    image = {"id": "ig_full", "type": "image_generation_call", "status": "completed", "result": "aGk=",
             "size": "1024x1536", "quality": "high", "action": "edit", "background": "transparent",
             "output_format": "webp", "revised_prompt": "a teal mug"}
    first = client.post("/v1/responses", json={"model": MODEL, "conversation": conv, "input": [image]})
    assert first.status_code == 200, first.text
    items = client.get(f"/v1/responses/{first.json()['id']}/input_items").json()
    strict(items, "response-items")
    assert items["data"] == [image]
    listed = client.get(f"/v1/conversations/{conv}/items?order=asc").json()
    strict(listed, "conversation-items")
    assert listed["data"][0] == image

    for field, bad in (("quality", "ultra"), ("action", "paint"), ("background", "green"),
                       ("output_format", "gif"), ("size", 1024), ("revised_prompt", 7)):
        r = client.post("/v1/responses", json={"model": MODEL, "input": [{**image, "id": f"ig_{field}", field: bad}]})
        assert r.status_code == 400 and r.json()["error"]["param"] == f"input[0].{field}", (field, r.text)


def test_the_duplicate_id_check_holds_under_the_append_lock(tmp_path):
    """Copilot on #332: the unique-id guarantee lived in the callers' preflight
    reads, and a request checks, awaits the model, then appends -- so two
    requests with the same caller id could both pass the preflight and both
    insert. The locked append refuses the second."""
    from chord.responses import duplicate_item_id
    from chord.responses_store import DuplicateItemId, ResponseStore
    store = ResponseStore(tmp_path / "responses.sqlite3")
    store.create_conversation("conv_a", {}, [])
    dup = {"id": "msg_dup", "type": "message"}
    # Both requests preflight before either appends: both pass.
    assert duplicate_item_id([dup], store.conversation_items("conv_a")) is None
    assert duplicate_item_id([dup], store.conversation_items("conv_a")) is None
    assert store.append_items("conv_a", [dup]) is True
    with pytest.raises(DuplicateItemId) as exc:
        store.append_items("conv_a", [{"id": "msg_new", "type": "message"}, dup])
    assert exc.value.item_id == "msg_dup"
    assert [i["id"] for i in store.conversation_items("conv_a")] == ["msg_dup"]   # nothing of the refused batch
    with pytest.raises(DuplicateItemId):
        store.append_items("conv_a", [{"id": "msg_twice"}, {"id": "msg_twice"}])


@pytest.mark.parametrize("door", ["items", "response", "stream", "background"])
def test_a_duplicate_that_slips_past_the_preflight_is_still_refused(tmp_path, monkeypatch, door):
    """The API boundary maps the locked refusal: a 400 naming the id where the
    answer is not yet sent, a failed response where it is (stream)."""
    import chord.responses as responses
    deps, client, sdk = make(tmp_path)
    conv = sdk.conversations.create().id
    dup = {"id": "msg_dup", "type": "message", "role": "user", "content": "x"}
    assert client.post(f"/v1/conversations/{conv}/items", json={"items": [dup]}).status_code == 200
    monkeypatch.setattr(responses, "duplicate_item_id", lambda items, existing: None)   # the race: preflight saw nothing
    if door == "items":
        r = client.post(f"/v1/conversations/{conv}/items", json={"items": [dup]})
    elif door == "response":
        r = client.post("/v1/responses", json={"model": MODEL, "conversation": conv, "input": [dup]})
    if door in ("items", "response"):
        assert r.status_code == 400 and "msg_dup" in r.json()["error"]["message"], r.text
        strict(r.json(), "error")
    elif door == "background":
        from test_responses_background import wait_for
        with client:   # the lifespan portal keeps the background task alive
            rid = client.post("/v1/responses", json={"model": MODEL, "conversation": conv, "input": [dup],
                                                     "background": True}).json()["id"]
            done = wait_for(sdk, rid, {"completed", "failed"})
            assert done.status == "failed" and "msg_dup" in done.error.message
            strict(client.get(f"/v1/responses/{rid}").json(), "response")
    else:
        from test_responses import events_of
        events = events_of(client, {"model": MODEL, "conversation": conv, "input": [dup]})
        for e in events:
            strict(e, "response-event")
        assert events[-1]["type"] == "response.failed" and "msg_dup" in events[-1]["response"]["error"]["message"]
        stored = client.get(f"/v1/responses/{events[-1]['response']['id']}").json()
        assert stored["status"] == "failed"
    ids = [i["id"] for i in client.get(f"/v1/conversations/{conv}/items").json()["data"]]
    assert ids == ["msg_dup"], ids
