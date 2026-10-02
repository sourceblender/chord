"""The pinned spec lists every Responses operation again with ?beta=true. Each
alias must behave as its plain operation: same status, same body shape, same
store effects."""
from test_responses import MODEL, make, strict
from test_responses_compact import Summarizing
from qa.conformance.schema import validate_payload


def test_every_beta_alias_is_the_plain_operation(tmp_path):
    deps, client, sdk = make(tmp_path, Summarizing())
    created = client.post("/v1/responses?beta=true", json={"model": MODEL, "input": "hello"})
    assert created.status_code == 200
    strict(created.json(), "response")
    assert validate_payload(created.json(), kind="beta-response", fields="strict")["verdict"] == "pass"
    rid = created.json()["id"]
    got = client.get(f"/v1/responses/{rid}?beta=true")
    assert got.status_code == 200 and got.json() == created.json()
    items = client.get(f"/v1/responses/{rid}/input_items?beta=true")
    assert items.status_code == 200 and items.json() == client.get(f"/v1/responses/{rid}/input_items").json()
    assert validate_payload(items.json(), kind="beta-response-items", fields="strict")["verdict"] == "pass"
    count = client.post("/v1/responses/input_tokens?beta=true", json={"model": MODEL, "input": "hello"})
    assert count.status_code == 200 and count.json()["object"] == "response.input_tokens"
    assert validate_payload(count.json(), kind="beta-response-input-tokens", fields="strict")["verdict"] == "pass"
    compacted = client.post("/v1/responses/compact?beta=true", json={"model": MODEL, "input": "My name is Ava."})
    assert compacted.status_code == 200 and compacted.json()["object"] == "response.compaction"
    assert validate_payload(compacted.json(), kind="beta-response-compaction", fields="strict")["verdict"] == "pass"
    not_bg = client.post(f"/v1/responses/{rid}/cancel?beta=true")
    assert not_bg.status_code == 400 and not_bg.json() == client.post(f"/v1/responses/{rid}/cancel").json()
    assert client.delete(f"/v1/responses/{rid}?beta=true").status_code == 200
    assert client.get(f"/v1/responses/{rid}?beta=true").status_code == 404
