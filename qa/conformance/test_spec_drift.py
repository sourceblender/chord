import json

from conformance.spec_drift import drift, reach, served_operations, main

BASE = {
    "paths": {"/things": {"post": {"requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Req"}}}},
                                   "responses": {"200": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Res"}}}}}}},
              "/other": {"get": {"responses": {"200": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Other"}}}}}}}},
    "components": {"schemas": {
        "Req": {"type": "object", "properties": {"a": {"$ref": "#/components/schemas/Leaf"}}},
        "Res": {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}},
        "Leaf": {"type": "string"},
        "Other": {"type": "object"},
    }},
}


def copy():
    return json.loads(json.dumps(BASE))


def test_reach_is_transitive():
    assert reach(BASE, BASE["paths"]["/things"]["post"]) == {"Req", "Res", "Leaf"}


def test_no_change_is_no_drift():
    assert drift(BASE, copy(), ["POST /things"])["served"] == {}


def test_a_new_required_field_in_a_reached_schema_is_reported():
    new = copy()
    new["components"]["schemas"]["Res"]["required"].append("status")
    entry = drift(BASE, new, ["POST /things"])["served"]["POST /things"]
    assert entry["schemas_changed"] == ["Res"] and entry["new_required"] == {"Res": ["status"]}


def test_a_nested_change_is_reached_and_an_unrelated_one_is_not():
    new = copy()
    new["components"]["schemas"]["Leaf"] = {"type": "string", "enum": ["x"]}
    new["components"]["schemas"]["Other"] = {"type": "array"}
    report = drift(BASE, new, ["POST /things"])
    assert report["served"]["POST /things"]["schemas_changed"] == ["Leaf"]


def test_surface_additions_and_a_served_removal():
    new = copy()
    new["paths"]["/fresh"] = {"get": {}}
    del new["paths"]["/things"]
    report = drift(BASE, new, ["POST /things"])
    assert report["operations_added_upstream"] == ["GET /fresh"]
    assert report["served"]["POST /things"] == {"removed_upstream": True}


def test_served_comes_from_the_profile():
    assert served_operations({"operations": {"POST /a": {"status": "supported"}, "GET /b": {"status": "unsupported-by-design"},
                                             "GET /c": {"status": "experimental"}}}) == ["GET /c", "POST /a"]


def test_the_pinned_spec_against_itself_is_clean(capsys):
    from conformance.spec_drift import PINNED
    assert main(["--candidate", str(PINNED)]) == 0
    assert "No served operation drifted." in capsys.readouterr().out


def test_latest_mode_fetches_the_official_candidate(monkeypatch, capsys):
    from conformance import spec_drift
    pinned = json.loads(spec_drift.PINNED.read_text())
    monkeypatch.setattr(spec_drift, "fetch_latest", lambda: pinned)

    assert main(["--latest"]) == 0
    assert "No served operation drifted." in capsys.readouterr().out
