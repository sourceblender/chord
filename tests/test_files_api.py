"""The Files API (#142): upload, list, retrieve, delete, download. Official SDK,
every object checked strictly against the pinned spec."""
import time

import openai
import pytest

from test_responses import make, strict

JSONL = b'{"custom_id": "a", "method": "POST", "url": "/v1/chat/completions", "body": {}}\n'


def test_an_upload_past_the_cap_is_refused_before_it_is_stored(tmp_path, monkeypatch):
    monkeypatch.setattr("chord.files_api.MAX_BYTES", 8)
    monkeypatch.setattr("chord.files_api.BODY_SLACK", 0)
    _, client, _sdk = make(tmp_path)
    r = client.post("/v1/files", data={"purpose": "user_data"},
                    files={"file": ("x.txt", b"0123456789abcdef", "text/plain")})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "file_too_large"
    assert list((tmp_path / "files").iterdir()) == []


def test_upload_retrieve_download_delete_round_trip(tmp_path):
    deps, client, sdk = make(tmp_path)
    f = sdk.files.create(file=("notes.txt", b"hello files"), purpose="user_data")
    raw = client.get(f"/v1/files/{f.id}").json()
    strict(raw, "file")
    assert f.id.startswith("file-") and (raw["bytes"], raw["filename"], raw["purpose"]) == (11, "notes.txt", "user_data")
    assert "expires_at" not in raw                                              # kept until deleted
    assert sdk.files.content(f.id).read() == b"hello files"
    gone = client.delete(f"/v1/files/{f.id}").json()
    strict(gone, "file-deleted")
    assert gone == {"id": f.id, "object": "file", "deleted": True}
    assert not (tmp_path / "files" / f.id).exists()                            # a real delete, bytes too
    for call in (lambda: sdk.files.retrieve(f.id), lambda: sdk.files.content(f.id), lambda: sdk.files.delete(f.id)):
        with pytest.raises(openai.NotFoundError):
            call()


def test_batch_files_expire_after_30_days_and_expires_after_sets_it(tmp_path):
    _, _, sdk = make(tmp_path)
    b = sdk.files.create(file=("in.jsonl", JSONL), purpose="batch")
    assert b.expires_at - b.created_at == 30 * 24 * 3600
    e = sdk.files.create(file=("x.txt", b"x"), purpose="user_data", expires_after={"anchor": "created_at", "seconds": 3600})
    assert e.expires_at - e.created_at == 3600


def test_an_expired_file_is_gone_for_every_reader(tmp_path, monkeypatch):
    _, client, sdk = make(tmp_path)
    f = sdk.files.create(file=("x.txt", b"x"), purpose="user_data", expires_after={"anchor": "created_at", "seconds": 3600})
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 3601)
    assert client.get(f"/v1/files/{f.id}").status_code == 404
    assert client.get("/v1/files").json()["data"] == []
    assert not (tmp_path / "files" / f.id).exists()


def test_list_orders_filters_and_pages(tmp_path):
    _, client, sdk = make(tmp_path)
    a = sdk.files.create(file=("a.txt", b"a"), purpose="user_data")
    b = sdk.files.create(file=("b.jsonl", JSONL), purpose="batch")
    c = sdk.files.create(file=("c.txt", b"c"), purpose="assistants")
    listing = client.get("/v1/files").json()
    strict(listing, "file-list")
    assert [f["id"] for f in listing["data"]] == [c.id, b.id, a.id]             # default desc
    assert [f.id for f in sdk.files.list(purpose="batch")] == [b.id]
    page = client.get("/v1/files", params={"order": "asc", "limit": 1, "after": a.id}).json()
    assert ([f["id"] for f in page["data"]], page["has_more"]) == ([b.id], True)
    assert [f.id for f in sdk.files.list(order="asc", limit=1)] == [a.id, b.id, c.id]   # the SDK follows the cursor


@pytest.mark.parametrize("form, files, param", [
    ({"purpose": "evals"}, {"file": ("x", b"x")}, "purpose"),
    ({"purpose": "fun"}, {"file": ("x", b"x")}, "purpose"),
    ({}, {"file": ("x", b"x")}, "purpose"),
    ({"purpose": "user_data"}, {}, "file"),
    ({"purpose": "user_data", "file": "not a part"}, {}, "file"),
    ({"purpose": "user_data"}, {"file": ("x", b"")}, "file"),
    ({"purpose": "user_data", "colour": "blue"}, {"file": ("x", b"x")}, "colour"),
    ({"purpose": "user_data", "expires_after[anchor]": "created_at"}, {"file": ("x", b"x")}, "expires_after[seconds]"),
    ({"purpose": "user_data", "expires_after[anchor]": "now", "expires_after[seconds]": "3600"}, {"file": ("x", b"x")}, "expires_after[anchor]"),
    ({"purpose": "user_data", "expires_after[anchor]": "created_at", "expires_after[seconds]": "3599"}, {"file": ("x", b"x")}, "expires_after[seconds]"),
    ({"purpose": "user_data", "expires_after[anchor]": "created_at", "expires_after[seconds]": "2592001"}, {"file": ("x", b"x")}, "expires_after[seconds]"),
])
def test_bad_uploads_are_refused_by_name_and_keep_nothing(tmp_path, form, files, param):
    _, client, _ = make(tmp_path)
    parts = [(k, (None, v)) for k, v in form.items()] + list(files.items())   # always multipart, even with no file part
    r = client.post("/v1/files", files=parts)
    assert r.status_code == 400 and r.json()["error"]["param"] == param, r.text
    strict(r.json(), "error")
    assert client.get("/v1/files").json()["data"] == []
    assert [p.name for p in (tmp_path / "files").iterdir()] == []               # no stray bytes, no temp file


def test_an_oversized_upload_is_refused_and_leaves_nothing(tmp_path, monkeypatch):
    from chord import files_api
    monkeypatch.setattr(files_api, "MAX_BYTES", 10)
    _, client, _ = make(tmp_path)
    r = client.post("/v1/files", data={"purpose": "user_data"}, files={"file": ("big", b"x" * 11)})
    assert r.status_code == 400 and r.json()["error"]["code"] == "file_too_large"
    assert list((tmp_path / "files").iterdir()) == []


def test_a_json_body_is_refused(tmp_path):
    _, client, _ = make(tmp_path)
    r = client.post("/v1/files", json={"purpose": "user_data"})
    assert r.status_code == 400 and set(r.json()) == {"error"}


def test_bad_list_parameters_are_refused(tmp_path):
    _, client, _ = make(tmp_path)
    for params, param in (({"limit": 0}, "limit"), ({"limit": 10001}, "limit"), ({"order": "up"}, "order"),
                          ({"after": "file-nope"}, "after")):
        r = client.get("/v1/files", params=params)
        assert r.status_code == 400 and r.json()["error"]["param"] == param


# #209: an OSError mid-write (disk full) escaped as a raw 500 and orphaned the temp file.
@pytest.mark.parametrize("fail_at", ["write", "replace"])
def test_a_failing_disk_is_an_enveloped_500_and_leaves_nothing(tmp_path, monkeypatch, fail_at):
    import errno
    import os as real_os
    from chord import files_api

    _, client, _ = make(tmp_path)

    def boom(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")

    if fail_at == "replace":
        monkeypatch.setattr(files_api.os, "replace", boom)
    else:
        real_fdopen = real_os.fdopen

        class Full:
            def __init__(self, f): self.f = f
            def __enter__(self): return self
            def __exit__(self, *a): self.f.close()
            def write(self, b): boom()

        monkeypatch.setattr(files_api.os, "fdopen", lambda fd, mode: Full(real_fdopen(fd, mode)))
    r = client.post("/v1/files", data={"purpose": "user_data"}, files={"file": ("x.txt", b"hello")})
    assert r.status_code == 500 and r.json()["error"]["code"] == "storage_failed" and r.json()["error"]["type"] == "server_error"
    strict(r.json(), "error")
    assert list((tmp_path / "files").iterdir()) == []                      # no temp file, no bytes
    monkeypatch.undo()
    assert client.get("/v1/files").json()["data"] == []                    # and no row


def test_a_third_concurrent_upload_is_refused_with_429(tmp_path, monkeypatch):
    """The capped body stays resident in process memory for the handler's whole
    life, so the slots bound aggregate memory: with none free, the route answers
    the API 429 envelope BEFORE reading a byte (review 2026-09-22: N
    concurrent uploads were N x 513 MB with nothing bounding N)."""
    import asyncio

    from chord import files_api
    monkeypatch.setattr(files_api, "_upload_slots", asyncio.Semaphore(0))
    _, client, _sdk = make(tmp_path)
    r = client.post("/v1/files", data={"purpose": "user_data"},
                    files={"file": ("x.txt", b"hi", "text/plain")})
    assert r.status_code == 429, r.text
    assert r.json()["error"]["code"] == "rate_limit_exceeded"
    strict(r.json(), "error")
    assert list((tmp_path / "files").iterdir()) == []


def test_load_stored_file_refuses_past_the_consumer_cap(tmp_path):
    """A stored file may be 512 MB; a consumer's cap is its own. The refusal is
    measured BEFORE the bytes are read, and an uncapped load still works."""
    from chord import files_api
    from chord.responses_store import ResponseStore

    store = ResponseStore(tmp_path / "responses.sqlite3")
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "file-x").write_bytes(b"0123456789abcdef0123")
    store.put_file({"id": "file-x", "object": "file", "bytes": 20, "created_at": int(time.time()),
                    "filename": "x.bin", "purpose": "user_data", "status": "processed"})
    with pytest.raises(files_api.StoredFileTooLarge):
        files_api.load_stored_file(store, files_dir, "file-x", max_bytes=8)
    file, data = files_api.load_stored_file(store, files_dir, "file-x")
    assert file["id"] == "file-x" and len(data) == 20
