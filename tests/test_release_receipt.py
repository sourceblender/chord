"""Release receipts bind independently supplied artifact bytes, not filenames."""

import base64
import csv
import hashlib
import io
import json
import os
import subprocess
import zipfile
from dataclasses import asdict

import pytest

from chord.store_schema import STORES as PRODUCT_STORES
from scripts.release_receipt import STORES, canonical, export_tree, receipt, source_files, wheel_record


def _wheel(tmp_path, package=b"original", recorded=b"original", version="0.1.0",
           extra=False, extra_recorded=False, outside_recorded=False, aliased=False):
    wheel = tmp_path / "chord-0.1.0-py3-none-any.whl"
    metadata = f"Metadata-Version: 2.4\nName: chord\nVersion: {version}\n".encode()
    files = {"chord/__init__.py": package, "chord-0.1.0.dist-info/METADATA": metadata}
    if extra_recorded:
        files["chord/evil.py"] = b"extra product code"
    if outside_recorded:
        files["other/__init__.py"] = b"other code"
    if aliased:
        files["chord//alias.py"] = b"aliased code"
    rows = []
    for path, data in files.items():
        hash_data = recorded if path == "chord/__init__.py" else data
        encoded = base64.urlsafe_b64encode(hashlib.sha256(hash_data).digest()).rstrip(b"=").decode()
        rows.append((path, f"sha256={encoded}", str(len(hash_data))))
    rows.append(("chord-0.1.0.dist-info/RECORD", "", ""))
    stream = io.StringIO()
    csv.writer(stream, lineterminator="\n").writerows(rows)
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, data in files.items():
            archive.writestr(path, data)
        archive.writestr(rows[-1][0], stream.getvalue())
        if extra:
            archive.writestr("chord/unrecorded.py", b"surprise")
    return wheel


def test_receipt_bytes_are_canonical_and_have_no_implicit_newline():
    data = canonical({"z": 1, "a": {"b": "é"}})
    assert data == b'{"a":{"b":"\xc3\xa9"},"z":1}'
    assert hashlib.sha256(data).hexdigest() == "60c2f2af8db83a7679bc7d89aa5de2fda0a7f9d6a16ec4aeede09b4dbceb357e"
    assert json.loads(data)["a"]["b"] == "é"


def test_store_declarations_match_the_product_schema():
    assert STORES == {name: asdict(schema) for name, schema in PRODUCT_STORES.items()}


def test_receipt_refuses_a_wheel_that_differs_from_the_reviewed_export(tmp_path):
    export = tmp_path / "export"
    package = export / "src" / "chord"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"original")
    (export / "pyproject.toml").write_text('[project]\nname = "chord"\nversion = "0.1.0"\n')
    (export / "uv.lock").write_text("version = 1\n")
    subprocess.run(["git", "init", "-q", str(export)], check=True)
    subprocess.run(["git", "-C", str(export), "add", "."], check=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "Release Test", "GIT_AUTHOR_EMAIL": "release@example.test",
           "GIT_COMMITTER_NAME": "Release Test", "GIT_COMMITTER_EMAIL": "release@example.test"}
    subprocess.run(["git", "-C", str(export), "commit", "-qm", "fixture"], check=True, env=env)
    wheel = _wheel(tmp_path)
    data = receipt(export, wheel, "sha256:" + "0" * 64)
    assert data["source_files"] == {"src/chord/__init__.py": hashlib.sha256(b"original").hexdigest()}
    (package / "__init__.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not carry the exported product file"):
        receipt(export, wheel, "sha256:" + "0" * 64)
    (package / "__init__.py").write_bytes(b"original")
    with pytest.raises(ValueError, match="Chord members differ"):
        receipt(export, _wheel(tmp_path, extra_recorded=True), "sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="outside Chord and its dist-info"):
        receipt(export, _wheel(tmp_path, outside_recorded=True), "sha256:" + "0" * 64)


def test_wheel_record_checks_actual_member_bytes_and_rejects_unrecorded_files(tmp_path):
    good = _wheel(tmp_path)
    members = wheel_record(good, "0.1.0")
    assert members["chord/__init__.py"] == {"sha256": hashlib.sha256(b"original").hexdigest(), "size": 8}
    assert members["chord-0.1.0.dist-info/RECORD"] == {"sha256": None, "size": None}
    with pytest.raises(ValueError, match="disagrees with member"):
        wheel_record(_wheel(tmp_path, package=b"changed!"), "0.1.0")
    with pytest.raises(ValueError, match="enumerate exactly"):
        wheel_record(_wheel(tmp_path, extra=True), "0.1.0")
    with pytest.raises(ValueError, match="metadata disagrees"):
        wheel_record(_wheel(tmp_path, version="0.2.0"), "0.1.0")
    with pytest.raises(ValueError, match="unsafe or duplicate path"):
        wheel_record(_wheel(tmp_path, aliased=True), "0.1.0")


def test_source_tree_refuses_linked_directories_and_ignored_product_files(tmp_path):
    export = tmp_path / "export"
    product = export / "src" / "chord"
    product.mkdir(parents=True)
    (product / "__init__.py").write_bytes(b"safe")
    (export / ".gitignore").write_text("src/chord/ignored.py\n")
    subprocess.run(["git", "init", "-q", str(export)], check=True)
    subprocess.run(["git", "-C", str(export), "add", "."], check=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "Release Test", "GIT_AUTHOR_EMAIL": "release@example.test",
           "GIT_COMMITTER_NAME": "Release Test", "GIT_COMMITTER_EMAIL": "release@example.test"}
    subprocess.run(["git", "-C", str(export), "commit", "-qm", "fixture"], check=True, env=env)
    outside = tmp_path / "outside"
    outside.mkdir()
    (product / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="link or special file"):
        source_files(export)
    (product / "linked").unlink()
    (product / "ignored.py").write_bytes(b"not in git")
    with pytest.raises(ValueError, match="differ from its Git tree"):
        export_tree(export, set(source_files(export)))
