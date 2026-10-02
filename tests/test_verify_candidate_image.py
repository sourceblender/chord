"""A candidate tag or false source label must never reach the runtime verifier."""

import json
import sys

import pytest

from scripts import verify_candidate_image


IMAGE = "ghcr.io/sourceblender/chord-candidates@sha256:" + "a" * 64
TREE = "b" * 40


def test_candidate_requires_private_digest_and_tree_before_docker(tmp_path, monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("Docker must not be called")

    monkeypatch.setattr(verify_candidate_image, "run", unexpected)
    for bad_image in ("ghcr.io/sourceblender/chord-candidates:latest",
                      "ghcr.io/sourceblender/chord@sha256:" + "a" * 64):
        with pytest.raises(ValueError, match="pinned private image"):
            verify_candidate_image.verify(bad_image, TREE, tmp_path / "wheel", tmp_path / "verifier.py")
    with pytest.raises(ValueError, match="pinned private image"):
        verify_candidate_image.verify(IMAGE, "not-a-tree", tmp_path / "wheel", tmp_path / "verifier.py")


def test_candidate_checks_source_label_before_python_runs(tmp_path, monkeypatch):
    calls = []

    def fake_run(*args, **_kwargs):
        calls.append(args)
        return json.dumps([{"Config": {"Labels": {verify_candidate_image.LABEL: "c" * 40},
                                        "User": "chord"}}]).encode()

    monkeypatch.setattr(verify_candidate_image, "run", fake_run)
    with pytest.raises(ValueError, match="source-tree label differs"):
        verify_candidate_image.verify(IMAGE, TREE, tmp_path / "wheel", tmp_path / "verifier.py")
    assert calls == [("docker", "image", "inspect", IMAGE)]


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_candidate_refuses_nonregular_retained_wheel(tmp_path, monkeypatch, kind):
    calls = []

    def fake_run(*args):
        calls.append(args)
        if args[:3] == ("docker", "image", "inspect"):
            return json.dumps([{"Config": {
                "Labels": {verify_candidate_image.LABEL: TREE}, "User": "chord",
                "Entrypoint": None, "Cmd": ["python", "-m", "chord"],
                "WorkingDir": "/app", "Env": ["PATH=/app/.venv/bin:/usr/bin"],
            }}]).encode()
        if args[:2] == ("docker", "create"):
            return b"container-id\n"
        if args[:2] == ("docker", "cp"):
            wheel = tmp_path / "wheel" / "chord-0.1.0.whl"
            if kind == "directory":
                wheel.mkdir()
            else:
                target = tmp_path / "outside.whl"
                target.write_bytes(b"outside")
                wheel.symlink_to(target)
            return b""
        if args[:2] == ("docker", "rm"):
            return b""
        raise AssertionError(f"candidate code must not run: {args}")

    monkeypatch.setattr(verify_candidate_image, "run", fake_run)
    with pytest.raises(ValueError, match="one regular wheel"):
        verify_candidate_image.verify(IMAGE, TREE, tmp_path / "wheel", tmp_path / "verifier.py")
    assert not any(call[0] == sys.executable for call in calls)
    assert not any(call[:2] == ("docker", "run") for call in calls)
    assert calls[-1] == ("docker", "rm", "container-id")


def test_candidate_verification_uses_runner_python_without_starting_image(tmp_path, monkeypatch):
    calls = []
    verifier = tmp_path / "verifier.py"
    verifier.write_text("raise AssertionError('stub is not actually run')")

    def fake_run(*args):
        calls.append(args)
        if args[:3] == ("docker", "image", "inspect"):
            return json.dumps([{"Config": {
                "Labels": {verify_candidate_image.LABEL: TREE}, "User": "chord",
                "Entrypoint": None, "Cmd": ["python", "-m", "chord"],
                "WorkingDir": "/app", "Env": ["PATH=/app/.venv/bin:/usr/bin"],
            }}]).encode()
        if args[:2] == ("docker", "create"):
            return b"container-id\n"
        if args[:2] == ("docker", "cp"):
            if args[2].endswith(":/opt/chord-wheel/."):
                (tmp_path / "wheel" / "chord-0.1.0.whl").write_bytes(b"wheel")
                (tmp_path / "wheel" / ".gitignore").write_bytes(b"*")
            return b""
        if args[:2] == ("docker", "rm") or args[0] == sys.executable:
            return b""
        raise AssertionError(args)

    monkeypatch.setattr(verify_candidate_image, "run", fake_run)
    wheel = verify_candidate_image.verify(IMAGE, TREE, tmp_path / "wheel", verifier)
    assert wheel.name == "chord-0.1.0.whl"
    assert not any(call[:2] == ("docker", "run") for call in calls)
    assert any(call[:4] == (sys.executable, "-I", "-S", str(verifier)) for call in calls)


def test_retained_wheel_allows_only_uv_marker_beside_regular_wheel(tmp_path):
    wheel = tmp_path / "chord-0.1.0.whl"
    wheel.write_bytes(b"wheel")
    marker = tmp_path / ".gitignore"
    marker.write_bytes(b"*")
    assert verify_candidate_image.retained_wheel(tmp_path) == wheel

    marker.write_bytes(b"ignore other files")
    with pytest.raises(ValueError, match="uv ignore"):
        verify_candidate_image.retained_wheel(tmp_path)
    marker.write_bytes(b"*")
    (tmp_path / "other.py").write_bytes(b"pass")
    with pytest.raises(ValueError, match="one regular wheel"):
        verify_candidate_image.retained_wheel(tmp_path)
