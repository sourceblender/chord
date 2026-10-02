"""The candidate's Python startup hooks must match the pinned uv build."""

import hashlib

import pytest

from scripts import verify_runtime_wheel


def test_startup_hooks_refuse_extra_pth(tmp_path, monkeypatch):
    module = b"pinned uv hook for the test"
    (tmp_path / "_virtualenv.pth").write_bytes(verify_runtime_wheel.UV_HOOK)
    (tmp_path / "_virtualenv.py").write_bytes(module)
    monkeypatch.setattr(verify_runtime_wheel, "UV_MODULE_SHA256", hashlib.sha256(module).hexdigest())
    verify_runtime_wheel.verify_startup_hooks(tmp_path)

    (tmp_path / "extra.pth").write_text("import os\n")
    with pytest.raises(ValueError, match="startup .pth"):
        verify_runtime_wheel.verify_startup_hooks(tmp_path)


def test_startup_hooks_refuse_customization_and_changed_uv_module(tmp_path, monkeypatch):
    module = b"pinned uv hook for the test"
    (tmp_path / "_virtualenv.pth").write_bytes(verify_runtime_wheel.UV_HOOK)
    (tmp_path / "_virtualenv.py").write_bytes(module)
    monkeypatch.setattr(verify_runtime_wheel, "UV_MODULE_SHA256", hashlib.sha256(module).hexdigest())

    (tmp_path / "sitecustomize.py").write_text("pass\n")
    with pytest.raises(ValueError, match="customization"):
        verify_runtime_wheel.verify_startup_hooks(tmp_path)
    (tmp_path / "sitecustomize.py").unlink()

    (tmp_path / "_virtualenv.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="pinned build"):
        verify_runtime_wheel.verify_startup_hooks(tmp_path)
