"""The source boundary that continues to run in a fresh public checkout."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_boundary", ROOT / "scripts" / "public_boundary.py")
assert spec is not None and spec.loader is not None
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)


def test_private_artifacts_and_special_files_refuse(tmp_path: Path) -> None:
    files = []
    for name in ("evidence/record.json", "deploy/compose.yaml", "qa/redteam/run.json", ".env"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n")
        files.append(path)
    link = tmp_path / "link.txt"
    link.symlink_to(files[0])
    files.append(link)
    findings = boundary.scan(tmp_path, files)
    assert len(findings) == len(files)
    assert all("private artifact path" in finding for finding in findings[:-1])
    assert "non-regular file" in findings[-1]


def test_credential_and_personal_paths_refuse_without_echoing_values(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    markers = [
        "op:" + "//private/item",
        "-----BEGIN " + "PRIVATE KEY-----",
        "-----BEGIN " + "ENCRYPTED PRIVATE KEY-----",
        "ghp_" + "A" * 30,
        "AKIA" + "A" * 16,
        "ASIA" + "A" * 16,
        "sk-" + "a" * 24,
        "sk-proj-" + "a" * 24,
        "/Users/" + "example/project",
    ]
    path.write_text("\n".join(markers) + "\n")
    findings = boundary.scan(tmp_path, [path])
    assert len(findings) == len(markers)
    assert [finding.split(":", 2)[1] for finding in findings] == [str(i) for i in range(1, len(markers) + 1)]
    assert all(marker not in "\n".join(findings) for marker in markers)


def test_api_key_rule_requires_a_left_boundary(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("disk-usage-threshold-limits\n" + "sk-" + "a" * 24 + "\n")
    assert boundary.scan(tmp_path, [path]) == ["sample.txt:2: API key"]


def test_public_domain_suffixes_are_not_blocked(tmp_path: Path) -> None:
    path = tmp_path / "samples.txt"
    path.write_text("hello.world\nplayer.world\nexample.world\nbackend.internal\n")
    assert boundary.scan(tmp_path, [path]) == []


def test_undecodable_file_refuses(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff")
    assert boundary.scan(tmp_path, [path]) == ["bad.txt:0: unreadable file"]


def test_plain_export_ignores_generated_files_but_checks_other_files(tmp_path: Path) -> None:
    for name in (".venv/lib/module.py", "src/chord/__pycache__/module.pyc",
                 ".pytest_cache/v/cache", "src/chord/extra.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n")
    assert [path.relative_to(tmp_path).as_posix() for path in boundary.checkout_files(tmp_path)] == [
        "src/chord/extra.py"
    ]


def test_public_file_set_passes() -> None:
    if (ROOT / "scripts" / "scrub_check.py").exists():
        gate_spec = importlib.util.spec_from_file_location("scrub_check", ROOT / "scripts" / "scrub_check.py")
        assert gate_spec is not None and gate_spec.loader is not None
        gate = importlib.util.module_from_spec(gate_spec)
        gate_spec.loader.exec_module(gate)
        files = gate.public_files(ROOT)
    else:
        files = boundary.checkout_files(ROOT)
    assert boundary.scan(ROOT, files) == []
