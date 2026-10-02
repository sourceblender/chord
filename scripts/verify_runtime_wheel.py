"""Verify that a runtime image executes the wheel it retained at build time.

The installer may reorder RECORD and add its own dist-info entries, so compare
attested members rather than RECORD bytes. Product modules and assets must have
no unrecorded files. The release gate runs this offline on a copy of /app with
the runner's trusted Python; the public image smoke also runs it in the image.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import sys
import zipfile
from pathlib import Path


UV_HOOK = b"import _virtualenv"
UV_MODULE_SHA256 = "cfb3db86aaa53bb62b5ff764970bec2d71c9228590a0ebec57f6ec926cc0bf1a"


def verify_startup_hooks(site: Path) -> None:
    hooks = {path.name: path for path in site.glob("*.pth")}
    if set(hooks) != {"_virtualenv.pth"}:
        raise ValueError("unexpected Python startup .pth files")
    hook = hooks["_virtualenv.pth"]
    module = site / "_virtualenv.py"
    if (hook.is_symlink() or not hook.is_file() or hook.read_bytes() != UV_HOOK
            or module.is_symlink() or not module.is_file()
            or hashlib.sha256(module.read_bytes()).hexdigest() != UV_MODULE_SHA256):
        raise ValueError("uv startup hook differs from the pinned build")
    if any((site / name).exists() for name in ("sitecustomize.py", "usercustomize.py")):
        raise ValueError("unexpected Python customization module")


def members(record: bytes) -> dict[str, tuple[str, str]]:
    rows = list(csv.reader(io.StringIO(record.decode("utf-8"))))
    result: dict[str, tuple[str, str]] = {}
    for path, digest, size in rows:
        if path in result or path.startswith("/") or ".." in Path(path).parts:
            raise ValueError(f"invalid wheel member: {path}")
        result[path] = (digest, size)
    return result


def verify(wheel: Path, venv: Path, *, offline_app: Path | None = None) -> None:
    if wheel.is_dir():
        wheels = list(wheel.glob("chord-*.whl"))
        if len(wheels) != 1:
            raise ValueError("runtime image must retain exactly one Chord wheel")
        wheel = wheels[0]
    app = offline_app if offline_app is not None else Path("/app")
    if (app / "src").exists():
        raise ValueError("source tree shadows the installed wheel")
    # The caller runs Python with -I -S, so neither .pth nor sitecustomize can
    # execute before these checks. Locate the distribution by path, not by
    # importlib metadata or an import of candidate code.
    distributions = list(venv.glob("lib/python*/site-packages/chord-*.dist-info"))
    if len(distributions) != 1:
        raise ValueError("venv has no unique Chord distribution")
    site = distributions[0].parent.resolve()
    if site.name != "site-packages" or not site.is_relative_to(venv.resolve()):
        raise ValueError(f"chord installed outside the venv site-packages: {site}")
    verify_startup_hooks(site)
    with zipfile.ZipFile(wheel) as archive:
        record_names = [n for n in archive.namelist() if n.endswith(".dist-info/RECORD")]
        if len(record_names) != 1:
            raise ValueError("wheel has no unique RECORD")
        expected = members(archive.read(record_names[0]))
    installed = members((site / record_names[0]).read_bytes())
    dist_info = record_names[0].split("/", 1)[0]
    allowed_installer_files = {
        f"{dist_info}/{name}" for name in
        ("INSTALLER", "REQUESTED", "direct_url.json", "uv_cache.json")
    }
    if set(installed) - set(expected) - allowed_installer_files:
        raise ValueError("installed RECORD has unrecognized additions")
    for name, (digest, size) in expected.items():
        if name == record_names[0]:  # RECORD has no self-hash in the wheel.
            continue
        if installed.get(name) != (digest, size):
            raise ValueError(f"installed RECORD changed wheel member: {name}")
        data = (site / name).read_bytes()
        if str(len(data)) != size:
            raise ValueError(f"installed wheel member has wrong size: {name}")
        algorithm, encoded = digest.split("=", 1)
        if algorithm != "sha256":
            raise ValueError(f"unsupported wheel member digest: {name}")
        actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        if actual != encoded:
            raise ValueError(f"installed wheel member changed: {name}")
    actual_product = {
        p.relative_to(site).as_posix() for p in (site / "chord").rglob("*")
        if p.is_file()
    }
    expected_product = {n for n in expected if n.startswith("chord/")}
    if actual_product != expected_product:
        raise ValueError("installed Chord product files differ from the wheel")
    actual_dist_info = {
        p.relative_to(site).as_posix() for p in (site / dist_info).rglob("*")
        if p.is_file()
    }
    if actual_dist_info != {n for n in installed if n.startswith(f"{dist_info}/")}:
        raise ValueError("installed dist-info files differ from RECORD")
    for root in (site / "chord", site / dist_info):
        if any(path.is_symlink() for path in root.rglob("*")):
            raise ValueError(f"installed distribution contains a symlink: {root}")
    if offline_app is None:
        sys.path.insert(0, str(site))
        import chord

        module = Path(chord.__file__).resolve()
        if module != site / "chord" / "__init__.py":
            raise ValueError(f"chord imported outside the verified wheel: {module}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        verify(Path(sys.argv[1]), Path(sys.argv[2]))
    elif len(sys.argv) == 5 and sys.argv[3] == "--offline-app":
        verify(Path(sys.argv[1]), Path(sys.argv[2]), offline_app=Path(sys.argv[4]))
    else:
        raise SystemExit("usage: verify_runtime_wheel.py WHEEL_OR_DIR VENV [--offline-app APP]")
    print("runtime wheel verified")
