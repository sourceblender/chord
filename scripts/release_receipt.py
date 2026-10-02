"""Attest one reviewed export, wheel, and candidate image by immutable digest.

The output is canonical UTF-8 JSON: sorted keys, compact separators, no
trailing newline. Its SHA-256 is the pin consumers record. This script never
builds or publishes an image; the caller supplies the digest it observed after
the candidate push and checks the image labels separately.

The receipt attests Chord package files and declared startup settings. The
Python interpreter and third-party dependencies rely on the pinned base-image
digest and uv.lock used by the reviewed build; they are not member-attested by
this receipt.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import posixpath
import re
import subprocess
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path


DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
HEX = re.compile(r"[0-9a-f]{40}\Z")
INTERFACE = {
    "user": "chord",
    "venv": "/app/.venv",
    "internal_port": 8711,
    "preflight_modules": ["chord.comfy_preflight", "chord.backend_preflight"],
}
STORES = {
    "responses": {"path": "responses.sqlite3", "readable_min": 0, "readable_max": 1, "write_version": 1},
    "videos": {"path": "videos/index.sqlite3", "readable_min": 0, "readable_max": 1, "write_version": 1},
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(data: dict[str, object]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def source_files(root: Path) -> dict[str, str]:
    product = root / "src" / "chord"
    if not product.is_dir() or product.is_symlink():
        raise ValueError("export has no src/chord product tree")
    entries = list(product.rglob("*"))
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in entries):
        raise ValueError("export product tree contains a link or special file")
    files = sorted(p for p in entries if p.is_file())
    if not files or any("__pycache__" in p.parts for p in files):
        raise ValueError("export product tree is empty or contains generated/link files")
    return {p.relative_to(root).as_posix(): sha256(p.read_bytes()) for p in files}


def wheel_record(wheel: Path, version: str) -> dict[str, dict[str, object]]:
    if not wheel.name.startswith(f"chord-{version}-") or wheel.suffix != ".whl":
        raise ValueError("wheel filename does not match the product version")
    with zipfile.ZipFile(wheel) as archive:
        names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        records = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(records) != 1 or len(names) != len(set(names)):
            raise ValueError("wheel has no unique member list and RECORD")
        record_name = records[0]
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
        members: dict[str, dict[str, object]] = {}
        for row in rows:
            if len(row) != 3:
                raise ValueError("wheel RECORD row has the wrong shape")
            path, digest, size = row
            parts = Path(path).parts
            if (not path or path.startswith("/") or "\\" in path or ".." in parts
                    or posixpath.normpath(path) != path or path in members):
                raise ValueError("wheel RECORD contains an unsafe or duplicate path")
            if path == record_name:
                if digest or size:
                    raise ValueError("wheel RECORD self-entry must have an empty hash and size")
                members[path] = {"sha256": None, "size": None}
                continue
            if not digest.startswith("sha256=") or not size.isdecimal():
                raise ValueError(f"wheel RECORD has no SHA-256 and size: {path}")
            raw = archive.read(path)
            encoded = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
            if digest != f"sha256={encoded}" or int(size) != len(raw):
                raise ValueError(f"wheel RECORD disagrees with member: {path}")
            members[path] = {"sha256": sha256(raw), "size": len(raw)}
        if set(members) != set(names):
            raise ValueError("wheel RECORD does not enumerate exactly the wheel members")
        if not any(path.startswith("chord/") for path in members):
            raise ValueError("wheel has no Chord package files")
        metadata_names = [path for path in members if path.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1 or Path(metadata_names[0]).parent != Path(record_name).parent:
            raise ValueError("wheel has no metadata beside its RECORD")
        metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))
        if metadata.get("Name") != "chord" or metadata.get("Version") != version:
            raise ValueError("wheel metadata disagrees with the product version")
        return members


def export_tree(root: Path, product_paths: set[str]) -> str:
    """Hash the exact exported files as a Git tree without adding a commit."""
    result = subprocess.run(["git", "-C", str(root), "write-tree"], capture_output=True,
                            text=True, check=False)
    if result.returncode or not HEX.fullmatch(result.stdout.strip()):
        raise ValueError("export must be a staged Git tree")
    tree = result.stdout.strip()
    tracked = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
                             capture_output=True, text=True, check=False)
    if tracked.returncode or tracked.stdout:
        raise ValueError("export working tree differs from its staged tree")
    indexed = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--", "src/chord"],
                             capture_output=True, check=False)
    if indexed.returncode or set(indexed.stdout.decode().rstrip("\0").split("\0")) != product_paths:
        raise ValueError("export product files differ from its Git tree")
    return tree


def receipt(root: Path, wheel: Path, image_digest: str) -> dict[str, object]:
    if not DIGEST.fullmatch(image_digest):
        raise ValueError("candidate image must be pinned by SHA-256 digest")
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    members = wheel_record(wheel, version)
    files = source_files(root)
    for path, digest in files.items():
        installed_path = path.removeprefix("src/")
        if members.get(installed_path, {}).get("sha256") != digest:
            raise ValueError(f"wheel does not carry the exported product file: {path}")
    expected_product = {path.removeprefix("src/") for path in files}
    actual_product = {path for path in members if path.startswith("chord/")}
    if actual_product != expected_product:
        raise ValueError("wheel Chord members differ from the exported product files")
    dist_info = f"chord-{version}.dist-info/"
    if any(not path.startswith(("chord/", dist_info)) for path in members):
        raise ValueError("wheel contains files outside Chord and its dist-info")
    return {
        "receipt_version": 1,
        "version": version,
        "source_tree": export_tree(root, set(files)),
        "source_files": files,
        "wheel": {"sha256": sha256(wheel.read_bytes()), "record": members},
        "image": {"digest": image_digest, "interface": INTERFACE},
        "stores": STORES,
        "build_inputs": {"uv_lock_sha256": sha256((root / "uv.lock").read_bytes())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = canonical(receipt(args.export.resolve(), args.wheel.resolve(), args.image_digest))
    if args.output.exists():
        raise ValueError("release receipt output already exists")
    args.output.write_bytes(data)
    print(f"receipt sha256:{sha256(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
