"""Check a public checkout for private artifacts and credential-shaped content.

This script ships with Chord. It uses general patterns so the public checker
does not carry an operator's names, endpoints, or private identifier list.
"""

from __future__ import annotations

import re
import stat
import subprocess
import sys
from pathlib import Path


DENIED_COMPONENTS = {"evidence", "deploy", "harness", "secrets"}
DENIED_PAIRS = {("qa", "redteam"), ("qa", "certification")}
DENIED_NAMES = {".env", "id_rsa", "id_ed25519"}
DENIED_SUFFIXES = {".pem", ".p12", ".pfx"}
GENERATED_COMPONENTS = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
GENERATED_SUFFIXES = {".pyc", ".pyo"}

# Keep markers split where needed so checking this script does not mistake its
# own patterns for credentials. A changed or broader rule needs a test.
CONTENT_RULES = {
    "secret-manager URI": re.compile(r"op:" + r"//", re.I),
    "private-key block": re.compile(r"-----BEGIN\s+(?:RSA\s+|OPENSSH\s+|EC\s+|DSA\s+|ENCRYPTED\s+)?PRIVATE\s+KEY-----"),
    "GitHub token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "AWS access key": re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    "API key": re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}"),
    "personal home path": re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+(?:/|\b)"),
}


def checkout_files(root: Path) -> list[Path]:
    """Use the Git index in a checkout; an exported tree has no index."""
    if (root / ".git").exists():
        result = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True)
        names = [Path(name.decode("utf-8")) for name in result.stdout.split(b"\0") if name]
        return [root / name for name in names]
    return [path for path in root.rglob("*")
            if not any(part in GENERATED_COMPONENTS | {".git"} for part in path.relative_to(root).parts)
            and path.suffix not in GENERATED_SUFFIXES
            and (path.is_symlink() or not path.is_dir())]


def scan(root: Path, files: list[Path]) -> list[str]:
    findings: list[str] = []
    for path in files:
        rel = path.relative_to(root)
        name = rel.as_posix()
        parts = rel.parts
        lowered = tuple(part.lower() for part in parts)
        if (any(part in DENIED_COMPONENTS for part in lowered)
                or any(pair == lowered[i:i + 2] for pair in DENIED_PAIRS for i in range(len(parts) - 1))
                or path.name.lower() in DENIED_NAMES
                or path.suffix.lower() in DENIED_SUFFIXES):
            findings.append(f"{name}:0: private artifact path")
            continue
        try:
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                findings.append(f"{name}:0: non-regular file")
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            findings.append(f"{name}:0: unreadable file")
            continue
        for lineno, line in enumerate(lines, 1):
            for label, pattern in CONTENT_RULES.items():
                if pattern.search(line):
                    findings.append(f"{name}:{lineno}: {label}")
    return findings


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python scripts/public_boundary.py CHECKOUT", file=sys.stderr)
        return 2
    root = Path(argv[0]).resolve()
    if not root.is_dir():
        print("public boundary: checkout is not a directory", file=sys.stderr)
        return 2
    findings = scan(root, checkout_files(root))
    for finding in findings:
        print(finding)
    print(f"public boundary: {'FAIL' if findings else 'PASS'} ({len(findings)} findings)")
    return bool(findings)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
