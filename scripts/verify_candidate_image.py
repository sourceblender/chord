"""Verify a digest-pinned candidate image and extract its exact retained wheel.

The caller later issues a release receipt from that wheel and the reviewed
export. This gate reads image metadata and copies files from a created
container. The runner's Python verifies those files; no candidate executable
is started.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


IMAGE = re.compile(r"ghcr\.io/sourceblender/chord-candidates@sha256:[0-9a-f]{64}\Z")
TREE = re.compile(r"[0-9a-f]{40}\Z")
LABEL = "org.sourceblender.chord.source-tree"


def run(*args: str) -> bytes:
    return subprocess.run(args, capture_output=True, check=True).stdout


def retained_wheel(directory: Path) -> Path:
    entries = {path.name: path for path in directory.iterdir()}
    # uv build creates this marker in its output directory. It is not part of
    # the wheel, but the Dockerfile copies the whole /dist directory.
    ignore = entries.pop(".gitignore", None)
    if ignore is not None and (ignore.is_symlink() or not ignore.is_file()
                               or ignore.read_bytes() != b"*"):
        raise ValueError("candidate wheel directory has an unexpected uv ignore file")
    wheels = list(entries.values())
    if (len(wheels) != 1 or not wheels[0].name.startswith("chord-")
            or wheels[0].suffix != ".whl" or wheels[0].is_symlink()
            or not wheels[0].is_file()):
        raise ValueError("candidate retained wheel directory is not exactly one regular wheel")
    return wheels[0]


def verify(image: str, source_tree: str, wheel_dir: Path, verifier: Path) -> Path:
    if not IMAGE.fullmatch(image) or not TREE.fullmatch(source_tree):
        raise ValueError("candidate must be a pinned private image and a Git tree hash")
    if wheel_dir.exists():
        raise ValueError("wheel output directory already exists")
    metadata = json.loads(run("docker", "image", "inspect", image))[0]
    config = metadata["Config"]
    if config.get("Labels", {}).get(LABEL) != source_tree:
        raise ValueError("candidate image source-tree label differs from reviewed export")
    if config.get("User") != "chord":
        raise ValueError("candidate image runtime user differs from release interface")
    if config.get("Entrypoint") not in (None, []):
        raise ValueError("candidate image has an unexpected entrypoint")
    if config.get("Cmd") != ["python", "-m", "chord"]:
        raise ValueError("candidate image has an unexpected startup command")
    if config.get("WorkingDir") != "/app":
        raise ValueError("candidate image has an unexpected working directory")
    environment = config.get("Env", [])
    if any(item.startswith("PYTHONPATH=") for item in environment):
        raise ValueError("candidate image sets PYTHONPATH")
    if not any(item.startswith("PATH=/app/.venv/bin:") for item in environment):
        raise ValueError("candidate image does not start with the installed venv")
    container = run("docker", "create", image).decode().strip()
    if not container:
        raise ValueError("docker create returned no container id")
    try:
        wheel_dir.mkdir(parents=True)
        run("docker", "cp", f"{container}:/opt/chord-wheel/.", str(wheel_dir))
        wheel = retained_wheel(wheel_dir)
        with tempfile.TemporaryDirectory(prefix="chord-candidate-") as temporary:
            app = Path(temporary) / "app"
            app.mkdir()
            run("docker", "cp", f"{container}:/app/.", str(app))
            run(sys.executable, "-I", "-S", str(verifier.resolve()),
                str(wheel.resolve()), str(app / ".venv"), "--offline-app", str(app))
        return wheel
    finally:
        run("docker", "rm", container)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    args = parser.parse_args()
    wheel = verify(args.image, args.source_tree, args.wheel_dir, args.verifier)
    print(wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
