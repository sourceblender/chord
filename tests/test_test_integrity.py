"""#141: a run that imports another checkout's chord aborts before any test."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(pythonpath: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": pythonpath}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_skeleton.py"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)


def test_importing_another_checkout_aborts_the_session_before_any_test(tmp_path):
    other = tmp_path / "other-checkout" / "src"
    shutil.copytree(ROOT / "src" / "chord", other / "chord")  # a clean second checkout
    run = _run(str(other))
    out = run.stdout + run.stderr
    assert run.returncode == 4, out
    assert f"expected under: {ROOT / 'src'}" in out and f"actually from:  {other / 'chord'}" in out, out
    assert " passed" not in out and " failed" not in out  # aborted before collection


def test_this_checkouts_own_source_is_accepted():
    run = _run(str(ROOT / "src"))
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
