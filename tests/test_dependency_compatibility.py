"""Regression gates for temporary dependency compatibility constraints."""

from __future__ import annotations

import subprocess
import sys


def test_starlette_testclient_import_is_deprecation_clean() -> None:
    """Remove the AnyIO upper bound only when released Starlette passes this."""
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::DeprecationWarning",
            "-c",
            "import starlette.testclient",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
