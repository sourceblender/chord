"""Test integrity: the code under test must be THIS checkout's code (#141).

The shared .venv has an editable install of chord pointing at one
checkout. A run from any other checkout (a worktree, the deploy gate's copy)
without PYTHONPATH=src silently imports that other checkout, and goes green or
red on the wrong code. On 2026-09-14 the shared checkout sat on a commit on no
branch, so such runs tested stale code. This aborts the session before any test
is collected, naming both paths. It compares against this file's own location,
not the venv, so it holds in main, in every worktree, and in the gate's copy.
"""

import os
from pathlib import Path

import pytest

EXPECTED_SRC = Path(__file__).resolve().parent / "src"
ALL_LANES_MANIFEST = Path(__file__).resolve().parent / "tests" / "fixtures" / "manifest-all-lanes.yaml"
ALL_LANES_REGISTRY = Path(__file__).resolve().parent / "tests" / "fixtures" / "registry-all-lanes.yaml"


class ProfileError(Exception):
    """The manifest/registry pair cannot be chosen without guessing."""


def select_profile(fixture_manifest: Path, fixture_registry: Path, environ) -> tuple[str, str, str]:
    """Choose the (manifest, registry) pair the suite runs against, as one unit.

    The default test pair enables every lane with placeholder model names.
    An operator may supply another pair explicitly. The public package's
    text-only default is tested separately (tests/test_public_profile.py).
    Refuse one override without its partner rather than mixing profiles.
    """
    env_manifest, env_registry = environ.get("CHORD_MANIFEST"), environ.get("CHORD_REGISTRY")
    if bool(env_manifest) != bool(env_registry):
        set_name = "CHORD_MANIFEST" if env_manifest else "CHORD_REGISTRY"
        raise ProfileError(f"{set_name} is set without its partner; set both or neither")
    if env_manifest and env_registry:
        return ("external", env_manifest, env_registry)
    return ("fixture", str(fixture_manifest), str(fixture_registry))


try:
    PROFILE, _manifest, _registry = select_profile(
        ALL_LANES_MANIFEST, ALL_LANES_REGISTRY, os.environ)
    PROFILE_ERROR = None
    # Set before any test imports chord, which reads these at import time.
    os.environ["CHORD_MANIFEST"], os.environ["CHORD_REGISTRY"] = _manifest, _registry
except ProfileError as exc:
    PROFILE, PROFILE_ERROR = None, str(exc)


def pytest_configure(config):
    if PROFILE_ERROR:
        pytest.exit(f"test profile: {PROFILE_ERROR}", returncode=4)

    import chord

    actual = Path(chord.__file__).resolve()
    if not actual.is_relative_to(EXPECTED_SRC):
        pytest.exit(
            "test integrity (#141): chord is imported from another checkout.\n"
            f"  expected under: {EXPECTED_SRC}\n"
            f"  actually from:  {actual}\n"
            "  Run with PYTHONPATH=src from this checkout's root.",
            returncode=4,
        )
