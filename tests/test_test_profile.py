"""The suite uses a complete operator override or its generic test fixtures."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("chord_root_conftest", ROOT / "conftest.py")
assert _spec is not None and _spec.loader is not None
root_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(root_conftest)
select, ProfileError = root_conftest.select_profile, root_conftest.ProfileError


@pytest.fixture
def paths(tmp_path: Path):
    return tmp_path / "fixture-manifest.yaml", tmp_path / "fixture-registry.yaml"


def test_no_override_selects_the_fixture_pair(paths) -> None:
    manifest, registry = paths
    assert select(manifest, registry, {}) == ("fixture", str(manifest), str(registry))


@pytest.mark.parametrize("name", ["CHORD_MANIFEST", "CHORD_REGISTRY"])
def test_one_override_without_its_partner_refuses(paths, name) -> None:
    manifest, registry = paths
    with pytest.raises(ProfileError, match="without its partner"):
        select(manifest, registry, {name: "/elsewhere.yaml"})


def test_an_operator_pair_is_used_whole(paths) -> None:
    manifest, registry = paths
    env = {"CHORD_MANIFEST": "/op/manifest.yaml", "CHORD_REGISTRY": "/op/registry.yaml"}
    assert select(manifest, registry, env) == ("external", "/op/manifest.yaml", "/op/registry.yaml")
