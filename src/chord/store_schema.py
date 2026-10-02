"""Persistent-store versions declared by each Chord release.

Version 0 is the historical unversioned schema. Version 1 records the current
additive migrations without changing their table layout. A release receipt can
publish these declarations for the operator's pre-swap compatibility gate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StoreSchema:
    path: str  # relative to CHORD_DATA_DIR
    readable_min: int
    readable_max: int
    write_version: int


STORES = {
    "responses": StoreSchema("responses.sqlite3", 0, 1, 1),
    "videos": StoreSchema("videos/index.sqlite3", 0, 1, 1),
}
