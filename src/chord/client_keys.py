"""Parse and match Chord's direct-client bearer credentials."""

from __future__ import annotations

import hmac
import json
import re


_CLIENT_ID = re.compile(r"[a-z][a-z0-9-]{1,63}\Z")
_BEARER = re.compile(r"[A-Za-z0-9._~-]{32,256}\Z")
_RESERVED = frozenset({"legacy", "local", "signed-link"})


def parse_client_keys(raw: str, legacy_key: str) -> tuple[tuple[str, str], ...]:
    """Return (client id, token) pairs; never include secret text in an error."""
    if not raw:
        return ()

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        if len({name for name, _ in pairs}) != len(pairs):
            raise ValueError
        return dict(pairs)

    try:
        parsed = json.loads(raw, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, TypeError, ValueError):
        raise ValueError("CHORD_CLIENT_KEYS_JSON must be a JSON object") from None
    if not isinstance(parsed, dict):
        raise ValueError("CHORD_CLIENT_KEYS_JSON must be a JSON object")
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for client_id, token in parsed.items():
        if (not isinstance(client_id, str) or _CLIENT_ID.fullmatch(client_id) is None
                or client_id in _RESERVED):
            raise ValueError("CHORD_CLIENT_KEYS_JSON has an invalid client id")
        if not isinstance(token, str) or _BEARER.fullmatch(token) is None:
            raise ValueError("CHORD_CLIENT_KEYS_JSON has an invalid bearer token")
        if token == legacy_key or token in seen:
            raise ValueError("CHORD_CLIENT_KEYS_JSON contains a repeated bearer token")
        seen.add(token)
        pairs.append((client_id, token))
    return tuple(pairs)


def identify_client(authorization: str, legacy_key: str,
                    clients: tuple[tuple[str, str], ...], *, legacy_enabled: bool = True) -> str | None:
    """Identify a direct client without echoing or short-circuiting on its key."""
    if not legacy_key and not clients:
        return "local"
    supplied = authorization.removeprefix("Bearer ")
    has_bearer = authorization.startswith("Bearer ")
    found: str | None = None
    if legacy_key:
        matches = hmac.compare_digest(supplied.encode("utf-8", "replace"), legacy_key.encode())
        if has_bearer and matches and legacy_enabled:
            found = "legacy"
    for client_id, token in clients:
        matches = hmac.compare_digest(supplied.encode("utf-8", "replace"), token.encode())
        if has_bearer and matches:
            found = client_id
    return found
