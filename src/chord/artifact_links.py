"""Signing and verification for browser-loadable artifact links."""

from __future__ import annotations

import hashlib
import hmac
import time

from fastapi import Request


def signature(key: str, artifact_id: str, expires: int) -> str:
    return hmac.new(key.encode(), f"{artifact_id}:{expires}".encode(), hashlib.sha256).hexdigest()


def is_signed_request(request: Request, key: str, max_ttl_s: int) -> bool:
    """Whether this is a valid, bounded-lifetime GET for one artifact."""
    parts = request.url.path.split("/")
    if request.method != "GET" or len(parts) != 4 or parts[:3] != ["", "v1", "artifacts"]:
        return False
    try:
        expires = int(request.query_params.get("expires", ""))
    except ValueError:
        return False
    now = time.time()
    if expires < now or expires > now + max_ttl_s:
        return False
    supplied = request.query_params.get("sig", "")
    expected = signature(key, parts[3], expires)
    # Bytes, never strs: compare_digest raises TypeError on a non-ASCII str,
    # and `sig` is a query parameter -- an UNAUTHENTICATED caller could reach
    # this line with anything and 500 the auth middleware (review
    # 2026-09-22, #5). errors="replace" cannot raise, and a replaced sig can
    # never equal the hex digest, so a wrong-charset signature is simply
    # invalid, which is the only honest answer it has.
    return hmac.compare_digest(
        supplied.encode("utf-8", "replace"), expected.encode())
