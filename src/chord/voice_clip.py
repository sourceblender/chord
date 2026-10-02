"""Voice clips we used to append to replies as an <audio> block (#82b, retired
2026-09-16: the public wire is the spec, and a voice reply is message.audio,
returned when modalities asks for it).

Conversations from before carry those blocks in assistant history. A client
that sends one back gets it replaced by a short marker, so the base64 doesn't
ride along on every later turn.
"""
from __future__ import annotations

import re

MARKER = "[voice clip you sent earlier]"
BLOCK = re.compile(r"\n*<audio controls>\ndata:audio/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=\s]*\n</audio>\n*")


def strip(text: str) -> str:
    return BLOCK.sub(f"\n\n{MARKER}", text).rstrip() if "<audio controls>" in text else text
