"""Job progress: what a specialist reports while it works.

A specialist reports only WHAT HAPPENED, as a stage name, at the moment it
happened. Since 2026-09-16 it is traced and never shown: Chat Completions has
no progress event, and a line in the reply or an Open WebUI status event is not
part of the spec's wire (the external interface is the spec). The
Responses API's own progress events are where it belongs when that API lands.

Replies from before then can still carry the lines at their start, so a
replayed assistant message has them removed (strip_leading).

Stage names must be true when they fire. "submitting" means the request is
about to be sent, not that anything is rendering; a stronger word needs a
real signal from the tool.
"""

from __future__ import annotations

import re

# Stages a specialist may report, and the line that used to be shown for each.
LINES: dict[str, dict[str, str]] = {
    "image": {
        "preparing": "Preparing the image…",
        "submitting": "Sending the render request…",
        "retrying": "The image request didn't start; trying again…",
    },
    "search": {
        "searching": "Searching the web…",
    },
}


def render(text: str) -> str:
    return f"_{text}_\n\n"


_RENDERED = sorted({render(t) for stages in LINES.values() for t in stages.values()}, key=len, reverse=True)
_LEADING = re.compile(r"^(?:" + "|".join(re.escape(r) for r in _RENDERED) + r")+")


def strip_leading(content: str) -> str:
    """Remove our progress lines from the start of a replayed assistant message,
    so a later turn doesn't feed them back to the model as its own words. Only
    exact rendered lines at the very start are removed; the model never wrote them."""
    return _LEADING.sub("", content, count=1)
