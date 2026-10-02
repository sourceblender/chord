"""Remove model-authored Markdown image targets from generated-image captions.

The facade supplies the actual attachment. Plain chat never uses this filter.
Only complete inline/reference image markup is removed; ordinary caption text
streams immediately, apart from an undecided image-markup prefix.
"""


def _close(text: str, start: int, opening: str, closing: str) -> int | None:
    depth, escaped = 0, False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if not depth:
                return index
    return None


class _MarkdownImages:
    def __init__(self):
        self.pending = ""
        self.removed = 0

    def feed(self, text: str, *, final: bool = False) -> str:
        self.pending += text
        output = []
        while self.pending:
            start = self.pending.find("![")
            if start < 0:
                keep = 1 if not final and self.pending.endswith("!") else 0
                output.append(self.pending[:-keep] if keep else self.pending)
                self.pending = self.pending[-keep:] if keep else ""
                break
            output.append(self.pending[:start])
            self.pending = self.pending[start:]
            alt_end = _close(self.pending, 1, "[", "]")
            if alt_end is None or alt_end + 1 == len(self.pending):
                if final:
                    output.append(self.pending)
                    self.pending = ""
                break
            opening = self.pending[alt_end + 1]
            if opening not in "([":
                output.append(self.pending[:alt_end + 1])
                self.pending = self.pending[alt_end + 1:]
                continue
            end = _close(self.pending, alt_end + 1, opening, ")" if opening == "(" else "]")
            if end is None:
                if final:
                    output.append(self.pending)
                    self.pending = ""
                break
            self.pending = self.pending[end + 1:]
            self.removed += 1
        return "".join(output)



def _tool_object(text: str) -> bool:
    import json
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return False
    return isinstance(obj, dict) and (
        {"action", "action_input"} <= obj.keys()
        or {"tool", "arguments"} <= obj.keys()
        or {"action", "args"} <= obj.keys())



class _ActionJSON:
    """Buffer candidate JSON objects/fences until they can be classified.

    Preserve malformed/incomplete text at EOF. This is not a general code or
    JSON sanitizer; only the two explicit top-level tool-call shapes qualify.
    """
    def __init__(self):
        self.pending = ""
        self.removed = 0

    def feed(self, text: str, *, final: bool = False) -> str:
        import json
        self.pending += text
        out = []
        while self.pending:
            starts = [p for p in (self.pending.find("{"), self.pending.find("```")) if p >= 0]
            if not starts:
                keep = min(2, len(self.pending) - len(self.pending.rstrip("`"))) if not final else 0
                out.append(self.pending[:-keep] if keep else self.pending)
                self.pending = self.pending[-keep:] if keep else ""
                break
            start = min(starts)
            out.append(self.pending[:start])
            self.pending = self.pending[start:]
            if self.pending.startswith("```"):
                end = self.pending.find("```", 3)
                if end < 0:
                    if not final:
                        break
                    out.append(self.pending[:3])
                    self.pending = self.pending[3:]
                    continue
                block = self.pending[3:end]
                if block[:4].lower() == "json":
                    block = block[4:]
                remove = _tool_object(block.strip())
                length = end + 3
            else:
                try:
                    _, length = json.JSONDecoder().raw_decode(self.pending)
                except json.JSONDecodeError:
                    if not final:
                        break
                    # This opening brace cannot currently form a complete
                    # object. Preserve it, then scan later candidates too.
                    out.append(self.pending[0])
                    self.pending = self.pending[1:]
                    continue
                remove = _tool_object(self.pending[:length])
            if remove:
                self.removed += 1
            else:
                out.append(self.pending[:length])
            self.pending = self.pending[length:]
        if final:
            out.append(self.pending)
            self.pending = ""
        return "".join(out)


class _BracketNarration:
    def __init__(self):
        self.pending = ""
        self.removed = 0

    def feed(self, text: str, *, final: bool = False) -> str:
        import re
        import json
        self.pending += text
        out = []
        while self.pending:
            start = self.pending.find("[")
            brace = self.pending.find("{")
            if brace >= 0 and (start < 0 or brace < start):
                out.append(self.pending[:brace])
                self.pending = self.pending[brace:]
                try:
                    _, end = json.JSONDecoder().raw_decode(self.pending)
                except ValueError:
                    if not final:
                        break
                    end = 1
                out.append(self.pending[:end])
                self.pending = self.pending[end:]
                continue
            if start < 0:
                out.append(self.pending)
                self.pending = ""
                break
            out.append(self.pending[:start])
            self.pending = self.pending[start:]
            end = _close(self.pending, 0, "[", "]")
            if end is None:
                if final:
                    out.append(self.pending[:1])
                    self.pending = self.pending[1:]
                    continue
                break
            # Delay one character to distinguish narration from link text.
            if end + 1 == len(self.pending) and not final:
                break
            is_link = self.pending[end + 1:end + 2] in ("(", "[")
            narration = re.match(r"\[(?:image|picture)(?::\s*|\s+of\s+)", self.pending, re.I)
            if narration and not is_link:
                self.removed += 1
            else:
                out.append(self.pending[:end + 1])
            self.pending = self.pending[end + 1:]
        return "".join(out)


class CaptionImages:
    def __init__(self):
        self.actions = _ActionJSON()
        self.images = _MarkdownImages()
        self.narration = _BracketNarration()

    @property
    def removed(self):
        return self.images.removed

    @property
    def actions_removed(self):
        return self.actions.removed

    @property
    def narration_removed(self):
        return self.narration.removed

    def feed(self, text: str, *, final: bool = False) -> str:
        return self.narration.feed(self.images.feed(self.actions.feed(text, final=final), final=final), final=final)

def clean_caption(text: str) -> tuple[str, int]:
    images = CaptionImages()
    return images.feed(text, final=True), images.removed


# Pretend-delivery narration on turns that deliver NOTHING (2026-09-13, #82):
# asked for a voice clip the service can't make, told so, the model still wrote
# "(audio clip playing)". Removes only a short bracketed or parenthesised aside
# that names a media thing AND claims it plays, is attached or is sent,
# including surrounding * or _ emphasis. Ordinary parentheses stream through.
# Both lists match WHOLE WORDS only: "represents", "display", "record player"
# and "profile" are ordinary words, not claims (#87). The emoji have no
# word boundary, so they are a separate alternative.
# This is a bounded filter, not an honesty check: it catches the short aside
# shape only. Pretend delivery written as ordinary prose ("I just sent it!")
# passes through; the unavailable note is what keeps the reply honest there.
_MEDIA = (r"\b(?:audio|voice(?:\s*(?:note|message|memo))?|sound|song|music|recording|clip|picture|photo|image"
          r"|selfie|video|file|attachment)s?\b")
_CLAIM = r"(?:\b(?:play|plays|playing|attached|attaching|sent|sending|sends|listen|listening)\b|🔊|🎵|🎶|▶)"
# A negated or unavailable statement is the honest version and always stays:
# "(audio is not playing)", "(the image was not sent)" (#87).
_NEGATED = __import__("re").compile(
    r"\b(?:not|no|never|none|isn't|wasn't|won't|can't|cannot|couldn't|didn't|doesn't|unable|unavailable|failed|"
    r"fails|without)\b|n't\b", __import__("re").I)
_FAKE = __import__("re").compile(
    r"[*_]{0,2}[\(\[]\s*(?=[^)\]]{0,60}" + _MEDIA + r")(?=[^)\]]{0,60}" + _CLAIM + r")[^)\]\n]{1,60}[\)\]][*_]{0,2}",
    __import__("re").I)


class FakeMediaNarration:
    """Stream-safe: holds text only from an opening '(' or '[' until it closes
    (at most 80 chars), then drops it if it is pretend-delivery narration."""

    MAX = 80

    def __init__(self):
        self.pending = ""
        self.removed = 0

    def feed(self, text: str, *, final: bool = False) -> str:
        self.pending += text
        out = []
        while self.pending:
            m = __import__("re").search(r"[*_]{0,2}[\(\[]", self.pending)
            if not m:
                keep = len(self.pending) - len(self.pending.rstrip("*_")) if not final else 0
                out.append(self.pending[:len(self.pending) - keep])
                self.pending = self.pending[len(self.pending) - keep:]
                break
            out.append(self.pending[:m.start()])
            self.pending = self.pending[m.start():]
            close = __import__("re").search(r"[\)\]][*_]{0,2}", self.pending)
            if close is None:
                if not final and len(self.pending) < self.MAX:
                    break
                out.append(self.pending[0])
                self.pending = self.pending[1:]
                continue
            if not final and close.end() == len(self.pending):
                break  # more closing * or _ may still arrive in the next chunk (#87)
            span = self.pending[:close.end()]
            if _FAKE.fullmatch(span) and not _NEGATED.search(span):
                self.removed += 1
            else:
                out.append(span)
            self.pending = self.pending[close.end():]
        return "".join(out)


# A claim that something was delivered, on a turn that delivers nothing (#80
# incident pair, 2026-09-13): asked a clarifying question after a turn that did
# carry a picture, the model copied that turn's shape, "Here you go — Ava at the
# gym…", 20 of 20 times on the chat model, 7 of 10 with a stronger note. A prompt
# can't hold it, so on such turns the finished reply is checked whole and, if it
# ACTIVELY claims a current delivery, replaced by a plain line with the real
# question. Not counted (#105): a negated claim in the same clause ("I'm
# not sending it now"), a quoted one ("you saw 'Here you go' before"), and
# "here's your" + anything that isn't media ("here's your question back").
# Clause, not sentence (#105): "No problem, here's your picture" and
# "Couldn't resist, here it is" are lies with an idiom in front, so a comma,
# semicolon, colon or dash ends a bare negation's reach. A hyphen does not ("not
# re-sending it" stays negated). A negation OF delivery ("I can't send it yet,
# here it is in words") still covers the whole sentence.
# Bounded: these phrases only. A claim worded some other way passes.
_RE = __import__("re")
_MEDIA_OBJ = r"(?:picture|photo|image|pic|selfie|clip|video|drawing|render|voice (?:note|message))"
_DELIVERY_CLAIM = _RE.compile(
    r"\b(?:here you go|here it is|here'?s your " + _MEDIA_OBJ + r"|here'?s (?:the|a|my) " + _MEDIA_OBJ + r"|"
    r"i(?:'ve| have)? (?:made|drawn|drew|rendered|attached|sent) (?:it|this one|that one|her|him|them|one for you|"
    r"you (?:a|one|this|her))\b|i pictured (?:her|him|them|it)|it'?s attached|attached (?:below|here|above)|"
    r"sending (?:it|this) (?:over|now|your way))\b", _RE.I)
# "Here's Ava putting in the work…": a capitalised name straight after here's
# (case-sensitive on purpose: "here's the thing" and "here's what I think" pass).
_PRESENTING_NAME = _RE.compile(r"\b[Hh]ere'?s (?!I\b)[A-Z][a-z]+\b")
# Markdown quoting is quoting too (#105): `inline code` and "> " blockquote lines.
_QUOTED = _RE.compile(r"\"[^\"\n]{0,200}\"|“[^”\n]{0,200}”|‘[^’\n]{0,200}’|(?<!\w)'[^'\n]{1,200}'(?!\w)"
                      r"|`[^`\n]{1,200}`|(?m:^[ \t]*>[^\n]*)")
_NEGATION = _RE.compile(r"\b(?:not|never|no|haven'?t|hasn'?t|didn'?t|won'?t|can'?t|cannot|isn'?t|wasn'?t|aren'?t)\b|n't\b", _RE.I)
_DENIES_DELIVERY = _RE.compile(
    r"(?:\b(?:not|never|no|cannot)\b|n't\b)(?:\s+\w+){0,2}?\s+"
    r"(?:make|made|making|send|sent|sending|draw|drawn|drew|render|rendered|attach|attached|ready|done|finished|started)\b",
    _RE.I)


def claims_delivery(text: str) -> bool:
    text = _QUOTED.sub(lambda m: " " * len(m.group(0)), text or "")      # quoted words are not a delivery claim
    for pattern in (_DELIVERY_CLAIM, _PRESENTING_NAME):
        for m in pattern.finditer(text):
            sentence_start = max(text.rfind(c, 0, m.start()) for c in ".!?\n") + 1
            clause_start = max(text.rfind(c, 0, m.start()) for c in ".!?\n,;:—–") + 1
            if not (_NEGATION.search(text[clause_start:m.start()])
                    or _DENIES_DELIVERY.search(text[sentence_start:m.start()])):
                return True
    return False
