"""Shadow-scoring moderation gate — Phase 1 of #121.

The shadow scorer runs alongside the chat door. Its outputs land on the
trace row, never on the wire. Block-mode decisions are a Phase-3 build
gated on the measured FP/FN data this trace is meant to produce. The
tests below pin:

  * the taxonomy is what the design comment said (BLOCKING + ADVISORY);
  * the policy text is pinned by SHA256, so a verdict is bound to it;
  * every flagged verdict carries a quote that IS in the source;
  * a verdict whose quote is not in the source is treated as unflagged
    (the identity-resolver discipline the design explicitly named);
  * the guard failing degrades gracefully -- the chat door must always
    get a verdict, never an exception;
  * shadow scoring on chat turns writes the trace field WITHOUT
    modifying the wire response.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chord import moderation
from chord.moderation import (
    ADVISORY_CATEGORIES, ALL_CATEGORIES, BLOCKING_CATEGORIES,
    POLICY_NAME, CategoryVerdict, Verdict, policy_audit_blob, policy_sha256,
    rule_stub_score, score,
)
from test_skeleton import FakeUpstream, make


# --- taxonomy + policy ---------------------------------------------------------


def test_blocking_categories_match_the_design():
    assert BLOCKING_CATEGORIES == ("minors", "non_consent", "real_person_harm")


def test_advisory_categories_match_the_design():
    assert ADVISORY_CATEGORIES == ("adult_themes", "self_harm")


def test_all_categories_is_blocking_then_advisory():
    """The order is part of the trace contract."""
    assert ALL_CATEGORIES == BLOCKING_CATEGORIES + ADVISORY_CATEGORIES


def test_policy_name_and_pinned_sha256():
    """The policy text + sha are the audit anchor. Pinned so a future
    bump is intentional (the SHA on every trace row changes)."""
    assert POLICY_NAME == "chord-default-1"
    # Recompute and assert it matches what the module returns -- a
    # silent text change is a silent audit-link change.
    expected = moderation.policy_sha256()
    assert policy_sha256() == expected
    # And the policy text itself.
    assert "minor" in moderation.POLICY_TEXT.lower()
    assert "coercion" in moderation.POLICY_TEXT.lower() or "incapacity" in moderation.POLICY_TEXT.lower()
    assert "defamation" in moderation.POLICY_TEXT.lower() or "harm" in moderation.POLICY_TEXT.lower()


def test_policy_audit_blob_lists_everything():
    blob = policy_audit_blob()
    assert blob["name"] == POLICY_NAME
    assert blob["sha256"] == policy_sha256()
    assert blob["blocking_categories"] == list(BLOCKING_CATEGORIES)
    assert blob["advisory_categories"] == list(ADVISORY_CATEGORIES)
    assert blob["all_categories"] == list(ALL_CATEGORIES)


# --- stub scoring --------------------------------------------------------------


@pytest.mark.asyncio
async def test_clean_text_scores_zero_everywhere():
    v = await rule_stub_score("hello there, what is the weather in Paris?")
    assert v.flagged is False
    assert v.blocking_flagged is False
    assert all(not cv.flagged for cv in v.categories)
    assert v.guard_model == "rule-stub-v1"
    assert v.policy_name == POLICY_NAME
    assert v.policy_sha256 == policy_sha256()
    assert v.error is None


@pytest.mark.asyncio
@pytest.mark.parametrize("text,expected_cat,expected_quote_substr", [
    ("the child should not be in sexual contact with adults", "minors", "child"),
    ("she was raped by an acquaintance", "non_consent", "rape"),
    ("someone should kill John Smith for what he did", "real_person_harm", "kill John Smith"),
    ("a porn site I was on earlier", "adult_themes", "porn"),
    ("I want to kill myself tonight", "self_harm", "kill myself"),
])
async def test_every_category_has_a_pattern_that_fires_on_real_text(text, expected_cat, expected_quote_substr):
    v = await rule_stub_score(text)
    assert v.flagged is True
    cats = {cv.category: cv for cv in v.categories}
    assert cats[expected_cat].flagged is True
    assert cats[expected_cat].score == 1.0
    assert cats[expected_cat].quote is not None
    assert expected_quote_substr.lower() in cats[expected_cat].quote.lower(), \
        f"expected a substring of the quote, got {cats[expected_cat].quote!r}"
    # Check-the-evidence: the quote MUST be a substring of the source.
    assert cats[expected_cat].quote in text


@pytest.mark.asyncio
async def test_blocking_categories_set_blocking_flagged():
    """blocking_flagged is the field the future block-mode gate will
    consult. Pin its semantic."""
    v = await rule_stub_score("the child is in sexual danger")
    assert v.flagged is True
    assert v.blocking_flagged is True
    v2 = await rule_stub_score("the porn link is in the description")
    assert v2.flagged is True
    assert v2.blocking_flagged is False  # adult_themes is advisory


@pytest.mark.asyncio
async def test_score_returns_evidence_checked_verdict():
    """Every category verdict in the result is evidence-checked (the
    check-the-evidence discipline ran inside `score`). The stub's
    quotes are always in the source, so this should be a no-op for
    the stub -- but it should still hold."""
    text = "the child is in sexual danger"
    v = await score(text)
    for cv in v.categories:
        if cv.flagged:
            assert cv.quote is not None
            assert cv.quote in text


@pytest.mark.asyncio
async def test_score_rejects_a_quote_not_in_source():
    """A custom guard that returns a quote NOT in the source must be
    downgraded to unflagged by `score`. This is the discipline the
    design comment named."""
    async def lying_guard(text: str) -> Verdict:
        cv = CategoryVerdict(
            category="non_consent", flagged=True, score=1.0,
            quote="fabricated quote not in source",
            reason="the model decided without evidence",
        )
        others = tuple(CategoryVerdict(category=c, flagged=False, score=0.0,
                                        quote=None, reason=None)
                       for c in ALL_CATEGORIES if c != "non_consent")
        return Verdict(
            flagged=True, blocking_flagged=True,
            categories=(cv,) + others,
            guard_model="lying-guard", policy_name=POLICY_NAME,
            policy_sha256=policy_sha256(), latency_ms=0,
        )

    v = await score("harmless input", guard=lying_guard)
    assert v.guard_model == "lying-guard"
    # The fabricated quote was downgraded.
    cats = {cv.category: cv for cv in v.categories}
    assert cats["non_consent"].flagged is False
    assert cats["non_consent"].score == 0.0
    assert cats["non_consent"].quote is None
    assert "evidence" in (cats["non_consent"].reason or "").lower()
    assert v.flagged is False
    assert v.blocking_flagged is False


@pytest.mark.asyncio
async def test_score_fails_open_when_guard_raises():
    """A guard exception must NEVER propagate to the chat door. Phase 1
    is internal-only and the design says fail open for internal traffic.
    The verdict returned has error set and every category unflagged."""

    async def exploding_guard(text: str) -> Verdict:
        raise RuntimeError("guard is down")

    v = await score("anything", guard=exploding_guard)
    assert v.error is not None and "RuntimeError" in v.error
    assert v.guard_model == "<failed>"
    assert v.flagged is False
    assert v.blocking_flagged is False
    assert all(not cv.flagged for cv in v.categories)


@pytest.mark.asyncio
async def test_score_handles_empty_text():
    v = await rule_stub_score("")
    assert v.flagged is False
    assert v.blocking_flagged is False
    assert all(not cv.flagged for cv in v.categories)


# --- trace integration: chat door, no wire change ------------------------------


@pytest.mark.asyncio
async def test_shadow_score_writes_trace_fields_without_modifying_wire(tmp_path):
    """The chat door must accept the request, run shadow scoring, write
    the verdict to the trace, and return a normal 200 response with NO
    moderation field on the wire. The Phase-1 contract."""
    # Use a message that would flag a blocking category so we can see
    # the trace field populated.
    deps, client = make(tmp_path, FakeUpstream())

    # Find the day's trace file.
    day_dir = Path(deps.settings.trace_dir)
    # Send a clean chat completion request.
    r = client.post("/v1/chat/completions", json={
        "model": "chord-1-poly",
        "messages": [{"role": "user", "content": "hello there"}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    # No moderation field on the wire.
    assert "moderation" not in body
    # Trace row exists and carries a moderation_scored entry.
    rows = [json.loads(line) for p in day_dir.rglob("*.jsonl")
            for line in p.read_text().splitlines() if line.strip()]
    assert rows, "no trace row was written"
    row = rows[-1]
    assert row["moderation_scored"] is True, row
    assert row["moderation_input"]["flagged"] is False
    # Output moderation: the FakeUpstream returns "hi there" (text-only),
    # which is clean.
    assert row["moderation_output"]["flagged"] is False
    assert row["moderation_guard"] == "rule-stub-v1"
    assert row["moderation_policy"] == POLICY_NAME


@pytest.mark.asyncio
async def test_shadow_score_records_a_blocking_flag_in_the_trace(tmp_path):
    """The point of shadow scoring is that the trace carries the verdict
    even when the wire says nothing. Send a message that would trigger a
    blocking flag and verify the trace records it -- but the wire is
    unchanged."""
    deps, client = make(tmp_path, FakeUpstream())
    day_dir = Path(deps.settings.trace_dir)

    r = client.post("/v1/chat/completions", json={
        "model": "chord-1-poly",
        "messages": [{"role": "user",
                      "content": "the child is in sexual danger and needs help"}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    # Wire is unchanged. No block, no moderation field, no error.
    assert "moderation" not in body
    # Trace row carries the shadow verdict.
    rows = [json.loads(line) for p in day_dir.rglob("*.jsonl")
            for line in p.read_text().splitlines() if line.strip()]
    row = rows[-1]
    assert row["moderation_scored"] is True
    assert row["moderation_input"]["flagged"] is True
    assert row["moderation_input"]["blocking_flagged"] is True
    assert row["moderation_input"]["categories"]["minors"]["flagged"] is True
    # The quote is kept as a hash, never as text (review 2026-09-24 B5).
    assert row["moderation_input"]["categories"]["minors"]["quote_sha256"] is not None
    # Output is clean (FakeUpstream returns "hi there").
    assert row["moderation_output"]["flagged"] is False


@pytest.mark.asyncio
async def test_a_lone_surrogate_in_a_flagged_quote_still_writes_its_trace(tmp_path):
    """JSON accepts an escaped lone surrogate, the rule's match can include it,
    and a strict UTF-8 encode of that quote for its hash raised
    UnicodeEncodeError while the trace was written -- a 500 for a request the
    chat validator had accepted (Copilot on #335, review 2026-09-24 B5)."""
    deps, client = make(tmp_path, FakeUpstream())
    day_dir = Path(deps.settings.trace_dir)

    raw = ('{"model": "chord-1-poly", "messages": [{"role": "user", '
           '"content": "the child \\ud800 sexual danger"}]}')
    r = client.post("/v1/chat/completions", content=raw,
                    headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text
    rows = [json.loads(line) for p in day_dir.rglob("*.jsonl")
            for line in p.read_text().splitlines() if line.strip()]
    minors = rows[-1]["moderation_input"]["categories"]["minors"]
    assert minors["flagged"] is True
    assert minors["quote_sha256"] is not None


@pytest.mark.asyncio
async def test_shadow_score_does_not_modify_existing_moderations_route(tmp_path):
    """The existing /v1/moderations endpoint is unchanged. Pin that:
    its wire shape is still {flagged: false, all categories false,
    all scores 0.0} regardless of input, and MODERATION_DECIDES_NOTHING
    is still True. Phase 1 is additive, not replacement."""
    from chord import moderations
    assert moderations.MODERATION_DECIDES_NOTHING is True
    deps, client = make(tmp_path, FakeUpstream())
    r = client.post("/v1/moderations", json={"input": "the child is in sexual danger"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "chord-1-poly"
    for result in body["results"]:
        assert result["flagged"] is False
        assert not any(result["categories"].values())
        assert set(result["category_scores"].values()) == {0.0}


# --- a malformed verdict fails open too (review 2026-09-24) ---------------------


def _malformed_verdict() -> Verdict:
    """A verdict a real guard could plausibly return: flagged, with a quote
    that is not a string. Checking its evidence raises TypeError."""
    categories = tuple(
        CategoryVerdict(category=c, flagged=(c == ALL_CATEGORIES[0]), score=1.0 if c == ALL_CATEGORIES[0] else 0.0,
                        quote=42 if c == ALL_CATEGORIES[0] else None,  # type: ignore[arg-type]
                        reason="bad" if c == ALL_CATEGORIES[0] else None)
        for c in ALL_CATEGORIES)
    return Verdict(flagged=True, blocking_flagged=True, categories=categories, guard_model="real-guard",
                   policy_name=POLICY_NAME, policy_sha256=moderation.policy_sha256(), latency_ms=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("returned", [_malformed_verdict, lambda: None], ids=["bad-quote", "not-a-verdict"])
async def test_score_fails_open_when_the_guard_returns_a_malformed_verdict(returned):
    """Checking the evidence ran outside the fail-open boundary, so a guard
    that answered instead of raising, but answered badly, escaped as an
    exception (review 2026-09-24)."""

    async def malformed_guard(text: str) -> Verdict:
        return returned()

    v = await score("anything", guard=malformed_guard)
    assert v.error is not None
    assert v.guard_model == "<failed>"
    assert v.flagged is False and v.blocking_flagged is False
    assert all(not cv.flagged for cv in v.categories)


@pytest.mark.asyncio
async def test_a_malformed_shadow_verdict_never_turns_a_finished_reply_into_a_500(tmp_path, monkeypatch):
    """The chat door scores after the reply is built. A scoring error there must
    be traced and never reach the caller (review 2026-09-24)."""

    async def malformed_guard(text: str) -> Verdict:
        return _malformed_verdict()

    monkeypatch.setattr(moderation, "rule_stub_score", malformed_guard)
    deps, client = make(tmp_path, FakeUpstream())
    r = client.post("/v1/chat/completions", json={
        "model": "chord-1-poly",
        "messages": [{"role": "user", "content": "hello there"}],
    })
    assert r.status_code == 200, r.text
    rows = [json.loads(line) for p in Path(deps.settings.trace_dir).rglob("*.jsonl")
            for line in p.read_text().splitlines() if line.strip()]
    row = rows[-1]
    assert row["moderation_scored"] is True
    assert row["moderation_input"]["error"] and row["moderation_output"]["error"]
    assert row["moderation_guard"] == "<failed>"


# --- the trace keeps the verdict, never the quoted text (review 2026-09-24 B5) ----


class _FlaggedReplyUpstream(FakeUpstream):
    """Replies with text the stub flags, so the OUTPUT verdict carries a quote too."""

    REPLY = "some nights I want to kill myself"

    async def complete(self, body):
        payload, headers = await super().complete(body)
        payload["choices"][0]["message"]["content"] = self.REPLY
        return payload, headers


@pytest.mark.asyncio
async def test_the_shadow_trace_stores_a_quote_hash_and_never_the_quoted_text(tmp_path):
    """Traces are kept 30 days and read at /internal/traces; every other trace
    avoids content (STT stores sha256 + length). The shadow scorer wrote the
    matched quote from the user's message and the reply verbatim (review
    2026-09-24 B5). The row keeps category, score, a sha256 of the quote and
    its length, and no text."""
    import hashlib
    user_text = "the child is in sexual danger and needs help"
    deps, client = make(tmp_path, _FlaggedReplyUpstream())
    r = client.post("/v1/chat/completions", json={
        "model": "chord-1-poly", "messages": [{"role": "user", "content": user_text}]})
    assert r.status_code == 200, r.text
    raw = [line for p in Path(deps.settings.trace_dir).rglob("*.jsonl")
           for line in p.read_text().splitlines() if line.strip()]
    row = json.loads(raw[-1])
    quotes = {
        "input": (await rule_stub_score(user_text)).categories,
        "output": (await rule_stub_score(_FlaggedReplyUpstream.REPLY)).categories,
    }
    for side, cats in quotes.items():
        field = row[f"moderation_{side}"]
        flagged = [cv for cv in cats if cv.flagged]
        assert flagged, f"the {side} fixture must flag something"
        for cv in flagged:
            entry = field["categories"][cv.category]
            assert entry["flagged"] is True and entry["score"] == cv.score
            assert "quote" not in entry, entry
            assert cv.quote not in raw[-1], f"{side} quote {cv.quote!r} is in the trace row"
            assert entry["quote_sha256"] == hashlib.sha256(cv.quote.encode()).hexdigest()
            assert entry["quote_chars"] == len(cv.quote)
        for cv in cats:
            if not cv.flagged:
                entry = field["categories"][cv.category]
                assert entry["quote_sha256"] is None and entry["quote_chars"] is None
