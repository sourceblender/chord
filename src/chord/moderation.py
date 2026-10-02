"""Shadow-scoring moderation gate — Phase 1 of #121.

This is the infrastructure for a future moderation gate that blocks external
traffic on categories selected by the policy. Today it does NOT block,
and today it does NOT modify the wire. It runs alongside the chat door as
a shadow scorer:

  * input is the last user message (or empty if no user message exists);
  * output is the assistant's reply text;
  * every verdict lands on the trace row, never on the response;
  * the verdict is structured per the design: per-category scores, a verdict
    per category, and a quote required for every flagged category (the
    identity-resolver discipline: the model decides, the code checks the
    evidence).

The guard model is a stub: a small rule-based classifier with regex
patterns on a tiny taxonomy. It is replaced by configuration (real model,
real prompts) before any blocking happens. The shape it returns is the
shape the real guard will return, so swapping it is config not code.

Why a stub
----------
The design comment on #121 says *"v1 ships one mechanism, measured"*.
A rule-based stub that runs in microseconds is the right Phase-1 guard:
it generates trace data with the right shape, lets the field-by-field
verifiers run, and produces a base population against which false-positive
rates can be measured when the real model lands. The stub's behaviour is
explicit about its limits (see `_RULES`) so a regression that mistakes
the stub for the gate is loud.

Why async but awaited
---------------------
The design says *"trace-only, async, zero wire change"*. We achieve async
in the sense that the scoring does not block the model's generation: it
runs after the model returns, before the trace is written. The added
latency is the scoring time, which the stub keeps under a millisecond.
A future real-guard scoring would still be pre-trace-write and would
similarly add only its own runtime to the request path; if the real
guard's runtime becomes the bottleneck, the proper answer is moving the
score post-trace-write (with a "score pending" trace field) rather than
silently dropping scoring on slow turns.

What this module does NOT do (the build-deferred list)
------------------------------------------------------
* Block input. Phase 1 is shadow-only; block-mode refuses with
  `moderation_blocked` are a Phase-3 decision once the FP/FN data exists.
* Block output. Same.
* Stream-and-block. The streaming path does not yet wire the shadow
  scorer; that lands with the output-gate decision in Phase 3.
* Score images or audio. Only text today.
* Persist decisions beyond the trace row. The 30-day retention on the
  trace directory is the audit store, by design (the design comment).

What this module DOES do today
------------------------------
* Define the policy taxonomy (BLOCKING_CATEGORIES, ADVISORY_CATEGORIES).
* Define the policy (`chord-default-1`) as a versioned text blob with a
  pinned SHA256, so a verdict is bound to the policy that produced it.
* Score a single string, returning one `Verdict` per BLOCKING + ADVISORY
  category, each with a 0..1 score, a flag, and (if flagged) a quote
  span that is a substring of the input.
* Reject verdicts whose quote is not a substring of the input -- the
  check-the-evidence discipline.
* Fail-open for internal traffic, traced. The design says external
  traffic must fail-closed with 503 `moderation_unavailable`; that
  gate fires in #121's Phase-3 once the wire shape is decided.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


# --- taxonomy (#121, design comment decision #2) --------------------------------

# Blocking categories: a verdict over the threshold REQUIRES a block-mode
# decision in Phase 3. Shadow-scored today; not blocking.
BLOCKING_CATEGORIES: tuple[str, ...] = (
    "minors",           # sexualization / endangerment of under-18 or age-ambiguous framing
    "non_consent",      # sexual content involving coercion or incapacity
    "real_person_harm", # harm / threats / fabricated defamation about a named real person
)

# Advisory categories: scored, never block. The design's own list.
ADVISORY_CATEGORIES: tuple[str, ...] = (
    "adult_themes",     # consensual adult material -- allowed by the default policy
    "self_harm",        # scored for visibility, never a violation
)

ALL_CATEGORIES: tuple[str, ...] = BLOCKING_CATEGORIES + ADVISORY_CATEGORIES


# --- policy text ----------------------------------------------------------------

POLICY_NAME = "chord-default-1"
POLICY_TEXT = (
    "Chord default policy: allow consensual adult material. Block content that "
    "sexualizes or endangers minors (including age-ambiguous framing), content "
    "involving coercion or incapacity, and harm or fabricated defamation "
    "directed at a named real person. Score adult themes and self-harm for "
    "visibility without blocking. A flagged category without an in-text quote "
    "is treated as unflagged -- evidence is the contract, not the verdict."
)


def policy_sha256() -> str:
    """Pinned by the policy text; recomputed when the text changes.

    A verdict carries its `policy_sha256` into the trace row, so a future
    audit can recover the exact policy that produced it. Bumping the policy
    text creates a new gate and a new comparison population."""
    return hashlib.sha256(POLICY_TEXT.encode()).hexdigest()


# --- guard stub (#121, design comment decision #3) -------------------------------

# A handful of patterns per category. This is the SHADOW STUB; it is NOT
# the gate. It exists so the shadow lane produces trace data with the
# shape the real guard will return. When the real guard lands, the
# rule-based stub stays as the dev/test fallback and the live scoring
# uses the configured model. The rule set is intentionally narrow -- a
# wider set would teach us the wrong thing about FP/FN rates against a
# real classifier.

_RULES: dict[str, tuple[tuple[re.Pattern[str], str], ...]] = {
    "minors": (
        (re.compile(r"\b(child|children|kid|minor|underage|teen)\b.{0,40}\b(sex|sexual|nude|naked|abuse)\b",
                    re.IGNORECASE),
         "sexualization of an under-18 framing"),
        (re.compile(r"\b(schoolgirl|schoolboy|lolita|infantil)\b", re.IGNORECASE),
         "age-coded sexualization"),
    ),
    "non_consent": (
        # Word-internal morphology: 'raped', 'forced', 'coerced' must hit too,
        # so the rule is anchored at the start and uses a word boundary on
        # the right only.
        (re.compile(r"\b(rape|forced|coerced|coercion|non[- ]?consensual)\w*\b", re.IGNORECASE),
         "non-consent indicator"),
        (re.compile(r"\b(incapacitated|unconscious|drugged|intoxicated).{0,30}\bsex", re.IGNORECASE),
         "incapacity"),
        (re.compile(r"without consent", re.IGNORECASE),
         "explicit non-consent"),
    ),
    "real_person_harm": (
        (re.compile(r"\b(kill|murder|assassinate|attack|bomb|shoot)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"),
         "harm directed at a named person"),
        (re.compile(r"\b(fake|forged|fabricated)\s+(story|claim|affair|scandal)\s+(about|against|on)\s+[A-Z][a-z]+"),
         "fabricated defamation"),
    ),
    "adult_themes": (
        (re.compile(r"\b(porn|nsfw|xxx|explicit sexual)\b", re.IGNORECASE),
         "consensual adult material"),
    ),
    "self_harm": (
        (re.compile(r"\b(kill myself|end my life|commit suicide|self[- ]?harm|suicide)\b", re.IGNORECASE),
         "self-harm indicator"),
    ),
}


@dataclass(frozen=True)
class CategoryVerdict:
    """One category's verdict on one piece of text."""
    category: str
    flagged: bool
    score: float              # 0.0 (clean) ... 1.0 (rule hit hard)
    quote: str | None         # required iff flagged; must be a substring of input
    reason: str | None        # required iff flagged; explains the rule that fired


@dataclass(frozen=True)
class Verdict:
    """One moderation decision on one piece of text."""
    flagged: bool                       # any category flagged
    blocking_flagged: bool              # any BLOCKING category flagged
    categories: tuple[CategoryVerdict, ...]
    guard_model: str                    # "rule-stub-v1" today; configurable later
    policy_name: str
    policy_sha256: str
    latency_ms: int                     # scoring wall time
    error: str | None = None            # None on success; string on guard failure

    def as_trace_field(self) -> dict:
        """The shape that lands on the trace row.

        Flat enough to read in a one-line JSONL viewer; the per-category
        detail is preserved so a future analyst can rebuild the verdict.

        The quote itself is NOT written: it is text from the user's message or
        the reply, and traces are kept 30 days and readable at /internal/traces,
        where every other row avoids content (STT stores sha256 + length). The
        row carries the quote's sha256 and length instead, enough to match a
        verdict against a known text without storing it (review 2026-09-24 B5).
        The hash encodes with surrogatepass: JSON admits an escaped lone
        surrogate, a rule match can include it, and a strict encode raised
        while the trace was written (Copilot on #335)."""
        return {
            "flagged": self.flagged,
            "blocking_flagged": self.blocking_flagged,
            "categories": {
                v.category: {
                    "flagged": v.flagged,
                    "score": v.score,
                    "quote_sha256": (hashlib.sha256(v.quote.encode("utf-8", "surrogatepass")).hexdigest()
                                     if isinstance(v.quote, str) else None),
                    "quote_chars": len(v.quote) if isinstance(v.quote, str) else None,
                    "reason": v.reason,
                }
                for v in self.categories
            },
            "guard_model": self.guard_model,
            "policy_name": self.policy_name,
            "policy_sha256": self.policy_sha256,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


# --- the guard ------------------------------------------------------------------

Guard = Callable[[str], Awaitable[Verdict]]
"""The guard's signature: a string in, a Verdict out, async.

A real guard wraps a model call. The stub is async to match that shape;
the real swap is configuration, not a refactor."""


# Default per-category threshold. The design said the blocking categories
# default to 0.9 conservatively. The stub's score is binary (1.0 on rule
# hit), so the threshold is mostly meaningful once the real model lands;
# the same constant works for both.
DEFAULT_THRESHOLD = 0.9


async def rule_stub_score(text: str) -> Verdict:
    """The Phase-1 guard. Synchronous-looking, awaited.

    The rule-stub scores 1.0 on any rule hit. A quoted span is the rule
    pattern's match (which is a substring of the input -- the
    check-the-evidence discipline falls out for free). For non-blocking
    categories the score is also 1.0 on hit; the design says advisory
    categories never block, so the score is for visibility only.
    """
    import time
    t0 = time.monotonic()
    verdicts: list[CategoryVerdict] = []
    for cat in ALL_CATEGORIES:
        rules = _RULES.get(cat, ())
        for pattern, reason in rules:
            match = pattern.search(text)
            if match is None:
                continue
            quote = match.group(0)
            verdicts.append(CategoryVerdict(
                category=cat, flagged=True, score=1.0,
                quote=quote, reason=reason,
            ))
            break  # one rule hit per category is enough; further hits don't change the verdict
        else:
            verdicts.append(CategoryVerdict(
                category=cat, flagged=False, score=0.0, quote=None, reason=None,
            ))
    flagged = any(v.flagged for v in verdicts)
    blocking_flagged = any(v.flagged for v in verdicts if v.category in BLOCKING_CATEGORIES)
    return Verdict(
        flagged=flagged,
        blocking_flagged=blocking_flagged,
        categories=tuple(verdicts),
        guard_model="rule-stub-v1",
        policy_name=POLICY_NAME,
        policy_sha256=policy_sha256(),
        latency_ms=round((time.monotonic() - t0) * 1000),
    )


def _check_evidence(verdict: Verdict, source: str) -> None:
    """The discipline: a verdict without checkable evidence is treated as
    unflagged. The stub always produces evidence that IS in the source,
    but a real guard might not. We don't trust its word; we trust what
    we can verify in the source.

    Called by `score` after every guard call, before returning to the
    chat door. A regression that bypasses this is the bug #121's
    "identity resolver discipline" comment warns against.
    """
    for v in verdict.categories:
        if v.flagged and (v.quote is None or v.quote not in source):
            # Replace with an unflagged verdict in-place. This is silent
            # from the wire's perspective (the trace shows it), and it
            # is the discipline: the model decides, the code checks.
            object.__setattr__(v, "flagged", False)
            object.__setattr__(v, "score", 0.0)
            object.__setattr__(v, "quote", None)
            object.__setattr__(v, "reason",
                "verdict without checkable evidence, treated as unflagged")
    object.__setattr__(verdict, "flagged", any(v.flagged for v in verdict.categories))
    object.__setattr__(verdict, "blocking_flagged",
        any(v.flagged for v in verdict.categories if v.category in BLOCKING_CATEGORIES))


async def score(text: str, *, guard: Guard | None = None) -> Verdict:
    """Score one piece of text. Always returns a verdict; failures degrade
    gracefully because Phase 1 must produce a base population, not gaps.

    On guard exception: a Verdict with `error` set and every category
    unflagged. The chat door sees this and records it; nothing breaks.
    The design's fail-closed-for-external posture is the Phase-3 wire
    policy; today everything is internal and we fail open (traced).
    """
    g = guard or rule_stub_score
    try:
        verdict = await g(text)
        # Check the evidence regardless of where the verdict came from, and
        # inside the fail-open boundary: a guard that answers badly (a quote
        # that is not a string, not a Verdict at all) is as much a scoring
        # failure as one that raises. Outside this try it escaped the chat
        # door AFTER the reply was built and turned it into a 500 (review
        # 2026-09-24).
        _check_evidence(verdict, text)
    except Exception as exc:
        logger.warning("moderation guard raised (%s); failing open (traced)",
                       type(exc).__name__)
        verdict = Verdict(
            flagged=False, blocking_flagged=False,
            categories=tuple(CategoryVerdict(category=c, flagged=False, score=0.0,
                                            quote=None, reason=None)
                            for c in ALL_CATEGORIES),
            guard_model="<failed>",
            policy_name=POLICY_NAME,
            policy_sha256=policy_sha256(),
            latency_ms=0,
            error=f"{type(exc).__name__}: {exc}",
        )
    return verdict


# --- audit log for the policy file ---------------------------------------------

def policy_audit_blob() -> dict:
    """The versioned policy text, the SHA, and the category sets -- for
    an operator archive if you want to keep a policy artifact next to the
    shipped verdict rows. The trace carries the SHA so the link is
    one query away."""
    return {
        "name": POLICY_NAME,
        "sha256": policy_sha256(),
        "text": POLICY_TEXT,
        "blocking_categories": list(BLOCKING_CATEGORIES),
        "advisory_categories": list(ADVISORY_CATEGORIES),
        "all_categories": list(ALL_CATEGORIES),
        "default_threshold": DEFAULT_THRESHOLD,
        "guard": "rule-stub-v1",
    }


if __name__ == "__main__":
    # Smoke: print the audit blob so a CI run can diff it if the policy
    # text changes intentionally.
    print(json.dumps(policy_audit_blob(), indent=2, sort_keys=True))
