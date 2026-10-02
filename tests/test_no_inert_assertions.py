"""No assertion in this repository may be structurally incapable of failing.

Added after this assertion survived an implementation and a later review on
2026-09-19 without being noticed:

    assert committed.read_bytes() == subprocess.run(...).stdout or True

`==` binds tighter than `or`, so it parses as `assert (bytes == stdout) or True` --
true for every input, forever. It guarded the byte-integrity of the archived
attestation, which is the N1 trust anchor, and it had been inert since it was written.

The comment on it read "untracked before the lift lands" -- a deliberate temporary
bypass for a transient state. **The lift landed. The bypass did not leave.** An expiry
condition written in prose has no expiry; only an executable one does.

A grep for `or True` catches that exact spelling and nothing else. This parses instead,
so `or 1`, `or "x"`, `or [0]` and a bare `assert True` are caught by the same rule --
and so is the next spelling nobody has thought of, because the test is about the
STRUCTURE being constant-true rather than about the text.

2026-09-19.
"""
import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
ROOTS = ("tests", "qa")


def _python_files():
    for root in ROOTS:
        for p in sorted((REPO / root).rglob("*.py")):
            if "__pycache__" not in p.parts:
                yield p


def _inert(node):
    """Why this assertion can never fail, or None.

    Only the unambiguous cases: a truthy constant, or an `or` with a truthy constant
    anywhere in it. Deliberately not clever -- a check that tries to decide
    satisfiability would itself become something nobody trusts."""
    test = node.test
    if isinstance(test, ast.Constant) and test.value:
        return f"assert {ast.unparse(test)} is always true"
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
        for value in test.values:
            if isinstance(value, ast.Constant) and value.value:
                return (f"`or {ast.unparse(value)}` makes the whole assertion true; "
                        f"note `==` binds tighter than `or`")
    return None


def test_no_assertion_in_the_repo_is_structurally_inert():
    found = []
    for path in _python_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                why = _inert(node)
                if why:
                    found.append(f"{path.relative_to(REPO)}:{node.lineno}: {why}")
    assert not found, "assertions that cannot fail:\n  " + "\n  ".join(found)


@pytest.mark.parametrize("src,caught", [
    ("assert x == y or True", True),
    ("assert True", True),
    ("assert x or 1", True),
    ('assert x or "why"', True),
    ("assert x == y", False),
    ("assert x or y", False),
    ("assert x == y, 'message'", False),
    ("assert False", False),          # can fail -- and always does; not this rule's job
])
def test_the_detector_catches_the_shapes_it_claims_to(src, caught):
    """The check gets its own falsifiers. A detector for inert assertions that is itself
    inert would be the exact joke this file exists to stop being."""
    node = ast.parse(src).body[0]
    assert (_inert(node) is not None) is caught, ast.unparse(node)
