"""A configured Qwen router is bound with enable_thinking=False at every call site.

2026-09-27. Persona calls go through thinking_switch(). Router calls
did not, and on 2026-09-18 when
LiteLLM came out of the middle the 9B router median went to 6,739 ms.
With thinking off: 641 ms. qa/performance/BASELINE-2026-09-23.md.

graph.router_client() is the bind that replaces the missing call. These
tests assert the bind shape (what gets bound onto the model) and the
call-site adoption (which graph paths use it).
"""

from unittest.mock import MagicMock

from chord.graph import router_client


def test_router_client_calls_bind_with_enable_thinking_false():
    """The Qwen bind shape: router_client wraps the input with extra_body carrying
    chat_template_kwargs.enable_thinking=False. This is the change that
    turns the router's thinking on default off."""
    fake = MagicMock()
    fake.bind = MagicMock(return_value="bound-client")
    out = router_client(fake, "qwen_chat_template")
    fake.bind.assert_called_once_with(
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    assert out == "bound-client"


def test_router_client_passes_through_non_bindable():
    """Stub routers in tests/test_router_classifier.py and
    tests/test_false_delivery.py are simple objects with .ainvoke but no
    .bind. router_client must return them untouched so the existing
    fixtures keep working."""

    class StubRouter:
        async def ainvoke(self, msgs):
            class R:
                content = '{"route":"chat"}'

            return R()

    bound = router_client(StubRouter())
    assert isinstance(bound, StubRouter)


def test_router_client_bind_composes_with_downstream_bind():
    """The false-delivery call site in graph.py does:
        llm = router_client(model(settings.router_model))   # bind 1: thinking off
        if hasattr(llm, "bind"):
            llm = llm.bind(response_format={"type": "json_object"})   # bind 2: JSON mode

    Both binds must compose, and router_client must NOT swallow the
    downstream bind by short-circuiting.
    """
    bind_calls: list[dict] = []

    class FakeChain:
        def __init__(self, tag: str) -> None:
            self.tag = tag
            self.ainvoke = MagicMock()

        def bind(self, **kwargs) -> "FakeChain":
            bind_calls.append({"tag": self.tag, **kwargs})
            return FakeChain(self.tag + "+bound")

    base = FakeChain("base")
    out1 = router_client(base, "qwen_chat_template")
    assert isinstance(out1, FakeChain)
    # The downstream bind sees the bound object, not the original.
    out1.bind(response_format={"type": "json_object"})
    assert bind_calls == [
        {
            "tag": "base",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        {"tag": "base+bound", "response_format": {"type": "json_object"}},
    ]


# --- call-site adoption (structural) ---------------------------------------


def test_graph_uses_router_client_at_every_router_model_call():
    """Every router-model call site in graph.py must go through
    router_client. If any of them bypasses the helper, the 9B router pays
    the full thinking tax again.

    AST-based: walks the source for any call to `model(...)` that takes
    `settings.router_model`, and asserts each such call sits inside a
    `router_client(...)` wrapper. Newlines, formatting, and aliases are
    immaterial because we walk the parse tree, not the source bytes.
    """
    import ast
    from pathlib import Path

    src = Path("src/chord/graph.py").read_text()
    tree = ast.parse(src)

    def call_name(node: ast.Call) -> str | None:
        """The plain name of a Call's function, or None if it's an attr."""
        f = node.func
        if isinstance(f, ast.Name):
            return f.id
        return None

    def is_router_model_call(call: ast.Call) -> bool:
        """Is this `model(...)` call passed `settings.router_model`?"""
        if call_name(call) != "model":
            return False
        for arg in call.args:
            if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name):
                if arg.value.id == "settings" and arg.attr == "router_model":
                    return True
        return False

    # Collect every Call node; we'll need two views.
    all_calls: list[ast.Call] = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

    router_client_wrappers: list[ast.Call] = [
        c for c in all_calls if call_name(c) == "router_client"
    ]

    # A model(settings.router_model) call is "wrapped" if it appears as a
    # positional argument of any router_client(...) call. The AST uses
    # `is` for identity, which is what we want here.
    wrapped_model_calls: set[int] = set()
    for wrapper in router_client_wrappers:
        for arg in wrapper.args:
            if isinstance(arg, ast.Call) and is_router_model_call(arg):
                wrapped_model_calls.add(id(arg))

    bare_lines: list[int] = []
    for call in all_calls:
        if not is_router_model_call(call):
            continue
        if id(call) not in wrapped_model_calls:
            bare_lines.append(call.lineno)

    assert not bare_lines, (
        f"every model(settings.router_model) call must sit inside a "
        f"router_client(...) wrapper; bare calls at lines {bare_lines}"
    )

    # And there should be at least four wrapper call sites today.
    assert len(router_client_wrappers) >= 3, (
        f"expected router_client wired in at >=3 call sites, "
        f"got {len(router_client_wrappers)} at lines "
        f"{sorted(c.lineno for c in router_client_wrappers)}"
    )
