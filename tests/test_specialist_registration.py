"""A new specialist must be included in the startup entry-point list."""

import ast
from pathlib import Path

from chord.dependencies import SPECIALIST_ENTRY_MODULES


def test_every_specialist_decorator_has_a_startup_entry():
    package = Path(__file__).resolve().parents[1] / "src" / "chord" / "specialists"
    discovered = set()
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "specialist"
                for decorator in node.decorator_list
            ):
                discovered.add(path.stem)
    assert discovered == set(SPECIALIST_ENTRY_MODULES)
