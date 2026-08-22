"""Guard the structured English documentation of public numerical modules."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "qcnqs"


def test_public_modules_and_top_level_symbols_have_docstrings() -> None:
    missing = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        relative = path.relative_to(PACKAGE_ROOT)
        if ast.get_docstring(tree) is None:
            missing.append(f"{relative}: module")
        for node in tree.body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_") or ast.get_docstring(node) is not None:
                continue
            missing.append(f"{relative}: {node.name}")
    assert not missing, "missing public English docstrings:\n" + "\n".join(missing)
