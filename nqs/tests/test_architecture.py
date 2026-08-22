"""Executable dependency rules for the qcnqs package architecture.

The test intentionally inspects imports rather than relying on a diagram alone.
If a future change makes a low-level module import a concrete high-level
implementation, CI reports the exact forbidden edge.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "qcnqs"

# The lazy public ``__init__`` is the only in-package composition boundary.
# Every implementation group has an explicit, deliberately small dependency set.
ALLOWED_INTERNAL_IMPORTS = {
    "_version": set(),
    "assembly": {"chemistry", "hilbert"},
    "autodiff": set(),
    "chemistry": {"chemistry"},
    "determinant": set(),
    "geometry": {"autodiff"},
    "hamiltonian": {
        "chemistry",
        "determinant",
        "hilbert",
        "interfaces",
        "operator_tools",
    },
    "hilbert": set(),
    "hubbard": {"determinant", "hilbert", "interfaces", "operator_tools"},
    "interfaces": {"wavefunction"},
    "local_energy": {"hilbert", "interfaces", "wavefunction"},
    "models": {"models", "wavefunction"},
    "operator_tools": {"hilbert", "interfaces"},
    "sampler": {"determinant", "hilbert", "interfaces", "wavefunction"},
    "vmc": {"geometry", "hilbert", "interfaces", "local_energy", "wavefunction"},
    "wavefunction": set(),
}


def _group(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT)
    return relative.parts[0] if len(relative.parts) > 1 else path.stem


def _qcnqs_import_groups(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    groups = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        for module in modules:
            if module.startswith("qcnqs."):
                groups.add(module.split(".")[1])
    return groups


def test_internal_dependency_direction_is_enforced() -> None:
    violations = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        group = _group(path)
        if path == PACKAGE_ROOT / "__init__.py":
            continue
        allowed = ALLOWED_INTERNAL_IMPORTS[group]
        unexpected = _qcnqs_import_groups(path) - allowed
        if unexpected:
            violations.append(f"{path.relative_to(PACKAGE_ROOT)} -> {sorted(unexpected)}")
    assert not violations, "forbidden qcnqs dependency edges:\n" + "\n".join(violations)


def test_package_root_does_not_eagerly_load_concrete_layers() -> None:
    code = """
import sys
import qcnqs

forbidden = {
    'qcnqs.chemistry', 'qcnqs.hamiltonian', 'qcnqs.hubbard',
    'qcnqs.sampler', 'qcnqs.vmc'
}
assert forbidden.isdisjoint(sys.modules)
assert qcnqs.QCHamiltonian.__name__ == 'QCHamiltonian'
assert 'qcnqs.hamiltonian' in sys.modules
assert 'qcnqs.sampler' not in sys.modules
for name in qcnqs.__all__:
    getattr(qcnqs, name)
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=PACKAGE_ROOT.parents[1])
