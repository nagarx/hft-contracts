"""AST regression test — atomic_io must not pull torch at module load.

Phase #PY-73 closure (2026-05-11, hft-contracts v2.7.0):
``atomic_write_torch`` was added to hft_contracts.atomic_io. It MUST
lazy-import torch inside the function body so that ``import
hft_contracts.atomic_io`` doesn't drag torch into ``sys.modules`` for
torch-free consumers (notably hft-ops, which enforces the same
invariant via ``test_contract_preflight_module_imports_are_torch_free``).

This test locks the lazy-import discipline at TWO levels:

1. **AST level** — parse atomic_io.py, walk top-level ``Import`` /
   ``ImportFrom`` nodes, assert no ``torch`` import appears at module
   scope.
2. **Runtime level** — subprocess that does ``import
   hft_contracts.atomic_io`` and asserts ``'torch' not in sys.modules``
   after the import returns. Subprocess isolation is REQUIRED because
   pytest's main ``sys.modules`` is already poisoned by collection.

The 2-level check defends against both an accidental top-level
``import torch`` AND a future refactor that adds ``from torch import
save`` at module scope (which the AST check catches and the runtime
check defends in depth).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import hft_contracts.atomic_io  # noqa: F401 — ensure module loads


def _atomic_io_path() -> Path:
    """Return the path to hft_contracts/atomic_io.py source."""
    return Path(hft_contracts.atomic_io.__file__).resolve()


def test_atomic_io_no_top_level_torch_import() -> None:
    """AST-level: torch must not appear in any top-level Import/ImportFrom."""
    source = _atomic_io_path().read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden_modules = {"torch"}
    violations: list[str] = []

    # Walk only top-level statements — NOT recursing into function bodies.
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                # alias.name can be "torch" or "torch.X.Y" — root name match.
                root = alias.name.split(".")[0]
                if root in forbidden_modules:
                    violations.append(
                        f"line {node.lineno}: import {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            root = node.module.split(".")[0]
            if root in forbidden_modules:
                imported = ", ".join(a.name for a in node.names)
                violations.append(
                    f"line {node.lineno}: from {node.module} import {imported}"
                )

    assert not violations, (
        f"hft_contracts.atomic_io has top-level torch import(s): {violations}. "
        f"torch MUST be lazy-imported inside atomic_write_torch body to "
        f"preserve hft-ops torch-free invariant."
    )


def test_atomic_io_module_load_does_not_import_torch() -> None:
    """Runtime: importing hft_contracts.atomic_io must not pull torch into sys.modules.

    Uses subprocess isolation since pytest's sys.modules is already
    poisoned by collection (test imports torch indirectly via
    test_atomic_io.py round-trip tests).
    """
    code = (
        "import sys\n"
        "import hft_contracts.atomic_io  # noqa: F401\n"
        "assert 'torch' not in sys.modules, (\n"
        "    f'torch leaked into sys.modules after importing '\n"
        "    f'hft_contracts.atomic_io. Modules: '\n"
        "    f'{sorted(m for m in sys.modules if \"torch\" in m)}'\n"
        ")\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "OK" in result.stdout


def test_atomic_io_module_load_does_not_import_numpy_lazily() -> None:
    """Numpy IS a hft-contracts dep — top-level import is expected.

    Sanity test: verify numpy IS in sys.modules after importing
    atomic_io (the opposite of the torch invariant). If this test
    fails, atomic_io was inadvertently refactored to lazy-import numpy
    — that would be FINE for the torch-free invariant, but should be
    a deliberate decision (current design: numpy is top-level because
    it's already a runtime dep).
    """
    code = (
        "import sys\n"
        "import hft_contracts.atomic_io  # noqa: F401\n"
        "assert 'numpy' in sys.modules, (\n"
        "    'numpy should be top-level imported by atomic_io.py '\n"
        "    '(it is already a hft-contracts runtime dep). If you '\n"
        "    'changed to lazy import, update this test.'\n"
        ")\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
