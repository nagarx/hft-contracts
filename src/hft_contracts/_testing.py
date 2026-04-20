"""Internal testing helpers — resolve fixture paths for cross-module integration tests.

This module is INTERNAL (leading underscore). Do NOT import from it in production code.
Consumer integration tests in sibling repos (lob-models, lob-model-trainer, lob-backtester,
hft-ops) use `phase0_fixture_dir()` to locate committed Phase 0 benchmark fixtures without
hardcoding monorepo-relative paths.

Invariants:
  - Only editable installs (`pip install -e .`) can reach these fixtures.
  - Wheel / sdist installs raise FileNotFoundError with an actionable message.
  - No runtime cost on import (lazy path resolution).

Rationale (plan v2.0, §Architectural Invariants #4 "Load-bearing contracts only"):
  - Fixtures are a testing concern, not a runtime contract — keeps the `hft_contracts`
    public surface clean of test-only machinery.
  - Underscore prefix signals "internal to the package" per PEP 8.
  - Single source of truth for the fixture location (all consumers reach the same bytes).
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["phase0_fixture_dir"]


def phase0_fixture_dir() -> Path:
    """Return absolute path to the committed Phase 0 benchmark fixtures directory.

    Resolution: walks up from ``hft_contracts/__init__.py`` to the repo root, then down
    into ``tests/fixtures/phase0_benchmark/``. Works when hft-contracts is installed in
    editable mode (``pip install -e .``), which is the pipeline's canonical install mode.

    Returns:
        Path to the fixtures directory containing ``synthetic_mbo.npz``, ``synthetic_basic.npz``,
        ``golden_values.json``, and ``README.md``.

    Raises:
        FileNotFoundError: if the fixtures directory cannot be resolved — typically because
            hft-contracts was installed from a wheel (tests/ is not packaged) or the repo
            tree has been moved. The error message documents the recovery path.

    Example:
        >>> from hft_contracts._testing import phase0_fixture_dir
        >>> import numpy as np
        >>> with np.load(phase0_fixture_dir() / "synthetic_mbo.npz") as npz:
        ...     sequences = npz["sequences"]
    """
    # hft_contracts/__init__.py lives at:
    #   <repo_root>/hft-contracts/src/hft_contracts/__init__.py
    # so fixtures are at:
    #   <repo_root>/hft-contracts/tests/fixtures/phase0_benchmark/
    # which is three levels up from this file (_testing.py → __init__.py → src/ → hft-contracts/)
    import hft_contracts  # local to avoid circular resolution during package init

    package_init = Path(hft_contracts.__file__)
    # package_init → src/hft_contracts/__init__.py
    # parent(0) → src/hft_contracts/
    # parent(1) → src/
    # parent(2) → hft-contracts/ (the standalone-repo root)
    repo_root = package_init.parent.parent.parent
    candidate = repo_root / "tests" / "fixtures" / "phase0_benchmark"

    if not candidate.exists():
        raise FileNotFoundError(
            f"Phase 0 benchmark fixtures not found at {candidate}. "
            f"Possible causes: (1) hft-contracts was installed from a wheel or sdist "
            f"(tests/ is not shipped) — reinstall with `pip install -e hft-contracts/`; "
            f"(2) the repository tree has been relocated — verify that `hft_contracts/__init__.py` "
            f"resolves to `<repo>/hft-contracts/src/hft_contracts/__init__.py`. "
            f"Current resolution: {package_init}."
        )
    return candidate
