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
from typing import Optional

__all__ = ["phase0_fixture_dir", "require_monorepo_root"]

# Name of the monorepo's top-level directory. Checked via `.name` equality
# on each ancestor during an upward walk from `Path(__file__)`. Hardcoded
# here (not configurable) because the monorepo layout is the project's
# canonical DEVELOPER working configuration; CI / standalone-clone
# environments don't have this directory and should skip rather than
# configure around it.
_MONOREPO_DIR_NAME = "HFT-pipeline-v2"


def require_monorepo_root(*required_subpaths: str) -> Path:
    """Return the HFT-pipeline-v2 monorepo root, or skip the enclosing test.

    Tests that genuinely need the monorepo layout (e.g., tests loading
    fixtures from sibling repos, comparing against cross-module goldens,
    or walking up to find trainer config merge.py) should use this helper
    INSTEAD of ``Path(__file__).resolve().parents[N]`` + bare
    ``assert ... .exists()`` + RuntimeError. The assert/RuntimeError path
    produces test ERRORs (pytest exit code 2 on module-level asserts);
    this helper produces test SKIPs (pytest exit code 0), which is the
    correct behavior per hft-rules §6 ("tests document behavior; missing
    preconditions are not failures") and consistent with the established
    ``pytest.importorskip`` convention used at 22+ call-sites.

    Formula / walk: starts at ``Path(__file__).resolve()`` (i.e., the
    *helper's own file*, not the caller's — so the walk is stable
    regardless of where this helper is imported from), then iterates
    ``current.parents`` looking for a directory named
    ``HFT-pipeline-v2``. On first match, that path is the monorepo root.
    If no parent matches, the helper calls ``pytest.skip(...,
    allow_module_level=True)`` — safe at both module-scope and
    function-scope (pytest ignores ``allow_module_level`` in function
    context).

    Args:
        *required_subpaths: Additional paths that MUST exist under the
            monorepo root. Each is passed to ``root / sub`` via Path
            concatenation, so POSIX-style forward slashes work
            cross-platform. Example::

                require_monorepo_root(
                    "lob-model-trainer/src/lobtrainer/config/merge.py",
                )

            skips the enclosing test if EITHER the monorepo root OR the
            ``merge.py`` file is absent. Pass no subpaths to skip only
            when the root itself is absent.

    Returns:
        Absolute ``Path`` to the monorepo root, e.g.,
        ``/Users/knight/code_local/HFT-pipeline-v2``. Callers commonly
        assign to ``_REPO_ROOT`` / ``_REAL_PIPELINE_ROOT`` module
        constants and pass to ``PipelinePaths(pipeline_root=...)`` or
        build cross-repo asset paths.

    Raises:
        pytest.skip.Exception: When the monorepo root or any required
            subpath cannot be resolved. Surfaces as pytest's SKIP
            status, never as a test FAILURE. The skip message cites the
            specific missing piece for triage.

    Example (module-level collection-time gate)::

        from hft_contracts._testing import require_monorepo_root
        _REAL_PIPELINE_ROOT = require_monorepo_root(
            "lob-model-trainer/src/lobtrainer/config/merge.py",
        )

        # ... tests below use _REAL_PIPELINE_ROOT freely ...

    Example (function-level runtime gate)::

        def test_cross_repo_manifest():
            root = require_monorepo_root()
            manifest = root / "hft-ops" / "experiments" / "foo.yaml"
            # ... test body ...

    Versioning note: Introduced 2026-04-21 (Phase V.A.0) to replace 20+
    ad-hoc ``parents[N]`` + ``.exists()`` sites across 9 test files in
    hft-ops and lob-model-trainer. New tests requiring the monorepo
    layout MUST use this helper; old sites migrated in the same Phase
    V.A.0 commit series.
    """
    # Lazy import: pytest is not a runtime dependency of hft-contracts;
    # it's only present via the [dev] extra. Importing at function-scope
    # means hft-contracts's public surface doesn't depend on pytest.
    import pytest as _pytest

    current = Path(__file__).resolve()
    root: Optional[Path] = None
    for parent in current.parents:
        if parent.name == _MONOREPO_DIR_NAME:
            root = parent
            break

    if root is None:
        _pytest.skip(
            f"{_MONOREPO_DIR_NAME} monorepo root not found — this test "
            f"requires the monorepo layout (sibling repos under a common "
            f"`{_MONOREPO_DIR_NAME}/` parent directory) and skips cleanly "
            f"when absent (e.g., on GitHub Actions CI, on a fresh "
            f"standalone clone of a single sibling repo). To run this "
            f"test locally, check out the full monorepo such that this "
            f"file resolves to a path containing a `{_MONOREPO_DIR_NAME}/` "
            f"ancestor.",
            allow_module_level=True,
        )

    for sub in required_subpaths:
        if not (root / sub).exists():
            _pytest.skip(
                f"Monorepo root resolved at {root!s} but required subpath "
                f"`{sub}` not found. Full resolution: {(root / sub)!s}. "
                f"This test needs the subpath for cross-repo fixture "
                f"loading or a module import; skipping cleanly so CI / "
                f"partial-checkout environments don't fail spuriously.",
                allow_module_level=True,
            )

    return root


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
