"""Tests for hft_contracts._testing — the cross-module fixture-path resolver.

Locks the contract that consumer integration tests depend on:
  - phase0_fixture_dir() returns a concrete Path
  - the returned directory contains the canonical Phase 0 fixtures
  - the error path is actionable when fixtures are absent
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hft_contracts._testing import phase0_fixture_dir, require_monorepo_root


class TestPhase0FixtureDir:
    def test_returns_existing_directory(self):
        fixture_dir = phase0_fixture_dir()
        assert isinstance(fixture_dir, Path)
        assert fixture_dir.is_dir()

    def test_contains_mbo_fixture(self):
        assert (phase0_fixture_dir() / "synthetic_mbo.npz").exists()

    def test_contains_basic_fixture(self):
        assert (phase0_fixture_dir() / "synthetic_basic.npz").exists()

    def test_contains_golden_values(self):
        assert (phase0_fixture_dir() / "golden_values.json").exists()

    def test_contains_readme(self):
        assert (phase0_fixture_dir() / "README.md").exists()

    def test_contains_generate_script(self):
        assert (phase0_fixture_dir() / "generate.py").exists()

    def test_contains_metadata_jsons(self):
        d = phase0_fixture_dir()
        assert (d / "fixture_metadata_mbo.json").exists()
        assert (d / "fixture_metadata_basic.json").exists()

    def test_resolves_absolute(self):
        """Resolver must return an absolute path — relative paths break when tests cd."""
        assert phase0_fixture_dir().is_absolute()

    def test_idempotent(self):
        """Repeated calls return the same path (no cache staleness)."""
        a = phase0_fixture_dir()
        b = phase0_fixture_dir()
        assert a == b


class TestRequireMonorepoRoot:
    """Lock `require_monorepo_root(*subpaths)` contract.

    Phase V.A.0 (2026-04-21): SSoT pattern replacing 20+ ad-hoc
    ``Path(__file__).resolve().parents[N]`` + bare ``assert ... .exists()``
    sites across 9 test files in hft-ops and lob-model-trainer. Consumer
    tests that genuinely need the monorepo layout (cross-repo fixture
    loading, trainer merge.py reflection, etc.) must use this helper
    INSTEAD of raw path walks — it converts module-level ``assert`` errors
    (pytest exit-code-2) into clean ``pytest.skip`` (exit-code-0), matching
    the established ``pytest.importorskip`` convention.

    These tests assume running FROM INSIDE the monorepo. In a
    standalone-clone CI environment, `require_monorepo_root` would itself
    call ``pytest.skip`` — so these tests too would skip cleanly rather
    than ERROR. That's consistent behavior, not a bug.
    """

    def test_returns_monorepo_root_when_present(self):
        """Happy path: monorepo root resolves via upward walk."""
        root = require_monorepo_root()
        assert root.name == "HFT-pipeline-v2"
        assert root.is_dir()
        assert root.is_absolute()

    def test_idempotent(self):
        """Multiple calls return equal paths."""
        a = require_monorepo_root()
        b = require_monorepo_root()
        assert a == b

    def test_accepts_single_valid_subpath(self):
        """Passing a valid required subpath returns the same root."""
        root = require_monorepo_root("hft-contracts/pyproject.toml")
        assert root.name == "HFT-pipeline-v2"

    def test_accepts_multiple_valid_subpaths(self):
        """All valid subpaths present → returns root."""
        root = require_monorepo_root(
            "hft-contracts/pyproject.toml",
            "hft-contracts/src/hft_contracts/__init__.py",
            "hft-contracts/tests",
        )
        assert root.name == "HFT-pipeline-v2"

    def test_bare_call_equivalent_to_all_valid_subpaths(self):
        """No-subpath call and all-valid-subpath call return equal roots."""
        bare = require_monorepo_root()
        with_subpath = require_monorepo_root("hft-contracts/pyproject.toml")
        assert bare == with_subpath

    def test_skips_when_required_subpath_missing(self):
        """Missing subpath → pytest.skip (caught as Skipped via pytest.raises)."""
        with pytest.raises(pytest.skip.Exception):
            require_monorepo_root(
                "this/path/intentionally/does/not/exist/anywhere.impossible"
            )

    def test_skips_when_second_of_two_subpaths_missing(self):
        """Short-circuit not required — any failing subpath triggers skip."""
        with pytest.raises(pytest.skip.Exception):
            require_monorepo_root(
                "hft-contracts/pyproject.toml",  # exists
                "nonexistent/second/subpath",     # missing
            )

    def test_skip_message_is_actionable(self):
        """Skip message is non-empty and mentions ``HFT-pipeline-v2`` or the
        missing subpath — so triage can tell which invariant fired.

        CI vs dev divergence note: on CI (standalone hft-contracts checkout)
        the FIRST skip fires ("monorepo root not found") — the message
        mentions ``HFT-pipeline-v2``. On dev (monorepo present but subpath
        absent) the SECOND skip fires — the message cites the missing
        subpath. We accept either; what matters is that the message helps
        a developer understand why the test skipped."""
        try:
            require_monorepo_root("this/path/is/not/here.missing")
        except pytest.skip.Exception as exc:
            msg = str(exc)
            assert len(msg) > 50, f"Skip message should be actionable; got: {msg!r}"
            # Either the monorepo-absent path OR the subpath-missing path must fire.
            cites_monorepo = "HFT-pipeline-v2" in msg
            cites_subpath = "this/path/is/not/here.missing" in msg
            assert cites_monorepo or cites_subpath, (
                f"Skip message should mention either the monorepo name or "
                f"the missing subpath. Got: {msg!r}"
            )
        else:
            pytest.fail("require_monorepo_root should have skipped")

    def test_returns_existing_directory(self):
        """Returned root must exist as a directory (not a dangling symlink)."""
        root = require_monorepo_root()
        assert root.exists()
        assert root.is_dir()

    def test_signature_file_required_under_candidate(self):
        """V.A.0 audit C1: root must contain `contracts/pipeline_contract.toml`.

        This guards against the pytest fixture collision where
        ``tmp_path / "HFT-pipeline-v2"`` is created as a mock monorepo
        tree; bare name matching would match the synthetic tree first if
        TMPDIR is set inside the real monorepo. The signature-file check
        ensures only the real root (which has the canonical TOML)
        resolves.
        """
        root = require_monorepo_root()
        assert (root / "contracts" / "pipeline_contract.toml").exists(), (
            "Resolved monorepo root must contain the signature file "
            "`contracts/pipeline_contract.toml`; otherwise the walk's "
            "directory-name match collided with a tmp_path fixture."
        )

    def test_walk_is_memoized(self):
        """V.A.0 audit F6: walk cached via lru_cache for perf + stability."""
        from hft_contracts._testing import _discover_monorepo_root

        # Clear cache to start from a known state, exercise twice, verify
        # identity (same Path object returned on 2nd call — proves cache hit).
        _discover_monorepo_root.cache_clear()
        first = _discover_monorepo_root()
        second = _discover_monorepo_root()
        # lru_cache stores the return value; `is` identity confirms cache hit
        # (both calls return the exact same Path instance, not a fresh resolve).
        assert first is second, (
            "_discover_monorepo_root should be memoized (lru_cache); "
            "got fresh Path instances on 2nd call."
        )

    def test_reason_prefix_included_in_skip_message(self):
        """V.A.0 audit F11: reason_prefix kwarg prepends context to skip."""
        context = "Integration tests require data/exports layout"
        with pytest.raises(pytest.skip.Exception) as exc_info:
            require_monorepo_root(
                "nonexistent/subpath/for/this/test",
                reason_prefix=context,
            )
        msg = str(exc_info.value)
        assert context in msg, (
            f"Skip message should prepend the reason_prefix context; "
            f"got: {msg!r}"
        )
        # Generic message should still be present (reason_prefix PREPENDS,
        # not replaces):
        assert "nonexistent/subpath" in msg, (
            f"Generic message should follow reason_prefix; got: {msg!r}"
        )

    def test_no_reason_prefix_omits_prefix(self):
        """Absent reason_prefix → skip message is the generic form only."""
        with pytest.raises(pytest.skip.Exception) as exc_info:
            require_monorepo_root("nonexistent/another/subpath")
        msg = str(exc_info.value)
        # No colon-based prefix pattern at the start
        assert not msg.startswith(":"), (
            f"No-prefix skip should not start with bare colon; got: {msg!r}"
        )
