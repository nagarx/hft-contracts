"""Tests for hft_contracts.signal_manifest (Phase 6 6B.5 co-move).

Mirrors the contract-level subset of backtester tests/test_signal_manifest.py.
Tests that exercise the full backtester pipeline (BacktestData.from_signal_dir,
engine integration) stay in lob-backtester/tests/test_signal_manifest.py. This
file exercises the dataclass + JSON parser + validate() + feature_set_ref
regex gate — all of which are contract-plane concerns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hft_contracts.signal_manifest import (
    ContractError,
    SignalManifest,
    CLASSIFICATION_REQUIRED,
    REGRESSION_REQUIRED,
    HYBRID_REQUIRED,
)


def _create_signal_dir(
    tmp_path: Path,
    *,
    n: int = 100,
    predictions: bool = False,
    returns: bool = False,
    metadata: dict = None,
) -> Path:
    """Create a minimal signal directory."""
    rng = np.random.RandomState(42)
    d = tmp_path / "signals"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "prices.npy", rng.uniform(100, 200, size=n).astype(np.float64))
    if predictions:
        np.save(d / "predictions.npy", rng.choice([0, 1, 2], size=n).astype(np.int64))
    if returns:
        np.save(d / "predicted_returns.npy", (rng.randn(n) * 5.0).astype(np.float64))
    if metadata is not None:
        with open(d / "signal_metadata.json", "w") as f:
            json.dump(metadata, f)
    return d


class TestSignalTypeDetection:
    def test_classification(self, tmp_path: Path):
        d = _create_signal_dir(tmp_path, predictions=True)
        m = SignalManifest.from_signal_dir(d)
        assert m.signal_type == "classification"
        assert set(CLASSIFICATION_REQUIRED).issubset(set(m.required_files))

    def test_regression(self, tmp_path: Path):
        d = _create_signal_dir(tmp_path, returns=True)
        m = SignalManifest.from_signal_dir(d)
        assert m.signal_type == "regression"
        assert set(REGRESSION_REQUIRED).issubset(set(m.required_files))

    def test_hybrid(self, tmp_path: Path):
        d = _create_signal_dir(tmp_path, predictions=True, returns=True)
        m = SignalManifest.from_signal_dir(d)
        assert m.signal_type == "hybrid"
        assert set(HYBRID_REQUIRED).issubset(set(m.required_files))


class TestMetadataParse:
    def test_from_metadata(self, tmp_path: Path):
        meta = {
            "model_type": "tlob",
            "split": "test",
            "total_samples": 100,
            "horizons": [10, 60, 300],
            "exported_at": "2026-03-17T00:00:00Z",
            "metrics": {"r2": 0.464, "ic": 0.677},
        }
        d = _create_signal_dir(tmp_path, predictions=True, metadata=meta)
        m = SignalManifest.from_signal_dir(d)
        assert m.model_type == "tlob"
        assert m.split == "test"
        assert m.n_samples == 100
        assert m.horizons == [10, 60, 300]
        assert m.model_metrics == {"r2": 0.464, "ic": 0.677}

    def test_from_files_no_metadata(self, tmp_path: Path):
        """Without signal_metadata.json, manifest infers from numpy shapes."""
        d = _create_signal_dir(tmp_path, predictions=True)
        m = SignalManifest.from_signal_dir(d)
        assert m.n_samples == 100
        assert m.model_type == "unknown"


class TestFeatureSetRefRegex:
    """Phase 6 6A.9 producer/consumer symmetry — the regex gate here must
    match `hft-ops/stages/signal_export.py::_harvest_feature_set_ref`.
    """

    def test_valid_lowercase_hex_64(self, tmp_path: Path):
        meta = {
            "total_samples": 100,
            "feature_set_ref": {
                "name": "nvda_98_stable_v1",
                "content_hash": "a" * 64,
            },
        }
        d = _create_signal_dir(tmp_path, predictions=True, metadata=meta)
        m = SignalManifest.from_signal_dir(d)
        assert m.feature_set_ref == {"name": "nvda_98_stable_v1", "content_hash": "a" * 64}

    def test_uppercase_hex_rejected(self, tmp_path: Path):
        meta = {
            "total_samples": 100,
            "feature_set_ref": {"name": "x", "content_hash": "A" * 64},
        }
        d = _create_signal_dir(tmp_path, predictions=True, metadata=meta)
        m = SignalManifest.from_signal_dir(d)
        assert m.feature_set_ref is None

    def test_short_hash_rejected(self, tmp_path: Path):
        meta = {
            "total_samples": 100,
            "feature_set_ref": {"name": "x", "content_hash": "a" * 32},
        }
        d = _create_signal_dir(tmp_path, predictions=True, metadata=meta)
        m = SignalManifest.from_signal_dir(d)
        assert m.feature_set_ref is None

    def test_missing_name_rejected(self, tmp_path: Path):
        meta = {
            "total_samples": 100,
            "feature_set_ref": {"content_hash": "a" * 64},
        }
        d = _create_signal_dir(tmp_path, predictions=True, metadata=meta)
        m = SignalManifest.from_signal_dir(d)
        assert m.feature_set_ref is None

    def test_absent_feature_set_ref(self, tmp_path: Path):
        meta = {"total_samples": 100}
        d = _create_signal_dir(tmp_path, predictions=True, metadata=meta)
        m = SignalManifest.from_signal_dir(d)
        assert m.feature_set_ref is None


class TestValidate:
    def test_valid_signal_dir(self, tmp_path: Path):
        d = _create_signal_dir(tmp_path, predictions=True)
        m = SignalManifest.from_signal_dir(d)
        warnings = m.validate(d)
        # Non-critical warnings are OK (e.g., optional files missing).
        # validate() must not raise.
        assert isinstance(warnings, list)

    def test_missing_required_raises(self, tmp_path: Path):
        """A manifest expecting predictions.npy raises when file is gone."""
        d = _create_signal_dir(tmp_path, predictions=True)
        m = SignalManifest.from_signal_dir(d)
        # Delete the required file
        (d / "predictions.npy").unlink()
        with pytest.raises(ContractError, match="Required signal file missing"):
            m.validate(d)

    def test_shape_mismatch_raises(self, tmp_path: Path):
        d = _create_signal_dir(tmp_path, predictions=True)
        # Overwrite predictions.npy with a different shape
        np.save(d / "predictions.npy", np.array([0, 1, 2], dtype=np.int64))
        m = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError, match="Shape mismatch|Sample count"):
            m.validate(d)

    def test_nan_in_prices_raises(self, tmp_path: Path):
        d = _create_signal_dir(tmp_path, predictions=True)
        prices = np.load(d / "prices.npy")
        prices[0] = np.nan
        np.save(d / "prices.npy", prices)
        m = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError, match="Non-finite values"):
            m.validate(d)


class TestSummary:
    def test_summary_includes_key_fields(self, tmp_path: Path):
        meta = {
            "model_type": "tlob",
            "split": "test",
            "total_samples": 100,
            "horizons": [10, 60, 300],
            "metrics": {"r2": 0.464},
        }
        d = _create_signal_dir(tmp_path, predictions=True, metadata=meta)
        m = SignalManifest.from_signal_dir(d)
        summary = m.summary()
        assert "classification" in summary
        assert "tlob" in summary
        assert "100" in summary
        assert "0.4640" in summary  # r2 formatted


class TestRev2PrePushHygiene:
    """REV 2 pre-push architectural hygiene regression tests (2026-04-20).

    These tests lock invariants established by Stage 0 of the hft-contracts
    public-push plan:

    - F1: ``ContractError`` is a SINGLE class (was previously defined twice
      in validation.py and signal_manifest.py — consumers catching one
      missed errors from the other).
    - F8: ``CONTENT_HASH_RE`` (public) is the canonical regex; the old
      ``_CONTENT_HASH_RE`` name is a module-level alias for pre-REV-2
      importers, scheduled for removal 2026-10-31.

    See ``/Users/knight/.claude/plans/gentle-brewing-quail.md`` (REV 2 plan).
    """

    def test_contract_error_is_single_class_across_modules(self):
        """F1: ``hft_contracts.validation.ContractError`` and
        ``hft_contracts.signal_manifest.ContractError`` are the SAME class
        object (not two independent classes). Consumers catching the
        package-level ``from hft_contracts import ContractError`` MUST
        also catch errors raised by ``SignalManifest.validate()``.
        """
        import hft_contracts
        import hft_contracts.validation as v
        import hft_contracts.signal_manifest as sm

        assert v.ContractError is sm.ContractError, (
            "F1 regression: validation.ContractError and "
            "signal_manifest.ContractError diverged — consumers catching "
            "one will silently miss errors from the other."
        )
        assert hft_contracts.ContractError is v.ContractError
        assert hft_contracts.ContractError is sm.ContractError

    def test_signal_manifest_validate_raises_package_level_contract_error(
        self, tmp_path: Path,
    ):
        """F1 end-to-end: catching the package-level ContractError catches
        errors raised from within SignalManifest.validate() (the pre-REV-2
        bug was this DID NOT catch because the classes were independent).
        """
        from hft_contracts import ContractError as PackageLevel

        # signal_dir missing required prices.npy → ContractError
        empty_dir = tmp_path / "empty_signals"
        empty_dir.mkdir()
        manifest = SignalManifest(
            signal_type="classification",
            model_type="unknown",
            split="test",
            n_samples=0,
            required_files=["prices.npy"],
        )

        with pytest.raises(PackageLevel):
            manifest.validate(empty_dir)

    def test_content_hash_re_public_alias(self):
        """F8: ``CONTENT_HASH_RE`` is the public name; ``_CONTENT_HASH_RE``
        is retained as a DEPRECATED alias pointing at the same compiled
        pattern.
        """
        from hft_contracts import CONTENT_HASH_RE as package_level
        from hft_contracts.signal_manifest import (
            CONTENT_HASH_RE as public,
            _CONTENT_HASH_RE as legacy_alias,
        )

        # Same object at all three access paths.
        assert public is package_level
        assert public is legacy_alias, (
            "F8 regression: _CONTENT_HASH_RE drifted from CONTENT_HASH_RE. "
            "Alias must point at the same compiled pattern until 2026-10-31."
        )

        # Functional: accepts 64-char lowercase hex only.
        assert public.match("a" * 64)
        assert public.match("0" * 64)
        assert not public.match("A" * 64)  # uppercase rejected
        assert not public.match("g" * 64)  # non-hex rejected
        assert not public.match("a" * 63)  # too short
        assert not public.match("a" * 65)  # too long
