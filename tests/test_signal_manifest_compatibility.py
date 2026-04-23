"""Phase II SignalManifest integration tests — CompatibilityContract + calibration precedence.

Locks the D1/D10/D11 fixes from plan v2.0:
    D1 signal_metadata.json now carries shape-determining fields (via CompatibilityContract)
    D10 calibrated_returns.npy precedence is manifest-driven, not file-existence
    D11 SignalManifest.validate performs 3-way fingerprint check + strict mode

Each test constructs a minimal signal directory on disk under ``tmp_path`` and
drives ``SignalManifest.validate`` through the specific branch being locked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from hft_contracts.compatibility import CompatibilityContract
from hft_contracts.signal_manifest import SignalManifest
from hft_contracts.validation import ContractError


def _base_contract(**overrides: Any) -> CompatibilityContract:
    defaults = dict(
        contract_version="2.2",
        schema_version="2.2",
        feature_count=98,
        window_size=100,
        feature_layout="default",
        data_source="mbo_lob",
        label_strategy_hash="a" * 64,
        calibration_method=None,
        primary_horizon_idx=0,
        horizons=(10, 60, 300),
        normalization_strategy="none",
    )
    defaults.update(overrides)
    return CompatibilityContract(**defaults)


def _write_classification_signal_dir(
    path: Path,
    contract: CompatibilityContract | None,
    n_samples: int = 16,
    calibration_method: str | None = None,
    write_calibrated_npy: bool = False,
    write_predicted_returns_npy: bool = False,
    manipulate: str | None = None,
) -> Path:
    """Produce a minimal signal directory on disk.

    ``manipulate``:
      - "tamper_block": mutate compatibility block AFTER computing fingerprint
      - "tamper_fp": mutate fingerprint AFTER serializing
      - "strip_block": drop compatibility block but KEEP fingerprint
      - "strip_fp": keep block but drop fingerprint
    """
    path.mkdir(parents=True, exist_ok=True)

    # Classification NPYs — required set: prices, predictions, labels
    np.save(path / "prices.npy", np.abs(np.random.RandomState(0).randn(n_samples)) + 100.0)
    np.save(path / "predictions.npy", np.random.RandomState(1).randint(0, 3, size=n_samples).astype(np.int32))
    np.save(path / "labels.npy", np.random.RandomState(2).randint(0, 3, size=n_samples).astype(np.int32))

    if write_predicted_returns_npy:
        np.save(path / "predicted_returns.npy", np.random.RandomState(3).randn(n_samples))
    if write_calibrated_npy:
        np.save(path / "calibrated_returns.npy", np.random.RandomState(4).randn(n_samples))

    meta: Dict[str, Any] = {
        "signal_type": "classification",
        "model_type": "hmhp",
        "split": "test",
        "total_samples": n_samples,
        "horizons": [10, 60, 300],
        "exported_at": "2026-04-20T00:00:00+00:00",
        "checkpoint": "/tmp/ckpt.pt",
    }
    if calibration_method is not None:
        meta["calibration_method"] = calibration_method

    if contract is not None:
        block = {
            "contract_version": contract.contract_version,
            "schema_version": contract.schema_version,
            "feature_count": contract.feature_count,
            "window_size": contract.window_size,
            "feature_layout": contract.feature_layout,
            "data_source": contract.data_source,
            "label_strategy_hash": contract.label_strategy_hash,
            "calibration_method": contract.calibration_method,
            "primary_horizon_idx": contract.primary_horizon_idx,
            "horizons": list(contract.horizons) if contract.horizons else None,
            "normalization_strategy": contract.normalization_strategy,
        }
        fp = contract.fingerprint()

        if manipulate == "tamper_block":
            block["feature_count"] = 9999  # diverges from fingerprint
        elif manipulate == "tamper_fp":
            fp = "0" * 64
        elif manipulate == "strip_block":
            block = None
        elif manipulate == "strip_fp":
            fp = None

        if block is not None:
            meta["compatibility"] = block
        if fp is not None:
            meta["compatibility_fingerprint"] = fp

    (path / "signal_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path


class TestLegacySignalBackCompat:
    """Pre-Phase-II manifest (no compatibility block) — non-strict loads with warning."""

    def test_legacy_manifest_lenient_mode_passes_with_warning(self, tmp_path):
        d = _write_classification_signal_dir(tmp_path / "legacy", contract=None)
        manifest = SignalManifest.from_signal_dir(d)
        warnings = manifest.validate(d, strict=False)
        # Exactly one warning mentioning the missing compatibility
        matching = [w for w in warnings if "CompatibilityContract" in w or "compatibility" in w.lower()]
        assert len(matching) >= 1, f"Expected legacy-manifest warning, got: {warnings}"

    def test_legacy_manifest_strict_mode_raises(self, tmp_path):
        d = _write_classification_signal_dir(tmp_path / "legacy_strict", contract=None)
        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError, match=r"(Phase II|strict|CompatibilityContract)"):
            manifest.validate(d, strict=True)


class TestPhaseIISignalHappyPath:
    def test_producer_fingerprint_matches_stored(self, tmp_path):
        c = _base_contract()
        d = _write_classification_signal_dir(tmp_path / "happy", contract=c)
        manifest = SignalManifest.from_signal_dir(d)
        # no expected_contract → only tamper-detection check runs, passes
        warnings = manifest.validate(d)
        # No "legacy" warnings since compatibility is present
        assert not any("CompatibilityContract" in w for w in warnings)

    def test_expected_contract_matches_passes(self, tmp_path):
        c = _base_contract()
        d = _write_classification_signal_dir(tmp_path / "match", contract=c)
        manifest = SignalManifest.from_signal_dir(d)
        manifest.validate(d, expected_contract=c)  # does not raise


class TestTamperDetection:
    def test_tampered_block_raises(self, tmp_path):
        """Mutating the compatibility block without updating the fingerprint raises."""
        c = _base_contract()
        d = _write_classification_signal_dir(
            tmp_path / "tamper_block", contract=c, manipulate="tamper_block"
        )
        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError, match="TAMPERED"):
            manifest.validate(d)

    def test_strip_block_keep_fp_raises(self, tmp_path):
        """compatibility_fingerprint without compatibility block = tamper indicator."""
        c = _base_contract()
        d = _write_classification_signal_dir(
            tmp_path / "strip_block", contract=c, manipulate="strip_block"
        )
        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError, match=r"(cannot re-verify|tamper)"):
            manifest.validate(d)

    def test_strip_fp_keep_block_raises(self, tmp_path):
        """compatibility block without fingerprint = malformed."""
        c = _base_contract()
        d = _write_classification_signal_dir(
            tmp_path / "strip_fp", contract=c, manipulate="strip_fp"
        )
        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError, match="malformed"):
            manifest.validate(d)


class TestExpectedContractMismatch:
    def test_feature_count_mismatch_raises_with_diff(self, tmp_path):
        producer = _base_contract(feature_count=98)
        consumer = _base_contract(feature_count=148)
        d = _write_classification_signal_dir(tmp_path / "skew_feat", contract=producer)
        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError) as excinfo:
            manifest.validate(d, expected_contract=consumer)
        assert "feature_count" in str(excinfo.value)

    def test_calibration_mismatch_raises(self, tmp_path):
        producer = _base_contract(calibration_method=None)
        consumer = _base_contract(calibration_method="variance_match")
        d = _write_classification_signal_dir(tmp_path / "skew_cal", contract=producer)
        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError, match="CompatibilityContract"):
            manifest.validate(d, expected_contract=consumer)

    def test_horizons_mismatch_raises(self, tmp_path):
        producer = _base_contract(horizons=(10, 60, 300))
        consumer = _base_contract(horizons=(5, 30, 120))
        d = _write_classification_signal_dir(tmp_path / "skew_horiz", contract=producer)
        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError) as excinfo:
            manifest.validate(d, expected_contract=consumer)
        assert "horizons" in str(excinfo.value)


class TestCalibrationPrecedence:
    """D10 fix: calibration file existence must match manifest claim."""

    def test_orphan_calibrated_file_raises(self, tmp_path):
        """calibrated_returns.npy exists + manifest.calibration_method=None → raise."""
        c = _base_contract(calibration_method=None)
        d = _write_classification_signal_dir(
            tmp_path / "orphan",
            contract=c,
            calibration_method=None,
            write_calibrated_npy=True,
            write_predicted_returns_npy=True,
        )
        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError, match=r"[Oo]rphan calibrated"):
            manifest.validate(d)

    def test_missing_calibrated_file_but_claimed_raises(self, tmp_path):
        """calibration_method set but calibrated_returns.npy absent → raise."""
        c = _base_contract(calibration_method="variance_match")
        d = _write_classification_signal_dir(
            tmp_path / "missing_cal",
            contract=c,
            calibration_method="variance_match",
            write_calibrated_npy=False,
        )
        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError, match=r"missing"):
            manifest.validate(d)

    def test_calibration_aligned_passes(self, tmp_path):
        """calibration_method set AND file exists → pass."""
        c = _base_contract(calibration_method="variance_match")
        d = _write_classification_signal_dir(
            tmp_path / "cal_ok",
            contract=c,
            calibration_method="variance_match",
            write_calibrated_npy=True,
            write_predicted_returns_npy=True,
        )
        manifest = SignalManifest.from_signal_dir(d)
        manifest.validate(d)  # does not raise

    def test_no_calibration_no_file_passes(self, tmp_path):
        """Neither calibration_method nor calibrated file → classic classification path OK."""
        c = _base_contract(calibration_method=None)
        d = _write_classification_signal_dir(
            tmp_path / "no_cal", contract=c, calibration_method=None, write_calibrated_npy=False
        )
        manifest = SignalManifest.from_signal_dir(d)
        manifest.validate(d)


class TestAlignedFilesIncludesCalibrated:
    """Shape-alignment check now covers calibrated_returns.npy (safeguard layer)."""

    def test_calibrated_wrong_shape_raises(self, tmp_path):
        c = _base_contract(calibration_method="variance_match")
        d = _write_classification_signal_dir(
            tmp_path / "bad_shape",
            contract=c,
            calibration_method="variance_match",
            write_calibrated_npy=True,
            write_predicted_returns_npy=True,
            n_samples=16,
        )
        # Overwrite calibrated with a mis-shaped array
        np.save(d / "calibrated_returns.npy", np.random.RandomState(5).randn(99))
        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError, match=r"[Ss]hape mismatch"):
            manifest.validate(d)


class TestBackCompatSignature:
    """Existing callers that invoke ``validate(signal_dir)`` without new kwargs must still work."""

    def test_old_signature_still_valid(self, tmp_path):
        c = _base_contract()
        d = _write_classification_signal_dir(tmp_path / "oldsig", contract=c)
        manifest = SignalManifest.from_signal_dir(d)
        # positional-only call — no expected_contract, no strict
        warnings = manifest.validate(d)
        assert isinstance(warnings, list)


class TestExpectedFieldsPartialAssertion:
    """Phase II hardening (2026-04-20): ``expected_fields`` partial-assertion API.

    Architecturally-honest alternative to full ``expected_contract`` for consumers
    whose config only covers a subset of the 11 shape-determining fields (e.g.,
    backtester only knows ``primary_horizon_idx``). Every field present in the
    dict must match the manifest's CompatibilityContract.<field> or ContractError
    is raised with a field-level diff. Typo'd keys raise ValueError (fail-loud).
    """

    def test_matching_single_field_passes(self, tmp_path):
        c = _base_contract(primary_horizon_idx=0)
        d = _write_classification_signal_dir(tmp_path / "ef_match", contract=c)
        manifest = SignalManifest.from_signal_dir(d)
        # Consumer asserts ONLY primary_horizon_idx — defers all other fields.
        manifest.validate(d, expected_fields={"primary_horizon_idx": 0})

    def test_matching_multiple_fields_passes(self, tmp_path):
        c = _base_contract(feature_count=98, window_size=100, primary_horizon_idx=0)
        d = _write_classification_signal_dir(tmp_path / "ef_multi", contract=c)
        manifest = SignalManifest.from_signal_dir(d)
        manifest.validate(
            d,
            expected_fields={
                "feature_count": 98,
                "window_size": 100,
                "primary_horizon_idx": 0,
            },
        )

    def test_mismatched_field_raises(self, tmp_path):
        c = _base_contract(primary_horizon_idx=0)
        d = _write_classification_signal_dir(tmp_path / "ef_mismatch", contract=c)
        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError) as excinfo:
            manifest.validate(d, expected_fields={"primary_horizon_idx": 1})
        msg = str(excinfo.value)
        assert "primary_horizon_idx" in msg
        # Field-level diff must surface both sides of the mismatch
        assert "0" in msg and "1" in msg

    def test_unknown_key_raises_valueerror(self, tmp_path):
        """Typo'd keys fail loud — no silent-accept of misspelled assertions."""
        c = _base_contract()
        d = _write_classification_signal_dir(tmp_path / "ef_typo", contract=c)
        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ValueError, match="horizen_idx"):
            manifest.validate(d, expected_fields={"horizen_idx": 0})  # typo

    def test_list_vs_tuple_horizons_interop(self, tmp_path):
        """Consumer-supplied list normalizes against tuple-stored horizons."""
        c = _base_contract(horizons=(10, 60, 300))
        d = _write_classification_signal_dir(tmp_path / "ef_list", contract=c)
        manifest = SignalManifest.from_signal_dir(d)
        # Consumer passes list — should compare equal to stored tuple
        manifest.validate(d, expected_fields={"horizons": [10, 60, 300]})

    def test_legacy_manifest_with_expected_fields_warns_not_raises(self, tmp_path):
        """Pre-Phase-II manifest + expected_fields → warning (check SKIPPED), not raise."""
        d = _write_classification_signal_dir(tmp_path / "ef_legacy", contract=None)
        manifest = SignalManifest.from_signal_dir(d)
        warnings = manifest.validate(
            d,
            expected_fields={"primary_horizon_idx": 0},
            strict=False,
        )
        # Must surface that expected_fields check did not run
        matching = [
            w for w in warnings
            if "expected_fields" in w or "version-skew" in w or "SKIPPED" in w
        ]
        assert len(matching) >= 1, (
            f"Expected a warning noting that expected_fields check was SKIPPED on "
            f"legacy manifest, got: {warnings}"
        )

    def test_expected_fields_and_expected_contract_both_checked(self, tmp_path):
        """Both gates apply: expected_contract first (fingerprint), expected_fields second."""
        c = _base_contract(primary_horizon_idx=0)
        d = _write_classification_signal_dir(tmp_path / "ef_both", contract=c)
        manifest = SignalManifest.from_signal_dir(d)
        # Both match → passes
        manifest.validate(
            d,
            expected_contract=c,
            expected_fields={"primary_horizon_idx": 0},
        )
        # expected_contract matches but expected_fields mismatches → raises on partial
        with pytest.raises(ContractError, match="primary_horizon_idx"):
            manifest.validate(
                d,
                expected_contract=c,
                expected_fields={"primary_horizon_idx": 99},
            )

    def test_calibration_method_mismatch_raises(self, tmp_path):
        """Partial assertion on calibration_method catches consumer mis-configuration."""
        c = _base_contract(calibration_method="variance_match")
        d = _write_classification_signal_dir(
            tmp_path / "ef_cal",
            contract=c,
            calibration_method="variance_match",
            write_calibrated_npy=True,
            write_predicted_returns_npy=True,
        )
        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError, match="calibration_method"):
            manifest.validate(d, expected_fields={"calibration_method": None})

    # --- Phase II hardening post-audit (2026-04-21) SB-D: empty-dict reject ---

    def test_empty_expected_fields_raises(self, tmp_path):
        """Empty dict is a caller logic error — fail loud per hft-rules §5.

        Previously: `expected_fields={}` silently no-op'd (unknown-keys check
        passed with empty set; for-loop iterated zero times). Now: raises
        ValueError early so the caller sees they produced no assertions.
        """
        c = _base_contract()
        d = _write_classification_signal_dir(tmp_path / "ef_empty", contract=c)
        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ValueError, match="non-empty dict"):
            manifest.validate(d, expected_fields={})


class TestRequireFingerprintKwarg:
    """Phase A (2026-04-23): ``validate(require_fingerprint=True)`` kwarg.

    Stricter than ``strict=True`` — opt-in guard that rejects ANY manifest
    whose ``compatibility_fingerprint`` is None, independent of whether the
    compatibility block is present or absent. Intended for post-Phase-A
    consumers that want to guarantee every manifest they ingest has a
    verifiable producer-side fingerprint.

    Default ``False`` preserves all pre-Phase-A call-site behavior.
    """

    def _legacy_signal_dir(self, tmp_path: Path) -> Path:
        """Build a pre-Phase-II signal directory (no compatibility block / fp)."""
        d = tmp_path / "legacy"
        d.mkdir()
        meta = {
            "model_type": "tlob",
            "split": "val",
            "total_samples": 10,
        }
        (d / "signal_metadata.json").write_text(json.dumps(meta))
        np.save(d / "predictions.npy", np.zeros(10, dtype=np.int64))
        np.save(d / "labels.npy", np.zeros(10, dtype=np.int64))
        np.save(d / "prices.npy", np.ones(10) * 100.0)
        np.save(d / "spreads.npy", np.zeros(10))
        return d

    def test_default_false_preserves_legacy_behavior(self, tmp_path):
        """Default ``require_fingerprint=False`` — pre-Phase-II manifest loads
        with legacy warning, no raise. Matches pre-Phase-A semantics exactly.
        """
        d = self._legacy_signal_dir(tmp_path)
        manifest = SignalManifest.from_signal_dir(d)
        warnings_out = manifest.validate(d)
        assert any("Legacy signal manifest" in w for w in warnings_out)

    def test_require_fingerprint_true_rejects_legacy_manifest(self, tmp_path):
        """``require_fingerprint=True`` rejects when fingerprint is None."""
        d = self._legacy_signal_dir(tmp_path)
        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError, match="require_fingerprint=True"):
            manifest.validate(d, require_fingerprint=True)

    def test_require_fingerprint_true_accepts_phase_ii_manifest(self, tmp_path):
        """``require_fingerprint=True`` with a valid fingerprint: no raise."""
        c = _base_contract()
        d = _write_classification_signal_dir(tmp_path / "rf_ok", contract=c)
        manifest = SignalManifest.from_signal_dir(d)
        manifest.validate(d, require_fingerprint=True)
