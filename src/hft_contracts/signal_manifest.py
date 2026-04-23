"""Signal manifest — validates signal exports at load time.

Phase 6 6B.5 (2026-04-17): Moved from `lobbacktest.data.signal_manifest`
to `hft_contracts.signal_manifest` so the dataclass + parser lives on the
contract plane alongside `FeatureSet`, `Provenance`, `LabelContract`, and
`canonical_hash`. Three consumers (trainer producer, backtester consumer,
hft-ops harvester) all align on the same JSON schema; home-alongside-the-
schema is the correct SSoT pattern.

The manifest is parsed from signal_metadata.json (written by the trainer's
SignalExporter) or inferred from file existence when metadata is absent.

Usage:
    from hft_contracts.signal_manifest import SignalManifest
    manifest = SignalManifest.from_signal_dir(signal_dir)
    warnings = manifest.validate(signal_dir)
    # warnings is List[str] of non-critical issues
    # Raises ContractError for critical issues (shape mismatch, missing files)

Reference:
    BACKTESTER_AUDIT_PLAN.md § M6 (from_signal_dir loads without validation)
    pipeline_contract.toml § [signals]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# REV 2 pre-push (2026-04-20): ``ContractError`` is now a single class
# owned by ``hft_contracts.validation``. Previously this module defined
# its own independent ``ContractError`` class, which meant consumers
# catching ``from hft_contracts import ContractError`` (the validation
# one) missed errors raised by ``SignalManifest.validate()`` (the
# signal_manifest one). Both now resolve to the SAME class. Kept in the
# module-local namespace + ``__all__`` so ``from hft_contracts.signal_manifest
# import ContractError`` remains a valid access path.
from hft_contracts.validation import ContractError

# Phase II (2026-04-20): CompatibilityContract embedded in the manifest for
# 3-way fingerprint validation. Imported here (intra-package, no cycle risk)
# so ``validate()`` can call ``contract.fingerprint()`` at check time.
from hft_contracts.compatibility import CompatibilityContract


# Phase 6 6A.9 (2026-04-17): module-level regex for content_hash validation.
# Matches hft_contracts.canonical_hash.sha256_hex output format (64 lowercase
# hex chars). Contract: pipeline_contract.toml:1211 specifies this pattern.
# Module-level (not per-call) so re.compile runs once at import time.
#
# REV 2 pre-push (2026-04-20): renamed from module-private ``_CONTENT_HASH_RE``
# to public ``CONTENT_HASH_RE``. The underscore-prefix was a mis-classification
# since ``hft-ops/src/hft_ops/stages/signal_export.py`` imports it across the
# module boundary.
#
# Legacy access via ``_CONTENT_HASH_RE`` is gated through the module-level
# ``__getattr__`` below — consumers receive a one-time ``DeprecationWarning``
# citing the migration path + 2026-10-31 removal deadline. This matches the
# Phase 6 6B.5 backtester-shim pattern and the REV 2 ``_atomic_io`` shim so
# every deprecated name in the contract plane has uniform migration telemetry.
CONTENT_HASH_RE = re.compile(r"^[a-f0-9]{64}$")

# REV 2 pre-push follow-up (2026-04-20): deprecation telemetry for
# ``_CONTENT_HASH_RE``. The alias pointer (via ``__getattr__``) returns the
# SAME compiled pattern, so consumers observe identical matching behavior;
# only the import path emits the warning.
_CONTENT_HASH_RE_REMOVAL_DATE = "2026-10-31"
_LEGACY_NAMES_WARNED: set[str] = set()


# --- Signal file definitions ---

CLASSIFICATION_REQUIRED = ["prices.npy", "predictions.npy"]
CLASSIFICATION_OPTIONAL = [
    "labels.npy",
    "agreement_ratio.npy",
    "confirmation_score.npy",
    "spreads.npy",
]

REGRESSION_REQUIRED = ["prices.npy", "predicted_returns.npy"]
REGRESSION_OPTIONAL = ["regression_labels.npy", "spreads.npy"]

HYBRID_REQUIRED = ["prices.npy", "predictions.npy", "predicted_returns.npy"]
HYBRID_OPTIONAL = [
    "labels.npy",
    "agreement_ratio.npy",
    "confirmation_score.npy",
    "regression_labels.npy",
    "spreads.npy",
]

# Files that must have shape[0] == N (first dimension alignment).
#
# Phase II (2026-04-20): ``calibrated_returns.npy`` added to the list so its
# shape is cross-checked against predicted_returns.npy when BOTH exist. Previously
# a stale calibrated file could silently win over a fresh predicted-returns file
# via the backtester's file-existence precedence (validation report D10). The
# shape-check closes one failure mode; the ``SignalManifest.calibration_method``
# field + ``validate()`` precedence rule close the other (orphan file without
# manifest claim).
ALIGNED_FILES = [
    "prices.npy",
    "predictions.npy",
    "labels.npy",
    "agreement_ratio.npy",
    "confirmation_score.npy",
    "spreads.npy",
    "predicted_returns.npy",
    "calibrated_returns.npy",
    "regression_labels.npy",
]


@dataclass(frozen=True)
class SignalManifest:
    """Contract for a signal export directory.

    Defines what files are expected, their shapes, and validation rules.
    Parsed from signal_metadata.json or inferred from file existence.

    Attributes:
        signal_type: "classification", "regression", or "hybrid".
        model_type: Model architecture (e.g., "hmhp", "tlob_regression").
        split: Data split ("train", "val", "test").
        n_samples: Expected number of samples (N) across all arrays.
        horizons: List of prediction horizons (e.g., [10, 60, 300]).
        required_files: Files that MUST exist (ContractError if missing).
        optional_files: Files that MAY exist (warning if missing).
        checkpoint_path: Path to model checkpoint (provenance).
        export_timestamp: When signals were exported (provenance).
        model_metrics: Training metrics (R², IC, DA) for reference.
        feature_set_ref: Optional reference to the FeatureSet registry
            entry used at trainer time (Phase 4 Batch 4c.4). None iff
            trainer did not use DataConfig.feature_set.
    """

    signal_type: str
    model_type: str
    split: str
    n_samples: int
    horizons: Optional[List[int]] = None
    required_files: List[str] = field(default_factory=list)
    optional_files: List[str] = field(default_factory=list)
    checkpoint_path: Optional[str] = None
    export_timestamp: Optional[str] = None
    model_metrics: Optional[Dict[str, float]] = None
    # Phase 4 Batch 4c.4 (2026-04-16): optional reference to the
    # FeatureSet registry entry used at trainer time. Propagated
    # trainer → signal_metadata.json → here read-only. Backtester
    # does NOT recompute content_hash (integrity is the resolver's job
    # at trainer load time; recomputation would create a 4th
    # canonical-form site per PA §13.4.2).
    feature_set_ref: Optional[Dict[str, str]] = None

    # Phase II (2026-04-20): CompatibilityContract fingerprint for cross-module
    # producer ↔ consumer validation. ``compatibility`` carries the 11-key
    # shape-determining fields; ``compatibility_fingerprint`` is the stored
    # producer-side SHA-256 that validate() re-derives (tamper detection).
    # Both None on pre-Phase-II signal directories — validate() warns unless
    # ``strict=True`` is passed.
    # See hft_contracts.compatibility for the contract surface.
    compatibility: Optional["CompatibilityContract"] = None
    compatibility_fingerprint: Optional[str] = None
    # Phase II D10 fix: manifest-declared calibration state. When non-None,
    # validate() requires calibrated_returns.npy to exist (and vice-versa —
    # orphan file with None claim raises). Backtester consumes this as the
    # authoritative precedence gate (file-existence alone is not sufficient).
    calibration_method: Optional[str] = None
    # Phase II: data source tag, echoed from compatibility.data_source for
    # quick inspection without parsing the nested contract.
    data_source: Optional[str] = None

    @classmethod
    def from_signal_dir(cls, signal_dir: Path) -> "SignalManifest":
        """Parse signal_metadata.json or infer manifest from files.

        Args:
            signal_dir: Path to directory containing .npy signal files.

        Returns:
            SignalManifest describing the signal directory.
        """
        signal_dir = Path(signal_dir)
        metadata_path = signal_dir / "signal_metadata.json"

        if metadata_path.exists():
            return cls._from_metadata(signal_dir, metadata_path)
        return cls._from_files(signal_dir)

    @classmethod
    def _from_metadata(cls, signal_dir: Path, metadata_path: Path) -> "SignalManifest":
        """Parse from signal_metadata.json."""
        with open(metadata_path) as f:
            meta = json.load(f)

        # Detect signal type from metadata and files
        signal_type = cls._detect_signal_type(signal_dir)

        # Extract fields with safe defaults
        model_type = meta.get("model_type", "unknown")
        split = meta.get("split", "unknown")
        n_samples = meta.get("total_samples", 0)
        horizons = meta.get("horizons")
        checkpoint = meta.get("checkpoint")
        timestamp = meta.get("exported_at")

        # Extract model metrics (nested under "metrics" key)
        metrics_dict = meta.get("metrics")
        if isinstance(metrics_dict, dict):
            model_metrics = {
                k: float(v) for k, v in metrics_dict.items() if isinstance(v, (int, float))
            }
        else:
            model_metrics = None

        # Phase 4 Batch 4c.4: read-only propagation of FeatureSet registry
        # reference. Validates shape ({"name": str, "content_hash": str})
        # AND content_hash format (SHA-256 hex, 64 lowercase chars) via
        # CONTENT_HASH_RE. Does NOT recompute content_hash.
        # Phase 6 6A.9 (2026-04-17): regex gate matches hft-ops harvester
        # — producer/consumer symmetry prevents asymmetric acceptance.
        feature_set_ref: Optional[Dict[str, str]] = None
        raw_fsr = meta.get("feature_set_ref")
        if isinstance(raw_fsr, dict):
            name = raw_fsr.get("name")
            content_hash = raw_fsr.get("content_hash")
            if (
                isinstance(name, str)
                and isinstance(content_hash, str)
                and CONTENT_HASH_RE.match(content_hash)
            ):
                feature_set_ref = {"name": name, "content_hash": content_hash}

        required, optional = cls._files_for_type(signal_type)

        # Phase II (2026-04-20): optional compatibility contract + stored fingerprint.
        # Older signal directories won't have these — fields stay None and validate()
        # emits a DeprecationWarning in non-strict mode.
        compat_block = meta.get("compatibility")
        compatibility: Optional[CompatibilityContract] = None
        compatibility_fp: Optional[str] = meta.get("compatibility_fingerprint")
        if isinstance(compat_block, dict):
            try:
                compatibility = _compatibility_from_dict(compat_block)
            except (TypeError, KeyError, ValueError) as exc:
                # Malformed block — treat as absent so legacy directories still load.
                # validate() will raise if compatibility_fingerprint is ALSO present
                # (that combination indicates tampering, not legacy).
                compatibility = None
                # Keep compat_fp so validate() can surface the inconsistency.
                _ = exc

        calibration_method = meta.get("calibration_method")
        data_source = meta.get("data_source")

        return cls(
            signal_type=signal_type,
            model_type=model_type,
            split=split,
            n_samples=n_samples,
            horizons=horizons,
            required_files=required,
            optional_files=optional,
            checkpoint_path=checkpoint,
            export_timestamp=timestamp,
            model_metrics=model_metrics,
            feature_set_ref=feature_set_ref,
            compatibility=compatibility,
            compatibility_fingerprint=compatibility_fp,
            calibration_method=calibration_method,
            data_source=data_source,
        )

    @classmethod
    def _from_files(cls, signal_dir: Path) -> "SignalManifest":
        """Infer manifest from file existence (no metadata.json)."""
        signal_type = cls._detect_signal_type(signal_dir)

        # Infer n_samples from prices.npy
        prices_path = signal_dir / "prices.npy"
        if prices_path.exists():
            prices = np.load(prices_path)
            n_samples = prices.shape[0]
        else:
            n_samples = 0

        required, optional = cls._files_for_type(signal_type)

        return cls(
            signal_type=signal_type,
            model_type="unknown",
            split="unknown",
            n_samples=n_samples,
            required_files=required,
            optional_files=optional,
        )

    @staticmethod
    def _detect_signal_type(signal_dir: Path) -> str:
        """Detect signal type from file existence."""
        has_predictions = (signal_dir / "predictions.npy").exists()
        has_returns = (signal_dir / "predicted_returns.npy").exists()
        if has_predictions and has_returns:
            return "hybrid"
        elif has_returns:
            return "regression"
        elif has_predictions:
            return "classification"
        return "classification"  # default

    @staticmethod
    def _files_for_type(signal_type: str):
        """Return (required, optional) file lists for signal type."""
        if signal_type == "hybrid":
            return list(HYBRID_REQUIRED), list(HYBRID_OPTIONAL)
        elif signal_type == "regression":
            return list(REGRESSION_REQUIRED), list(REGRESSION_OPTIONAL)
        return list(CLASSIFICATION_REQUIRED), list(CLASSIFICATION_OPTIONAL)

    def validate(
        self,
        signal_dir: Path,
        expected_contract: Optional[CompatibilityContract] = None,
        expected_fields: Optional[Dict[str, Any]] = None,
        strict: bool = False,
        *,
        require_fingerprint: bool = False,
    ) -> List[str]:
        """Validate signal directory against this manifest.

        Args:
            signal_dir: Path to signal directory (contains NPY files + signal_metadata.json).
            expected_contract: Phase II (2026-04-20). Consumer-derived
                CompatibilityContract. When provided, validate() performs a 3-way
                fingerprint check: ``stored_fingerprint == producer.fingerprint()``
                (tamper detection) AND ``stored_fingerprint == expected.fingerprint()``
                (version-skew detection). A mismatch raises ``ContractError`` with
                a field-level diff.
            expected_fields: Phase II hardening (2026-04-20). Consumer-derived
                PARTIAL assertion — a dict of ``{CompatibilityContract-field-name: expected_value}``
                that the consumer actually knows. Architecturally cleaner than
                ``expected_contract`` for consumers whose config only covers a subset
                of the 11-field contract (e.g., a backtester that only knows
                ``primary_horizon_idx`` but defers every other field to the trainer).
                Every field present in ``expected_fields`` MUST match the manifest's
                ``compatibility.<field>`` or ContractError is raised with a
                field-level diff. Precedence: if ``expected_contract`` is supplied,
                the full fingerprint check runs first; ``expected_fields`` runs as
                an additional gate on top. Silently-ignored invalid keys are
                disallowed — any key not on ``CompatibilityContract.key_fields()``
                raises ``ValueError`` (prevents silent typos like ``horizen_idx``).
            strict: Phase II (2026-04-20). When True, legacy manifests
                (``compatibility=None``) are REJECTED with ContractError. In lenient
                mode (default), legacy manifests emit a DeprecationWarning and skip
                the 3-way check but still run shape/NaN validation. Strict mode is
                the recommended setting for production consumers after the 2-week
                back-compat window — see ``hft_contracts/CHANGELOG.md`` v2.3.0.
            require_fingerprint: Phase A (2026-04-23). Keyword-only opt-in.
                When True, raise :class:`ContractError` if
                ``compatibility_fingerprint`` is None. This is a stricter guard
                than ``strict=True`` — ``strict=True`` rejects when BOTH the
                compatibility block AND fingerprint are missing (pre-Phase-II
                legacy); ``require_fingerprint=True`` rejects when just the
                fingerprint is missing (post-Phase-II producer-path regressions
                where the block was emitted without the fingerprint, OR any
                manifest where the fingerprint is unavailable). Intended for
                post-Phase-A consumers that want to guarantee EVERY manifest
                they validate has a verifiable producer-side fingerprint.
                Default ``False`` preserves all current call-site behavior.

        Raises:
            ContractError: For critical issues:
                - Missing required files
                - Shape mismatch across ALIGNED_FILES
                - Non-finite values in required arrays
                - (Phase II) compatibility_fingerprint is set but compatibility block absent
                  (tamper indicator)
                - (Phase II) producer-fingerprint mismatch (tamper detected)
                - (Phase II) expected-fingerprint mismatch (contract version skew)
                - (Phase II) orphan calibrated_returns.npy (file exists but manifest claims no calibration)
                - (Phase II) manifest claims calibration but calibrated_returns.npy missing
                - (Phase II) strict=True + compatibility=None (legacy manifest rejected)
                - (Phase II hardening) expected_fields mismatch — consumer asserted
                  a field-level expectation that the manifest contradicts
            ValueError: If ``expected_fields`` contains a key that is not a
                ``CompatibilityContract`` field name (typo detection — fail loud,
                not silent).

        Returns:
            List of non-critical warning strings (dtype coercion, range anomalies,
            legacy-manifest deprecation notices).
        """
        signal_dir = Path(signal_dir)
        warnings: List[str] = []

        # --- Phase A (2026-04-23): opt-in require_fingerprint gate. Runs
        # BEFORE the 3-way check so operators get a clear "no fingerprint"
        # error rather than a downstream "block present but fingerprint
        # missing" malformed-manifest error. ---
        if require_fingerprint and self.compatibility_fingerprint is None:
            raise ContractError(
                "SignalManifest was validated with require_fingerprint=True "
                f"but ``compatibility_fingerprint`` is None (signal_dir={signal_dir!s}). "
                "The producer either (a) was pre-Phase-II and did not emit a "
                "fingerprint, or (b) regressed the Phase A producer-path fix "
                "(see trainer exporter.py + lobtrainer.config.paths.resolve_labels_config). "
                "To accept legacy manifests, pass require_fingerprint=False "
                "(the default) instead."
            )

        # --- Phase II: CompatibilityContract 3-way validation (runs FIRST so that
        # a version-skewed signal is rejected before any NPY-load work is done). ---
        if self.compatibility is None and self.compatibility_fingerprint is None:
            # Pre-Phase-II signal directory. Legacy back-compat path.
            if strict:
                raise ContractError(
                    "Signal manifest lacks CompatibilityContract. strict=True mode "
                    "enforces Phase II (2026-04-20) signals only. Re-export signals "
                    "with a Phase-II-aware trainer OR pass strict=False to accept "
                    "legacy R1-R8 signal directories with shape/NaN/range checks only."
                )
            warnings.append(
                "Legacy signal manifest — no CompatibilityContract present. "
                "Validation limited to shape/NaN/range. Re-export signals to gain "
                "producer↔consumer fingerprint check (Phase II, 2026-04-20)."
            )
        elif self.compatibility is not None and self.compatibility_fingerprint is None:
            raise ContractError(
                "Signal manifest has compatibility block but no compatibility_fingerprint "
                "stored alongside. This is malformed — either both or neither must be present. "
                f"compatibility={self.compatibility!r}"
            )
        elif self.compatibility is None and self.compatibility_fingerprint is not None:
            raise ContractError(
                "Signal manifest has compatibility_fingerprint "
                f"({self.compatibility_fingerprint[:16]}...) but no compatibility block — "
                "cannot re-verify the hash. Tamper indicator per Phase II contract."
            )
        else:
            # Both present — run the 3-way check.
            # Check 1: producer fingerprint matches stored (detects tampering of the block).
            recomputed = self.compatibility.fingerprint()
            if recomputed != self.compatibility_fingerprint:
                raise ContractError(
                    f"Signal manifest CompatibilityContract fingerprint TAMPERED: "
                    f"stored={self.compatibility_fingerprint[:16]}..., "
                    f"recomputed={recomputed[:16]}.... "
                    f"The compatibility block has been modified without updating the "
                    f"fingerprint, OR the canonical_hash convention has drifted. "
                    f"Contract surface: {self.compatibility.key_fields()}"
                )
            # Check 2: consumer's expected fingerprint matches stored (detects version skew).
            if expected_contract is not None:
                expected_fp = expected_contract.fingerprint()
                if expected_fp != self.compatibility_fingerprint:
                    diff = self.compatibility.diff(expected_contract)
                    raise ContractError(
                        f"Signal was produced under a different CompatibilityContract than "
                        f"the consumer expects. "
                        f"Differing fields: {diff}. "
                        f"Re-export signals OR update consumer config to match producer."
                    )

        # --- Phase II hardening: partial expected_fields assertion ---
        # Consumers whose config only covers a narrow subset of the 11-key
        # contract (e.g., backtester knows ``primary_horizon_idx`` but defers
        # all other axes to the trainer) pass a dict of the fields they actually
        # know. Typo'd keys raise ValueError (fail-loud), not silently-accept.
        if expected_fields is not None:
            # SB-D: empty dict is a caller-side logic error — fail loud, not silent.
            # A caller passing an empty dict almost certainly intended to assert
            # SOMETHING; accepting silently hides that the assertion was dropped
            # (e.g., config parsing produced an empty dict instead of None).
            # This matches hft-rules §5 "fail fast with a precise error — never
            # silently degrade." Tested in test_signal_manifest_compatibility.py.
            if not expected_fields:
                raise ValueError(
                    "expected_fields must be None or a non-empty dict. "
                    "An empty dict indicates a consumer-side logic error — the "
                    "caller intended to assert but produced no assertions. Pass "
                    "None to explicitly skip the partial-assertion gate."
                )
            if self.compatibility is None:
                # Legacy manifest — the check can't run but the consumer's intent
                # was to verify; surface that as a warning so the operator knows
                # version-skew was NOT validated.
                warnings.append(
                    "Consumer supplied expected_fields but signal manifest has no "
                    "CompatibilityContract — version-skew check SKIPPED. "
                    "Re-export signals with Phase-II-aware trainer to enable."
                )
            else:
                valid_keys = set(self.compatibility.key_fields())
                unknown = set(expected_fields.keys()) - valid_keys
                if unknown:
                    raise ValueError(
                        f"expected_fields contains keys not on CompatibilityContract: "
                        f"{sorted(unknown)}. Valid field names: {sorted(valid_keys)}. "
                        f"This is a consumer-side bug — fix the caller."
                    )
                mismatches: Dict[str, Any] = {}
                for key, expected_val in expected_fields.items():
                    actual_val = getattr(self.compatibility, key)
                    # Tuple/list interop: consumers may pass [10, 60, 300] for
                    # horizons; the contract stores (10, 60, 300). Normalize
                    # via tuple(...) for sequence fields.
                    if isinstance(actual_val, tuple) and isinstance(expected_val, list):
                        expected_cmp: Any = tuple(expected_val)
                    else:
                        expected_cmp = expected_val
                    if actual_val != expected_cmp:
                        mismatches[key] = (actual_val, expected_cmp)
                if mismatches:
                    raise ContractError(
                        f"Signal compatibility fields diverge from consumer expectations: "
                        f"{mismatches}. Either the consumer is mis-configured or the "
                        f"signal directory was produced under a different contract. "
                        f"Format: {{field: (manifest_value, expected_value)}}."
                    )

        # --- Phase II: Calibration precedence rule (D10 fix) ---
        # The orphan-file rule treats a disagreement between manifest
        # ``calibration_method`` and the presence of ``calibrated_returns.npy``
        # as a strict contract violation. validate=True applies this to ALL
        # manifests — including legacy directories (pre-Phase-II, no
        # ``compatibility`` block). If an operator is intentionally re-running
        # a legacy directory (e.g., E6 ``calibrated_conviction``), they must
        # call ``BacktestData.from_signal_dir(..., validate=False)`` to use
        # the legacy file-existence precedence path documented in that
        # method's docstring. hft-rules §8: "Never silently drop, clamp, or
        # 'fix' data without recording diagnostics." — enforcing the strict
        # rule at validate=True catches stale-file contamination that would
        # otherwise corrupt the backtest silently; opt-out via validate=False
        # is explicit.
        calibrated_path = signal_dir / "calibrated_returns.npy"
        has_file = calibrated_path.exists()
        has_claim = self.calibration_method is not None
        if has_file and not has_claim:
            raise ContractError(
                f"Orphan calibrated_returns.npy at {calibrated_path} — manifest declares "
                f"calibration_method=None (or lacks the field). This may indicate a stale "
                f"calibration file from a previous export. To intentionally load a legacy "
                f"(pre-Phase-II) signal directory, pass validate=False — that path uses "
                f"file-existence precedence for back-compat. Otherwise: re-export signals "
                f"OR delete the orphan."
            )
        if has_claim and not has_file:
            raise ContractError(
                f"Manifest declares calibration_method={self.calibration_method!r} but "
                f"calibrated_returns.npy is missing at {calibrated_path}. Re-export signals."
            )

        # 1. Check required files exist
        for fname in self.required_files:
            fpath = signal_dir / fname
            if not fpath.exists():
                raise ContractError(
                    f"Required signal file missing: {fpath}. "
                    f"Signal type '{self.signal_type}' requires: {self.required_files}"
                )

        # 2. Check optional files, warn if missing
        for fname in self.optional_files:
            if not (signal_dir / fname).exists():
                warnings.append(f"Optional file missing: {fname}")

        # 3. Load all existing arrays and check shapes
        arrays: Dict[str, np.ndarray] = {}
        for fname in ALIGNED_FILES:
            fpath = signal_dir / fname
            if fpath.exists():
                arrays[fname] = np.load(fpath)

        # 4. Shape alignment: all arrays must have same first dimension
        if arrays:
            shapes = {name: arr.shape[0] for name, arr in arrays.items()}
            unique_ns = set(shapes.values())
            if len(unique_ns) > 1:
                shape_str = ", ".join(f"{name}={n}" for name, n in shapes.items())
                raise ContractError(
                    f"Shape mismatch across signal arrays: {shape_str}. "
                    f"All arrays must have identical first dimension."
                )

            actual_n = next(iter(unique_ns))

            # 5. Metadata sample count check
            if self.n_samples > 0 and actual_n != self.n_samples:
                raise ContractError(
                    f"Sample count mismatch: signal_metadata.json says "
                    f"total_samples={self.n_samples}, but arrays have N={actual_n}"
                )

        # 6. NaN/Inf check on required arrays
        for fname in self.required_files:
            if fname in arrays:
                arr = arrays[fname]
                if not np.all(np.isfinite(arr)):
                    nan_count = int(np.isnan(arr).sum())
                    inf_count = int(np.isinf(arr).sum())
                    raise ContractError(
                        f"Non-finite values in {fname}: "
                        f"{nan_count} NaN, {inf_count} Inf"
                    )

        # 7. Value range warnings (non-critical)
        if "prices.npy" in arrays:
            prices = arrays["prices.npy"]
            if np.any(prices <= 0):
                warnings.append(
                    f"prices.npy contains non-positive values "
                    f"(min={prices.min():.2f})"
                )

        if "agreement_ratio.npy" in arrays:
            agreement = arrays["agreement_ratio.npy"]
            if np.any(agreement < 0) or np.any(agreement > 1.01):
                warnings.append(
                    f"agreement_ratio.npy out of expected range [0, 1]: "
                    f"min={agreement.min():.4f}, max={agreement.max():.4f}"
                )

        if "predictions.npy" in arrays:
            preds = arrays["predictions.npy"]
            unique_vals = set(np.unique(preds).tolist())
            valid_vals = {0, 1, 2}
            if not unique_vals.issubset(valid_vals):
                extra = unique_vals - valid_vals
                warnings.append(
                    f"predictions.npy contains unexpected values: {extra}. "
                    f"Expected subset of {{0, 1, 2}}"
                )

        return warnings

    def summary(self) -> str:
        """Human-readable summary of this manifest."""
        lines = [
            f"Signal Manifest: {self.signal_type} ({self.model_type})",
            f"  Split: {self.split}, Samples: {self.n_samples:,}",
            f"  Required: {', '.join(self.required_files)}",
        ]
        if self.horizons:
            lines.append(f"  Horizons: {self.horizons}")
        if self.model_metrics:
            metrics_str = ", ".join(
                f"{k}={v:.4f}" for k, v in self.model_metrics.items()
            )
            lines.append(f"  Metrics: {metrics_str}")
        return "\n".join(lines)


def _compatibility_from_dict(payload: Dict[str, Any]) -> CompatibilityContract:
    """Construct a ``CompatibilityContract`` from a parsed signal_metadata.json block.

    Handles the JSON→Python type coercion (``horizons`` is a list on-disk, must be
    a tuple for the frozen dataclass to hash cleanly; ``horizons`` may be None).
    Raises ``TypeError`` / ``KeyError`` / ``ValueError`` if the block is malformed —
    callers that want to fall back to legacy loading should catch these.
    """
    horizons = payload.get("horizons")
    if horizons is not None and not isinstance(horizons, tuple):
        horizons = tuple(horizons)
    return CompatibilityContract(
        contract_version=payload["contract_version"],
        schema_version=payload["schema_version"],
        feature_count=int(payload["feature_count"]),
        window_size=int(payload["window_size"]),
        feature_layout=payload["feature_layout"],
        data_source=payload["data_source"],
        label_strategy_hash=payload["label_strategy_hash"],
        calibration_method=payload.get("calibration_method"),
        primary_horizon_idx=payload.get("primary_horizon_idx"),
        horizons=horizons,
        normalization_strategy=payload["normalization_strategy"],
    )


__all__ = [
    "ContractError",
    "CONTENT_HASH_RE",
    "SignalManifest",
    "CLASSIFICATION_REQUIRED",
    "CLASSIFICATION_OPTIONAL",
    "REGRESSION_REQUIRED",
    "REGRESSION_OPTIONAL",
    "HYBRID_REQUIRED",
    "HYBRID_OPTIONAL",
    "ALIGNED_FILES",
]


def __getattr__(name: str):  # noqa: D401
    """Module-level lazy attribute resolver for REV 2 deprecation shims.

    REV 2 pre-push follow-up (2026-04-20): gates access to the legacy
    ``_CONTENT_HASH_RE`` name. Access via the module emits a one-time
    ``DeprecationWarning`` citing the migration path + calendar removal
    deadline (``2026-10-31``) and returns the canonical ``CONTENT_HASH_RE``
    compiled pattern — consumers observe identical matching behavior while
    receiving migration telemetry. Matches the Phase 6 6B.5 backtester-shim
    and REV 2 ``_atomic_io`` shim patterns so every deprecated name in the
    contract plane has uniform deprecation lifecycle.

    This is a MODULE-level ``__getattr__`` (PEP 562) — Python falls back to
    it only for names NOT already defined in the module's ``__dict__``. So
    this does NOT shadow ``CONTENT_HASH_RE``, ``SignalManifest``, etc.,
    which remain instant attribute lookups.
    """
    if name == "_CONTENT_HASH_RE":
        import warnings as _warnings

        if name not in _LEGACY_NAMES_WARNED:
            _LEGACY_NAMES_WARNED.add(name)
            _warnings.warn(
                f"`hft_contracts.signal_manifest._CONTENT_HASH_RE` is a "
                f"REV 2 deprecation alias. Migrate to "
                f"`from hft_contracts.signal_manifest import CONTENT_HASH_RE` "
                f"(or `from hft_contracts import CONTENT_HASH_RE`) before "
                f"the {_CONTENT_HASH_RE_REMOVAL_DATE} removal deadline. "
                f"(This warning fires once per process.)",
                DeprecationWarning,
                stacklevel=2,
            )
        return CONTENT_HASH_RE
    raise AttributeError(
        f"module 'hft_contracts.signal_manifest' has no attribute {name!r}"
    )
