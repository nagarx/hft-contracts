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


# Phase 6 6A.9 (2026-04-17): module-level regex for content_hash validation.
# Matches hft_contracts.canonical_hash.sha256_hex output format (64 lowercase
# hex chars). Contract: pipeline_contract.toml:1211 specifies this pattern.
# Module-level (not per-call) so re.compile runs once at import time.
#
# REV 2 pre-push (2026-04-20): renamed from module-private ``_CONTENT_HASH_RE``
# to public ``CONTENT_HASH_RE``. The underscore-prefix was a mis-classification
# since ``hft-ops/src/hft_ops/stages/signal_export.py`` imports it across the
# module boundary. ``_CONTENT_HASH_RE`` kept as a DEPRECATED alias for
# pre-REV-2 importers; scheduled for removal 2026-10-31.
CONTENT_HASH_RE = re.compile(r"^[a-f0-9]{64}$")
_CONTENT_HASH_RE = CONTENT_HASH_RE  # DEPRECATED: use CONTENT_HASH_RE. Removed 2026-10-31.


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

# Files that must have shape[0] == N (first dimension alignment)
ALIGNED_FILES = [
    "prices.npy",
    "predictions.npy",
    "labels.npy",
    "agreement_ratio.npy",
    "confirmation_score.npy",
    "spreads.npy",
    "predicted_returns.npy",
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

    def validate(self, signal_dir: Path) -> List[str]:
        """Validate signal directory against this manifest.

        Raises:
            ContractError: For critical issues (missing required files,
                shape mismatch, non-finite values).

        Returns:
            List of non-critical warning strings (dtype coercion, range anomalies).
        """
        signal_dir = Path(signal_dir)
        warnings: List[str] = []

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
