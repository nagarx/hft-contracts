"""Phase 8C-α Stage C.2 (2026-04-20) — FeatureImportanceArtifact contract.

Post-training feature-importance artifacts produced by the trainer's
``PermutationImportanceCallback`` (Phase 8C-α Stage C.1). Persisted to
``outputs/{exp_name}/feature_importance_v1.json`` by the trainer;
content-addressed to ``hft-ops/ledger/feature_importance/{yyyy_mm}/<sha256>.json``
by the ledger routing hook (Phase 8C-α Stage C.3); consumed by the
feature-evaluator feedback-merge step (Phase 8C-β Stage C.5).

**Contract-first discipline** (hft-rules §14): this module lands BEFORE
the trainer produces the artifact and BEFORE hft-ops routes it,
ensuring all three consumers share a single frozen schema.

**Schema evolution policy**:
  - ``schema_version`` on the artifact (NOT on the data inside).
  - Current: ``"1"``.
  - Bump MAJOR for breaking field rename/remove (requires migration
    path). Bump MINOR via additive new fields with ``None`` defaults
    to preserve legacy-artifact load via ``from_dict``. The dataclass
    is frozen — consumers never mutate in place.

**Design references**:
  - Breiman, L. (2001). Random Forests. Machine Learning 45:5–32.
  - Strobl, C. et al. (2007). Bias in random forest variable importance.
    BMC Bioinformatics 8:25 [correlation-split caveat documented].
  - Politis & Romano (1994). The Stationary Bootstrap. JASA 89:1303-1313
    [block-permutation null rationale, consumed via ``hft_metrics.block_permutation``].
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hft_contracts.atomic_io import atomic_write_json
from hft_contracts.canonical_hash import canonical_json_blob, sha256_hex


__all__ = [
    "FEATURE_IMPORTANCE_SCHEMA_VERSION",
    "FeatureImportance",
    "FeatureImportanceArtifact",
]


FEATURE_IMPORTANCE_SCHEMA_VERSION: str = "1"


# ---------------------------------------------------------------------------
# Per-feature record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureImportance:
    """Per-feature importance estimate with bootstrap confidence interval.

    Frozen — produced once by the trainer callback; consumers read it.

    Fields:
      feature_name: Semantic name (e.g., "depth_norm_ofi"). Stable across
        feature-layout changes as long as the feature is retained.
      feature_index: Position in the FeatureLayout (0-147 for the
        148-feature layout). May shift if layout changes — consumers
        that want layout-stability should match on ``feature_name``.
      importance_mean: ``observed_metric - mean(permutation_nulls)``
        averaged across ``n_seeds_aggregated`` seeds. > 0 means feature
        IMPROVES the metric (desirable); < 0 means feature HARMS.
      importance_std: Cross-seed std of importance. Low std = stable
        estimate; high std = unreliable (consider more seeds).
      ci_lower_95: 2.5th percentile of the block-permutation null
        distribution. If > 0, feature is statistically significant at
        95% level (evidence against H0 "no association").
      ci_upper_95: 97.5th percentile.
      n_permutations: Per-seed permutation replicate count. Higher =
        tighter CI, longer compute.
      n_seeds_aggregated: Number of random seeds the estimate is
        averaged over. K≥5 recommended for reliability.
      stability: Cross-seed reliability in [0, 1]. Computed as
        ``1.0 - min(1.0, importance_std / max(|importance_mean|, EPS))``
        (coefficient-of-variation variant). 1.0 = perfect agreement;
        0.0 = std dominates mean (estimate unreliable). Consumers use
        ``stability >= 0.6`` as a gating threshold.
    """

    feature_name: str
    feature_index: int
    importance_mean: float
    importance_std: float
    ci_lower_95: float
    ci_upper_95: float
    n_permutations: int
    n_seeds_aggregated: int
    stability: float


# ---------------------------------------------------------------------------
# Full artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureImportanceArtifact:
    """Complete post-training feature-importance artifact.

    Produced by the trainer's ``PermutationImportanceCallback``; persisted
    to ``outputs/{exp_name}/feature_importance_v1.json``. The hft-ops
    ledger-routing hook (Phase 8C-α Stage C.3) content-addresses the
    file (via ``content_hash``) to ``ledger/feature_importance/{yyyy_mm}/``.

    Immutable (frozen dataclass). Consumers serialize via
    ``content_hash()`` for storage addressing; the feature-evaluator
    feedback-merge step consumes the ``features`` tuple.

    Fields:
      schema_version: ``"1"`` for current. Breaking changes bump MAJOR;
        MINOR bumps add nullable-default fields only.
      method: ``"permutation"`` currently. Future: ``"shap"``,
        ``"integrated_gradients"`` would add additional artifact types;
        the schema stays compatible as long as method-specific fields
        are surfaced via ``method_kwargs``.
      baseline_metric: Metric name used for importance (e.g.,
        ``"val_ic"``, ``"val_macro_f1"``, ``"test_directional_accuracy"``).
        Must match a key in ``training_metrics`` for the same experiment.
      baseline_value: Observed metric on un-permuted eval set.
      block_size_days: Permutation block length in day units. Preserves
        intraday autocorrelation per Politis & Romano (1994). Default 1.
      n_permutations: Total permutation replicates per seed per feature.
        Default 500 per plan.
      n_seeds: Number of random seeds the artifact aggregates over.
      seed: Base seed. Per-seed seeds derived as
        ``[seed, seed+1, ..., seed+n_seeds-1]``.
      eval_split: ``"test"`` (default) or ``"val"``. Which held-out set
        was used. ``"test"`` is preferred — val was implicitly used for
        early-stopping so model is indirectly optimized on it.
      features: Tuple of ``FeatureImportance`` per feature in the
        feature set. Ordered by feature_index.
      feature_set_ref: Optional ``{name, content_hash}`` linking this
        artifact to the FeatureSet it was computed over. Required for
        downstream feedback-merge to match artifacts to evaluator
        profiles.
      experiment_id: Source experiment id.
      fingerprint: Source experiment fingerprint (ledger-unique).
      model_type: e.g. ``"tlob"``, ``"xgboost"``. Enables method-specific
        post-hoc analyses.
      timestamp_utc: ISO 8601 with timezone. When the artifact was written.
      method_caveats: Tuple of documented limitations (e.g.,
        ``("correlation-split",)`` to flag Strobl et al. 2007 bias
        on correlated features).
    """

    schema_version: str
    method: str
    baseline_metric: str
    baseline_value: float
    block_size_days: int
    n_permutations: int
    n_seeds: int
    seed: int
    eval_split: str
    features: Tuple[FeatureImportance, ...]
    feature_set_ref: Optional[Dict[str, str]]
    experiment_id: str
    fingerprint: str
    model_type: str
    timestamp_utc: str
    method_caveats: Tuple[str, ...] = field(default_factory=tuple)

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict. Tuples become lists.

        Used by ``content_hash`` (for canonical-hash input) and
        ``save`` (for on-disk persistence). Reverse via ``from_dict``.
        """
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "baseline_metric": self.baseline_metric,
            "baseline_value": self.baseline_value,
            "block_size_days": self.block_size_days,
            "n_permutations": self.n_permutations,
            "n_seeds": self.n_seeds,
            "seed": self.seed,
            "eval_split": self.eval_split,
            "features": [dataclasses.asdict(f) for f in self.features],
            "feature_set_ref": self.feature_set_ref,
            "experiment_id": self.experiment_id,
            "fingerprint": self.fingerprint,
            "model_type": self.model_type,
            "timestamp_utc": self.timestamp_utc,
            "method_caveats": list(self.method_caveats),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureImportanceArtifact":
        """Reconstruct from a dict (e.g., loaded from JSON).

        Defensive: MINOR-bump fields with ``None`` default gracefully
        absent in legacy artifacts.
        """
        features = tuple(
            FeatureImportance(**f) for f in data.get("features", [])
        )
        return cls(
            schema_version=data["schema_version"],
            method=data["method"],
            baseline_metric=data["baseline_metric"],
            baseline_value=float(data["baseline_value"]),
            block_size_days=int(data["block_size_days"]),
            n_permutations=int(data["n_permutations"]),
            n_seeds=int(data["n_seeds"]),
            seed=int(data["seed"]),
            eval_split=data["eval_split"],
            features=features,
            feature_set_ref=data.get("feature_set_ref"),
            experiment_id=data.get("experiment_id", ""),
            fingerprint=data.get("fingerprint", ""),
            model_type=data.get("model_type", ""),
            timestamp_utc=data.get("timestamp_utc", ""),
            method_caveats=tuple(data.get("method_caveats", ())),
        )

    def save(self, path: Path) -> None:
        """Atomic write via ``hft_contracts.atomic_io`` SSoT.

        Produces a file that survives SIGKILL mid-write (tmp + fsync +
        os.replace). Content is canonical (sort_keys=True); trailing
        newline included.
        """
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "FeatureImportanceArtifact":
        """Load from a JSON file.

        Raises ``json.JSONDecodeError`` if malformed, ``KeyError`` if
        required fields missing (strict — partial artifacts are broken,
        not legacy). Future MAJOR-bump migration goes in ``from_dict``.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # -----------------------------------------------------------------
    # Content-addressing
    # -----------------------------------------------------------------

    def content_hash(self) -> str:
        """SHA-256 hex of the canonical-JSON-serialized artifact.

        Uses ``hft_contracts.canonical_hash`` SSoT — ZERO re-derivation.
        Stable across Python versions / platforms since canonical JSON
        is sort-keyed + default=str.

        Used by hft-ops ledger routing (Phase 8C-α Stage C.3) to
        content-address artifacts: two artifacts with bit-identical
        content produce the same hash, enabling de-duplication across
        experiments that share (feature_set, model, seed).
        """
        blob = canonical_json_blob(self.to_dict())
        return sha256_hex(blob)

    # -----------------------------------------------------------------
    # Lookup helpers
    # -----------------------------------------------------------------

    def get_by_name(self, feature_name: str) -> Optional[FeatureImportance]:
        """Find per-feature importance by name. O(N) scan; N <= 148.

        Returns None if feature not present (e.g., FeatureSet subset).
        """
        for f in self.features:
            if f.feature_name == feature_name:
                return f
        return None


def compute_stability(mean: float, std: float, eps: float = 1e-12) -> float:
    """Stability metric used in ``FeatureImportance.stability``.

    Formula: ``1.0 - min(1.0, std / max(|mean|, EPS))``

    Interpretation: if std is much smaller than |mean|, estimate is
    reliable (stability ≈ 1). If std dominates, stability ≈ 0.

    Caveat: when |mean| → 0 (near-zero importance), stability drops
    regardless of actual std — this is desired because near-zero
    features should not drive feedback-merge decisions.

    Args:
      mean: Importance point estimate (may be negative).
      std: Cross-seed standard deviation (non-negative).
      eps: Division guard. Consumers should use the same EPS as
        ``hft_metrics._sanitize`` (1e-12).

    Returns:
      Float in [0.0, 1.0].
    """
    if not math.isfinite(mean) or not math.isfinite(std):
        return 0.0
    if std < 0:
        return 0.0
    denom = max(abs(mean), eps)
    return float(max(0.0, min(1.0, 1.0 - std / denom)))
