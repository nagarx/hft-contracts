"""Phase 2 P2.A (2026-05-07) — TestMetricsCIArtifact contract.

Bootstrap-CI artifact produced by ``lobtrainer.analysis.stat_rigor.ci``
(Phase 2 STAT RIGOR FLOOR P2.A). Persisted to
``outputs/{exp_name}/test_metrics_ci_v1.json`` by the trainer-side
analysis library; content-addressed to
``hft-ops/ledger/test_metrics_ci/{yyyy_mm}/<sha256>.json`` by the
ledger routing hook (matches Phase 8C-α Stage C.3 precedent).

**Contract-first discipline** (hft-rules §14): this module lands BEFORE
the producer + consumer wire-up so all three layers share a single frozen
schema.

**Schema evolution policy**:
  - ``schema_version`` on the artifact (NOT on the data inside).
  - Current: ``"1"``.
  - Bump MAJOR for breaking field rename/remove (requires migration
    path). Bump MINOR via additive new fields with ``None`` defaults
    to preserve legacy-artifact load via ``from_dict``. The dataclass
    is frozen — consumers never mutate in place.

**Design references**:
  - Politis & Romano (1994). The Stationary Bootstrap. JASA 89:1303-1313
    [block-length auto-derive ``ceil(n^(1/3))``; consumed via
    ``hft_metrics.block_bootstrap_ci``].
  - Künsch, H. R. (1989). The jackknife and the bootstrap for general
    stationary observations. Annals of Statistics 17:1217-1241
    [moving-block bootstrap rationale: HFT regression metrics are NOT
    i.i.d. — block resampling preserves autocorrelation].

**Mirror precedent**: ``FeatureImportanceArtifact`` (Phase 8C-α Stage C.2)
— same frozen-dataclass pattern + ``content_hash()`` via SSoT +
``save()`` via atomic_io SSoT + ``from_dict`` migration shim.

**Phase Y composability**: artifact integrates with future
``experiment_provenance_hash`` via ``record.artifacts[].sha256`` projection
(routed through hft-ops ``_POST_STAGE_ARTIFACT_PATTERNS``).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

from hft_contracts.atomic_io import atomic_write_json
from hft_contracts.canonical_hash import canonical_json_blob, sha256_hex


__all__ = [
    "TEST_METRICS_CI_SCHEMA_VERSION",
    "MetricCIBound",
    "TestMetricsCIArtifact",
]


TEST_METRICS_CI_SCHEMA_VERSION: str = "1"
# v1 (2026-05-07, Cyclelet B P2.A initial ship): introduces the contract
# with point/ci_low/ci_high per metric + compat_fp + model_config_hash +
# normalization_stats_sha256 + signal_export_output_dir traceability fields.
# Future MINOR bumps: additive new fields with None defaults. MAJOR bumps
# require migration path in from_dict.


# ---------------------------------------------------------------------------
# Per-metric record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricCIBound:
    """Bootstrap confidence interval for a single test metric.

    Frozen — produced once by the analysis library; consumers read it.

    Fields:
      point: Observed point estimate on the un-resampled eval set, as
        returned by ``hft_metrics.block_bootstrap_ci`` (the ``estimate``
        return value). Caller is responsible for ensuring this matches
        any externally-stored single-point estimate (e.g.,
        ``signal_metadata.json::metrics[metric_name]``); cross-check is
        NOT performed at this layer.
      ci_low: Lower CI bound at the configured confidence level
        (e.g., 2.5th percentile of bootstrap distribution at ci=0.95).
      ci_high: Upper CI bound (e.g., 97.5th percentile at ci=0.95).
      n_samples: Number of paired (label, prediction) samples the CI
        was computed on. For HFT v3p0 corpora this is typically 8085
        (test split of e5_timebased_60s_v3p0).

    Invariants (validated at ``__post_init__``):
      - All 3 floats finite (no NaN/Inf)
      - ``ci_low <= point <= ci_high``
      - ``n_samples > 0``
    """

    point: float
    ci_low: float
    ci_high: float
    n_samples: int

    def __post_init__(self) -> None:
        """Validate at construction time per hft-rules §5/§8.

        Class-A SSoT primitive validation (Round 1 mid-impl adversarial
        finding §2 HIGH): leaf type validates its own invariants so the
        parent ``TestMetricsCIArtifact`` doesn't have to interpret error
        messages from cross-leaf invariant checks. Failures here surface
        directly (e.g., "MetricCIBound: point=NaN" rather than the
        misleading parent error "violates ci_low <= point <= ci_high
        invariant: (0.0, nan, 1.0)").
        """
        import math
        for field_name, value in (
            ("point", self.point),
            ("ci_low", self.ci_low),
            ("ci_high", self.ci_high),
        ):
            if not math.isfinite(value):
                raise ValueError(
                    f"MetricCIBound: {field_name}={value!r} is not finite "
                    f"(NaN/Inf) — bootstrap CI computation likely degenerate; "
                    f"investigate caller (statistic_fn output)"
                )
        if not (self.ci_low <= self.point <= self.ci_high):
            raise ValueError(
                f"MetricCIBound: invariant ci_low <= point <= ci_high "
                f"violated ({self.ci_low}, {self.point}, {self.ci_high}). "
                f"Likely cause: degenerate bootstrap resamples (constant "
                f"blocks producing undefined statistic). Re-run with "
                f"different seed or investigate input array."
            )
        if self.n_samples <= 0:
            raise ValueError(
                f"MetricCIBound: n_samples={self.n_samples} must be > 0"
            )


# ---------------------------------------------------------------------------
# Full artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestMetricsCIArtifact:
    """Complete bootstrap-CI artifact for one trained experiment's test metrics.

    Produced by ``lobtrainer.analysis.stat_rigor.ci``; persisted to
    ``outputs/{exp_name}/test_metrics_ci_v1.json``. The hft-ops
    ledger-routing hook content-addresses the file (via ``content_hash``)
    to ``ledger/test_metrics_ci/{yyyy_mm}/``.

    Immutable (frozen dataclass). Consumers serialize via
    ``content_hash()`` for storage addressing; downstream Phase 2 P2.C
    pairwise-comparison consumers read the ``metrics`` dict.

    Fields:
      schema_version: ``"1"`` for current. Breaking changes bump MAJOR;
        MINOR bumps add nullable-default fields only.
      method: ``"block_bootstrap"`` currently. Future variants:
        ``"stationary_bootstrap"``, ``"jackknife"`` would extend the
        artifact taxonomy via this discriminator.
      block_length: Block length in SAMPLE units. Auto-derived
        ``ceil(n^(1/3))`` per Politis-Romano (1994) when caller passes
        ``block_length=None`` to ``block_bootstrap_ci`` — so for
        N=8085, block_length=21 (n_blocks=385, full coverage).
      block_length_source: Human-readable rationale string
        (e.g., ``"auto-derive ceil(n^(1/3)) per Politis-Romano 1994"``
        or ``"explicit override via --block-length=50"``).
      n_bootstraps: Number of bootstrap replicates. Plan v4 §4.1
        recommends 10000 (Plan v4 §3 inventory cited 10K; plan-v4-v3
        rejected 1000 default as too few for stable bounds).
      ci: Confidence level in (0.0, 1.0). Default 0.95 → 2.5th/97.5th
        percentile bounds. NOT alpha (semantically equivalent but
        wrong-API param name per Plan v4 v1→v2 correction).
      seed: Base seed for bootstrap RNG. ``np.random.RandomState(seed)``
        per ``hft_metrics.block_bootstrap_ci`` impl.
      n_test_samples: Number of paired (label, prediction) test samples.
      metrics: Dict mapping metric name to ``MetricCIBound``. Standard
        keys per Plan v4 §4.1: ``test_ic``, ``test_directional_accuracy``,
        ``test_r2``, ``test_pearson``, ``test_mae``, ``test_rmse``,
        ``test_profitable_accuracy``. Stored sorted by key in
        ``to_dict`` for canonical-hash stability.
      compatibility_fingerprint: Optional 64-hex SHA-256 from
        ``signal_metadata.json::compatibility_fingerprint`` (Phase II
        contract). Enables ledger query
        ``ledger list --compatibility-fp <hex>`` to find paired CI
        artifacts. None for legacy / pre-Phase-Q.6.5 sklearn experiments.
      model_config_hash: Optional 64-hex SHA-256 from Phase X.1.C
        ``compute_model_config_hash`` (architectural-axis hash with
        ``_LOSS_TUNING_KEYS`` denylist filtering). None for pre-X.1
        artifacts.
      normalization_stats_sha256: Optional 64-hex SHA-256 from Phase 1
        N7 normalization-stats checkpoint binding. None for sklearn
        (Phase 1 #PY-53 deferred binding).
      signal_export_output_dir: Optional absolute path string from
        Phase V.1 L1.2 manifest-move resilience field. Used for
        consumer-side ``np.load(predicted_returns.npy)`` re-resolution
        when source signals are inspected.
      experiment_id: Source experiment id (matches ledger record).
      fingerprint: Source experiment fingerprint (ledger-unique,
        treatment-axis hash for cross-experiment grouping).
      model_type: e.g. ``"tlob"``, ``"hmhp_regression"``,
        ``"temporal_ridge"``. Enables filter
        ``hft-ops ledger list --model-type tlob --has-artifact-kind=test_metrics_ci``.
      timestamp_utc: ISO 8601 with timezone (e.g.,
        ``"2026-05-07T20:30:00Z"``). When the artifact was written.
      method_caveats: Tuple of documented limitations (e.g.,
        ``("ic_silent_sanitize",)`` to flag that spearman_ic was NOT
        migrated to fail-loud in #PY-63 so its bootstrap may absorb
        edge-case constant-block resamples silently).
    """

    # Pytest-discovery suppression: class name starts with "Test" but it
    # is a domain dataclass, not a pytest test class. ``__test__ = False``
    # tells pytest collection to skip it without renaming the semantically-
    # correct identifier ("Test" as in "test set" — held-out evaluation —
    # not "unit test"). ClassVar prevents dataclass field-treatment.
    __test__: ClassVar[bool] = False

    schema_version: str
    method: str
    block_length: int
    block_length_source: str
    n_bootstraps: int
    ci: float
    seed: int
    n_test_samples: int
    metrics: Dict[str, MetricCIBound]
    compatibility_fingerprint: Optional[str]
    model_config_hash: Optional[str]
    normalization_stats_sha256: Optional[str]
    signal_export_output_dir: Optional[str]
    experiment_id: str
    fingerprint: str
    model_type: str
    timestamp_utc: str
    method_caveats: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate at construction time (fail-loud per hft-rules §5/§8).

        Closes Agent 3 (hft-architect) HIGH finding #2: every artifact
        must validate its invariants at construction. Mirrors the
        ``CompatibilityContract.__post_init__`` Phase II hardening pattern
        documented at ``hft_contracts.compatibility``.
        """
        if self.n_test_samples <= 0:
            raise ValueError(
                f"TestMetricsCIArtifact: n_test_samples={self.n_test_samples} "
                f"must be > 0 (constructor invariant; bootstrap on "
                f"empty/inverted-size samples is undefined)"
            )
        if self.n_bootstraps < 100:
            raise ValueError(
                f"TestMetricsCIArtifact: n_bootstraps={self.n_bootstraps} "
                f"< 100 — too few for stable CI bounds (Plan v4 §4.1 "
                f"recommends 10000; below 100 is degenerate per "
                f"Politis-Romano sample-size analysis)"
            )
        if self.block_length < 2:
            raise ValueError(
                f"TestMetricsCIArtifact: block_length={self.block_length} "
                f"< 2 — degenerate (block resampling requires "
                f"block_length >= 2 to preserve autocorrelation; "
                f"block_length=1 is element-wise iid bootstrap which "
                f"is the WRONG primitive for HFT signal analysis)"
            )
        if not 0.0 < self.ci < 1.0:
            raise ValueError(
                f"TestMetricsCIArtifact: ci={self.ci} not in (0.0, 1.0). "
                f"Common values are 0.90, 0.95, 0.99 (NOT alpha=0.05 — "
                f"that's wrong-API param naming per Plan v4 v1→v2)"
            )
        if not self.metrics:
            raise ValueError(
                "TestMetricsCIArtifact: metrics dict must not be empty"
            )
        for metric_name, bound in self.metrics.items():
            if not isinstance(metric_name, str):
                raise TypeError(
                    f"TestMetricsCIArtifact: metric key must be str, "
                    f"got {type(metric_name).__name__}={metric_name!r}"
                )
            if not isinstance(bound, MetricCIBound):
                raise TypeError(
                    f"TestMetricsCIArtifact: metrics[{metric_name!r}] "
                    f"must be MetricCIBound instance, got "
                    f"{type(bound).__name__}"
                )
            # Sanity invariant: point estimate must lie within CI bounds.
            # If violated, EITHER the bootstrap was buggy OR the caller
            # passed mismatched metrics — fail-loud at construction.
            if not (bound.ci_low <= bound.point <= bound.ci_high):
                raise ValueError(
                    f"TestMetricsCIArtifact: metrics[{metric_name!r}] "
                    f"violates ci_low <= point <= ci_high invariant: "
                    f"({bound.ci_low}, {bound.point}, {bound.ci_high})"
                )
            if bound.n_samples != self.n_test_samples:
                raise ValueError(
                    f"TestMetricsCIArtifact: metrics[{metric_name!r}].n_samples="
                    f"{bound.n_samples} != self.n_test_samples="
                    f"{self.n_test_samples} (cross-metric n must agree — "
                    f"all metrics computed on same test split)"
                )
        # SHA-256 hex format check (mirror CompatibilityContract pattern):
        # if a fingerprint is supplied, it MUST be 64-char lowercase hex.
        for fp_field, fp_value in (
            ("compatibility_fingerprint", self.compatibility_fingerprint),
            ("model_config_hash", self.model_config_hash),
            ("normalization_stats_sha256", self.normalization_stats_sha256),
        ):
            if fp_value is not None:
                if not isinstance(fp_value, str) or len(fp_value) != 64:
                    raise ValueError(
                        f"TestMetricsCIArtifact: {fp_field!r}={fp_value!r} "
                        f"must be 64-char SHA-256 hex string or None; "
                        f"got len={len(fp_value) if isinstance(fp_value, str) else 'N/A'}"
                    )
                # Validate hex format: every char must be 0-9 or a-f.
                # Mirror hft_contracts.signal_manifest.CONTENT_HASH_RE pattern
                # (lower-case hex). Reject upper-case to enforce canonical form.
                if not all(c in "0123456789abcdef" for c in fp_value):
                    raise ValueError(
                        f"TestMetricsCIArtifact: {fp_field!r} must be "
                        f"lower-case hex; got {fp_value!r}"
                    )

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict (sorted-keys metrics for
        canonical-hash stability).

        Used by ``content_hash`` (canonical-JSON input) and ``save``
        (on-disk persistence). Reverse via ``from_dict``.
        """
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "block_length": self.block_length,
            "block_length_source": self.block_length_source,
            "n_bootstraps": self.n_bootstraps,
            "ci": self.ci,
            "seed": self.seed,
            "n_test_samples": self.n_test_samples,
            "metrics": {
                name: dataclasses.asdict(bound)
                for name, bound in sorted(self.metrics.items())
            },
            "compatibility_fingerprint": self.compatibility_fingerprint,
            "model_config_hash": self.model_config_hash,
            "normalization_stats_sha256": self.normalization_stats_sha256,
            "signal_export_output_dir": self.signal_export_output_dir,
            "experiment_id": self.experiment_id,
            "fingerprint": self.fingerprint,
            "model_type": self.model_type,
            "timestamp_utc": self.timestamp_utc,
            "method_caveats": list(self.method_caveats),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestMetricsCIArtifact":
        """Reconstruct from a dict (e.g., loaded from JSON).

        Defensive:
          - MINOR-bump fields with ``None`` default gracefully absent
            in legacy artifacts.
          - ``MetricCIBound`` entries filter unknown kwargs so a future
            v2 per-metric field addition doesn't crash a v1 consumer
            reading a newer artifact.
          - Strict on required scalar fields (``schema_version``,
            ``method``, etc.) — partial artifacts are broken, not legacy.
        """
        bound_fields = {f.name for f in dataclasses.fields(MetricCIBound)}
        metrics: Dict[str, MetricCIBound] = {}
        for name, bd in data.get("metrics", {}).items():
            metrics[name] = MetricCIBound(
                **{k: v for k, v in bd.items() if k in bound_fields}
            )
        return cls(
            schema_version=data["schema_version"],
            method=data["method"],
            block_length=int(data["block_length"]),
            block_length_source=data["block_length_source"],
            n_bootstraps=int(data["n_bootstraps"]),
            ci=float(data["ci"]),
            seed=int(data["seed"]),
            n_test_samples=int(data["n_test_samples"]),
            metrics=metrics,
            compatibility_fingerprint=data.get("compatibility_fingerprint"),
            model_config_hash=data.get("model_config_hash"),
            normalization_stats_sha256=data.get("normalization_stats_sha256"),
            signal_export_output_dir=data.get("signal_export_output_dir"),
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
        newline included. Mirror of FeatureImportanceArtifact.save.
        """
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "TestMetricsCIArtifact":
        """Load from a JSON file.

        Raises ``json.JSONDecodeError`` if malformed, ``KeyError`` if
        required fields missing (strict — partial artifacts are broken,
        not legacy). Future MAJOR-bump migration goes in ``from_dict``.

        Used by hft-ops ``_POST_STAGE_ARTIFACT_PATTERNS`` validator
        (matches FeatureImportanceArtifact.load registration pattern).
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

        Used by hft-ops ledger routing (matches Phase 8C-α Stage C.3
        content-addressing convention) to address artifacts: two
        artifacts with bit-identical content produce the same hash,
        enabling de-duplication across re-runs of the same experiment.
        """
        blob = canonical_json_blob(self.to_dict())
        return sha256_hex(blob)

    # -----------------------------------------------------------------
    # Lookup helpers
    # -----------------------------------------------------------------

    def get_metric(self, metric_name: str) -> Optional[MetricCIBound]:
        """Find per-metric bound by name. O(1).

        Returns None if metric not present. Standard names per
        Plan v4 §4.1: ``test_ic``, ``test_directional_accuracy``,
        ``test_r2``, ``test_pearson``, ``test_mae``, ``test_rmse``,
        ``test_profitable_accuracy``.
        """
        return self.metrics.get(metric_name)
