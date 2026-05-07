"""Phase 2 P2.C (2026-05-07) — PairwiseCompareArtifact contract.

K-way pairwise comparison artifact produced by
``lobtrainer.analysis.stat_rigor.pairwise`` (Phase 2 STAT RIGOR FLOOR
P2.C). Persisted to
``outputs/comparisons/<sorted_exp_ids>/pairwise_compare_v1.json`` by the
trainer-side analysis library; content-addressed to
``hft-ops/ledger/pairwise_compare/{yyyy_mm}/<sha256>.json`` by the
ledger routing hook (matches Phase 8C-α Stage C.3 + Phase 2 P2.A
TestMetricsCIArtifact precedents).

**Contract-first discipline** (hft-rules §14): this module lands
BEFORE the producer + consumer wire-up so all three layers share a
single frozen schema.

**Schema evolution policy**:
  - ``schema_version`` on the artifact (NOT on the data inside).
  - Current: ``"1"``.
  - Bump MAJOR for breaking field rename/remove (requires migration
    path). Bump MINOR via additive new fields with ``None`` defaults
    to preserve legacy-artifact load via ``from_dict``. The dataclass
    is frozen — consumers never mutate in place.

**Design references**:
  - Künsch, H. R. (1989). The jackknife and the bootstrap for general
    stationary observations. Annals of Statistics 17:1217-1241
    [moving-block bootstrap rationale; consumed via
    ``hft_metrics.pairwise.pairwise_paired_bootstrap_compare``].
  - Politis & Romano (1994). The Stationary Bootstrap. JASA 89:1303-1313
    [block_length auto-derive ``ceil(n^(1/3))``].
  - Efron, B. & Tibshirani, R. (1993). An Introduction to the
    Bootstrap. Ch 15, eq 15.22 [paired-bootstrap two-sided p-value].
  - Benjamini, Y. & Hochberg, Y. (1995). Controlling the false
    discovery rate. JRSS B 57:289-300 [BH FDR correction at K-pair level].

**Mirror precedent**: ``TestMetricsCIArtifact`` (Phase 2 P2.A, this
release) — same frozen-dataclass pattern + ``content_hash()`` via SSoT
+ ``save()`` via atomic_io SSoT + ``from_dict`` migration shim.

**Round 2 architectural critique applied** (2026-05-07):
  - K-arbitrary support (NOT just K=2 — primitive supports it; BH FDR
    is meaningful only at K>=3)
  - Effect size required (statistic_a, statistic_b, delta, delta_ci_low/high
    per pair — Agent B HIGH finding)
  - Strict ``paired_compat_fingerprint`` invariant (all K must share)
  - Phase Y composability via ``parent_experiment_ids`` tuple
  - Symmetric storage path ``outputs/comparisons/<sorted_exp_ids>/``

**Phase Y composability**: artifact integrates with future
``experiment_provenance_hash`` graph as a "comparison node" with K
parent ``experiment_provenance_hash`` references via the parallel-
indexed ``parent_*`` tuples.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

from hft_contracts.atomic_io import atomic_write_json
from hft_contracts.canonical_hash import canonical_json_blob, sha256_hex


__all__ = [
    "PAIRWISE_COMPARE_SCHEMA_VERSION",
    "PairwiseResultRecord",
    "PairwiseCompareArtifact",
]


PAIRWISE_COMPARE_SCHEMA_VERSION: str = "1"
# v1 (2026-05-07, Cyclelet B P2.C initial ship): introduces the contract
# with K-arbitrary parallel-indexed traceability + per-pair effect size
# (statistic_a, statistic_b, delta, delta_ci_low/high) + BH q-values +
# n_nonfinite_replaced observability per pair.
# Future MINOR bumps: additive new fields with None defaults. MAJOR bumps
# require migration path in from_dict.


# ---------------------------------------------------------------------------
# Per-pair record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairwiseResultRecord:
    """Per-pair pairwise-comparison result.

    Frozen — produced once by the analysis library; consumers read it.
    Mirrors ``hft_metrics.pairwise.PairwiseResult`` with two additions:
      (a) Human-readable ``treatment_a_label`` / ``treatment_b_label``
          (e.g., "R9_TLOB_no_CVML" / "R10_TLOB_CVML") — required for
          downstream artifact-readers + EXPERIMENT_INDEX entries.
      (b) Construction-time validation per Round 1 P2.A precedent.

    Sign convention (mirrors hft-metrics): ``delta = statistic_a -
    statistic_b``. Positive ``delta`` means treatment A is better on
    a higher-is-better metric (IC, accuracy). Caller responsible for
    statistic_fn sign convention.

    Fields:
      treatment_a_idx: Index of treatment A in original K-treatment
        list (0-based; matches ``hft_metrics.pairwise.PairwiseResult.i``).
      treatment_b_idx: Index of treatment B (always > treatment_a_idx).
      treatment_a_label: Human-readable label (e.g., "R9_TLOB_no_CVML").
      treatment_b_label: Human-readable label.
      statistic_a: Observed statistic on treatment A's predictions
        (un-resampled). Should match the single-point estimate from
        the experiment's ``test_metrics_ci_v1.json::metrics[name].point``
        — caller-side cross-check recommended per hft-rules §11.
      statistic_b: Observed statistic on treatment B's predictions.
      delta: ``statistic_a - statistic_b``.
      delta_ci_low: Lower bound of (1 - alpha) bootstrap percentile CI on delta.
      delta_ci_high: Upper bound of (1 - alpha) bootstrap percentile CI on delta.
      p_value_raw: Two-sided bootstrap p-value (Efron-Tibshirani 1993
        Ch 15 eq 15.22): ``2 * min(P(delta_b <= 0), P(delta_b >= 0))``.
      p_value_bh: Benjamini-Hochberg FDR-adjusted q-value, computed
        across ALL K*(K-1)/2 pairs simultaneously. Use this (not raw)
        for FDR-controlled significance decisions at K>=3.
      n_nonfinite_replaced: Count of bootstrap-iter × treatment samples
        where ``statistic_fn`` produced non-finite + conservative
        fallback applied (hft-metrics v0.1.7 observability field).
        Upper bound: ``2 * n_bootstraps``. High value (≈ n_bootstraps
        or more) means CI is suspect — DO NOT publish that pair's
        p-value.

    Invariants (validated at ``__post_init__``):
      - All numeric fields finite (no NaN/Inf)
      - ``treatment_a_idx < treatment_b_idx``
      - ``delta_ci_low <= delta <= delta_ci_high``
      - ``0 <= p_value_raw <= 1`` and ``0 <= p_value_bh <= 1``
      - ``n_nonfinite_replaced >= 0``
    """

    treatment_a_idx: int
    treatment_b_idx: int
    treatment_a_label: str
    treatment_b_label: str
    statistic_a: float
    statistic_b: float
    delta: float
    delta_ci_low: float
    delta_ci_high: float
    p_value_raw: float
    p_value_bh: float
    n_nonfinite_replaced: int

    def __post_init__(self) -> None:
        """Validate at construction time per hft-rules §5/§8.

        Round 1 P2.A precedent (MetricCIBound): leaf-type validation
        surfaces clearer errors than parent-artifact's cross-leaf check.
        """
        # Finite-floats check
        for fname, fval in (
            ("statistic_a", self.statistic_a),
            ("statistic_b", self.statistic_b),
            ("delta", self.delta),
            ("delta_ci_low", self.delta_ci_low),
            ("delta_ci_high", self.delta_ci_high),
            ("p_value_raw", self.p_value_raw),
            ("p_value_bh", self.p_value_bh),
        ):
            if not math.isfinite(fval):
                raise ValueError(
                    f"PairwiseResultRecord: {fname}={fval!r} is not finite — "
                    f"bootstrap computation likely degenerate; investigate "
                    f"caller (statistic_fn output or n_nonfinite_replaced)"
                )
        # Index ordering invariant (matches hft-metrics PairwiseResult convention)
        if self.treatment_a_idx >= self.treatment_b_idx:
            raise ValueError(
                f"PairwiseResultRecord: treatment_a_idx="
                f"{self.treatment_a_idx} must be < treatment_b_idx="
                f"{self.treatment_b_idx} (hft-metrics PairwiseResult "
                f"convention: i < j always)"
            )
        if self.treatment_a_idx < 0:
            raise ValueError(
                f"PairwiseResultRecord: treatment_a_idx="
                f"{self.treatment_a_idx} must be >= 0"
            )
        # Delta CI invariant
        if not (self.delta_ci_low <= self.delta <= self.delta_ci_high):
            raise ValueError(
                f"PairwiseResultRecord: delta_ci_low <= delta <= "
                f"delta_ci_high invariant violated "
                f"({self.delta_ci_low}, {self.delta}, {self.delta_ci_high}). "
                f"Likely degenerate bootstrap resamples."
            )
        # p-value range
        for pname, pval in (
            ("p_value_raw", self.p_value_raw),
            ("p_value_bh", self.p_value_bh),
        ):
            if not (0.0 <= pval <= 1.0):
                raise ValueError(
                    f"PairwiseResultRecord: {pname}={pval} not in [0.0, 1.0]"
                )
        # n_nonfinite_replaced non-negative
        if self.n_nonfinite_replaced < 0:
            raise ValueError(
                f"PairwiseResultRecord: n_nonfinite_replaced="
                f"{self.n_nonfinite_replaced} must be >= 0"
            )
        # Sanity on labels
        if not self.treatment_a_label or not self.treatment_b_label:
            raise ValueError(
                "PairwiseResultRecord: treatment_a_label and "
                "treatment_b_label must be non-empty strings"
            )

    @classmethod
    def from_hft_metrics_result(
        cls,
        result: Any,  # hft_metrics.pairwise.PairwiseResult
        treatment_labels: Tuple[str, ...],
    ) -> "PairwiseResultRecord":
        """Construct from hft_metrics.pairwise.PairwiseResult + label list.

        Bridges the runtime primitive's PairwiseResult dataclass to the
        artifact-side record. Adds human-readable labels by indexing
        treatment_labels[i] / treatment_labels[j].
        """
        return cls(
            treatment_a_idx=result.i,
            treatment_b_idx=result.j,
            treatment_a_label=treatment_labels[result.i],
            treatment_b_label=treatment_labels[result.j],
            statistic_a=float(result.statistic_i),
            statistic_b=float(result.statistic_j),
            delta=float(result.delta),
            delta_ci_low=float(result.ci_lower),
            delta_ci_high=float(result.ci_upper),
            p_value_raw=float(result.p_value_raw),
            p_value_bh=float(result.p_value_bh),
            n_nonfinite_replaced=int(getattr(result, "n_nonfinite_replaced", 0)),
        )


# ---------------------------------------------------------------------------
# Full artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairwiseCompareArtifact:
    """K-way pairwise-comparison artifact for trained-experiment statistics.

    Phase 2 P2.C artifact. Produced by
    ``lobtrainer.analysis.stat_rigor.pairwise.compare_k_way`` (consumer of
    ``hft_metrics.pairwise.pairwise_paired_bootstrap_compare`` SSoT
    primitive). Persisted to
    ``outputs/comparisons/<sorted_exp_ids>/pairwise_compare_v1.json``.
    Content-addressed via ``content_hash()`` to
    ``hft-ops/ledger/pairwise_compare/{yyyy_mm}/<sha>.json`` by hft-ops
    ledger routing.

    Frozen — immutable once produced.

    Fields:
      schema_version: ``"1"`` for current.
      method: ``"paired_block_bootstrap"`` currently.
      metric_name: e.g., ``"spearman_ic"``, ``"pearson_r"``,
        ``"directional_accuracy"``. Selects which statistic was compared.
      block_length: Block length used (auto-derived ``ceil(n^(1/3))``
        per Politis-Romano 1994 OR explicit override).
      block_length_source: Human-readable rationale string.
      n_bootstraps: Bootstrap iterations (default 10000 per Plan v4 §4.3).
      alpha: Significance level (default 0.05 → 95% CI). Note: hft-metrics
        primitive uses ``alpha`` (NOT ``ci=0.95`` like P2.A bootstrap_ci);
        this artifact preserves the primitive's parameter naming.
      seed: Bootstrap RNG seed.
      n_treatments: K — number of treatments compared. Must be >= 2;
        K=3+ enables meaningful BH FDR correction (K=2 BH ≡ raw p-value).
      n_samples_paired: Number of paired samples after NaN-row drop.
      n_samples_raw: Number of paired samples BEFORE NaN-row drop.
      n_dropped_nonfinite: Count of rows dropped (any column non-finite).
      drop_fraction: ``n_dropped_nonfinite / n_samples_raw``. Threshold
        check: if above ``max_drop_frac`` config (default 0.05), the
        producer should raise — present here for observability.
      primary_horizon_idx: Slice index for multi-horizon arrays (HMHP-R).
      parent_experiment_ids: K experiment_ids parallel-indexed with
        treatments; (Phase Y composability — future
        ``experiment_provenance_hash`` graph node points to these K).
      parent_compatibility_fingerprints: K compat_fps; ALL must be
        identical (validated in __post_init__) — pairwise comparison
        requires shared paired data.
      parent_model_config_hashes: K model_config_hashes (or None for
        sklearn pre-Phase-Q.6.5 experiments).
      paired_compat_fingerprint: The shared compat_fp (single string
        copy of parent_compatibility_fingerprints[0] after invariant
        check passes).
      paired_labels_sha256: SHA-256 of regression_labels.npy bytes
        (after horizon-slicing). Verifies that all K experiments
        consume the same labels.
      pairs: Tuple of ``PairwiseResultRecord`` of length ``K*(K-1)/2``.
        Ordered by lexicographic (i, j) pair indices (i < j).
      treatment_labels: K human-readable labels parallel-indexed with
        treatments (e.g., ``("R9_TLOB_no_CVML", "R10_TLOB_CVML",
        "R11_TLOB_GMADL_CVML")``).
      timestamp_utc: ISO 8601 with timezone (e.g.,
        ``"2026-05-07T20:30:00Z"``).
      method_caveats: Tuple of documented limitations.
    """

    # Pytest-discovery suppression: NOT a pytest test class.
    __test__: ClassVar[bool] = False

    schema_version: str
    method: str
    metric_name: str
    block_length: int
    block_length_source: str
    n_bootstraps: int
    alpha: float
    seed: int
    n_treatments: int
    n_samples_paired: int
    n_samples_raw: int
    n_dropped_nonfinite: int
    drop_fraction: float
    primary_horizon_idx: int
    parent_experiment_ids: Tuple[str, ...]
    parent_compatibility_fingerprints: Tuple[str, ...]
    parent_model_config_hashes: Tuple[Optional[str], ...]
    paired_compat_fingerprint: str
    paired_labels_sha256: str
    pairs: Tuple[PairwiseResultRecord, ...]
    treatment_labels: Tuple[str, ...]
    timestamp_utc: str
    method_caveats: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate at construction time (fail-loud per hft-rules §5/§8).

        Closes Agent B Round 1 architectural critique HIGH findings:
        K-pairs invariant + parallel-tuple length parity + strict
        compat_fp invariant + 64-hex SHA validation.
        """
        # n_treatments minimum
        if self.n_treatments < 2:
            raise ValueError(
                f"PairwiseCompareArtifact: n_treatments={self.n_treatments} "
                f"< 2 — pairwise comparison requires at least 2 treatments"
            )
        # n_pairs == K*(K-1)/2
        expected_n_pairs = self.n_treatments * (self.n_treatments - 1) // 2
        if len(self.pairs) != expected_n_pairs:
            raise ValueError(
                f"PairwiseCompareArtifact: len(pairs)={len(self.pairs)} != "
                f"K*(K-1)/2={expected_n_pairs} for K={self.n_treatments}"
            )
        # Parallel-tuple length parity
        for tname, tvalue in (
            ("parent_experiment_ids", self.parent_experiment_ids),
            ("parent_compatibility_fingerprints", self.parent_compatibility_fingerprints),
            ("parent_model_config_hashes", self.parent_model_config_hashes),
            ("treatment_labels", self.treatment_labels),
        ):
            if len(tvalue) != self.n_treatments:
                raise ValueError(
                    f"PairwiseCompareArtifact: len({tname})={len(tvalue)} "
                    f"!= n_treatments={self.n_treatments}"
                )
        # alpha in (0, 1)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(
                f"PairwiseCompareArtifact: alpha={self.alpha} not in (0, 1)"
            )
        # n_bootstraps >= 100
        if self.n_bootstraps < 100:
            raise ValueError(
                f"PairwiseCompareArtifact: n_bootstraps={self.n_bootstraps} "
                f"< 100 — too few for stable CI"
            )
        # block_length >= 2
        if self.block_length < 2:
            raise ValueError(
                f"PairwiseCompareArtifact: block_length={self.block_length} "
                f"< 2 — degenerate"
            )
        # n_samples_paired > 0
        if self.n_samples_paired <= 0:
            raise ValueError(
                f"PairwiseCompareArtifact: n_samples_paired="
                f"{self.n_samples_paired} must be > 0"
            )
        # n_dropped_nonfinite + n_samples_paired == n_samples_raw
        if self.n_dropped_nonfinite + self.n_samples_paired != self.n_samples_raw:
            raise ValueError(
                f"PairwiseCompareArtifact: "
                f"n_dropped_nonfinite ({self.n_dropped_nonfinite}) + "
                f"n_samples_paired ({self.n_samples_paired}) != "
                f"n_samples_raw ({self.n_samples_raw})"
            )
        # drop_fraction consistency (allow small float-eq tolerance)
        if self.n_samples_raw > 0:
            expected_drop_frac = self.n_dropped_nonfinite / self.n_samples_raw
            if abs(self.drop_fraction - expected_drop_frac) > 1e-9:
                raise ValueError(
                    f"PairwiseCompareArtifact: drop_fraction="
                    f"{self.drop_fraction} != "
                    f"n_dropped_nonfinite/n_samples_raw="
                    f"{expected_drop_frac:.10f}"
                )
        # primary_horizon_idx >= 0
        if self.primary_horizon_idx < 0:
            raise ValueError(
                f"PairwiseCompareArtifact: primary_horizon_idx="
                f"{self.primary_horizon_idx} < 0"
            )
        # 64-hex SHA validation: paired_compat_fingerprint + paired_labels_sha256
        for fp_field, fp_value in (
            ("paired_compat_fingerprint", self.paired_compat_fingerprint),
            ("paired_labels_sha256", self.paired_labels_sha256),
        ):
            if not (
                isinstance(fp_value, str)
                and len(fp_value) == 64
                and all(c in "0123456789abcdef" for c in fp_value)
            ):
                raise ValueError(
                    f"PairwiseCompareArtifact: {fp_field!r}={fp_value!r} "
                    f"must be 64-char lower-case hex"
                )
        # parent_compatibility_fingerprints: each must be 64-hex AND all equal
        for i, h in enumerate(self.parent_compatibility_fingerprints):
            if not (
                isinstance(h, str)
                and len(h) == 64
                and all(c in "0123456789abcdef" for c in h)
            ):
                raise ValueError(
                    f"PairwiseCompareArtifact: "
                    f"parent_compatibility_fingerprints[{i}]={h!r} "
                    f"must be 64-char lower-case hex"
                )
        unique_compat_fps = set(self.parent_compatibility_fingerprints)
        if len(unique_compat_fps) > 1:
            raise ValueError(
                f"PairwiseCompareArtifact: all K parent_compatibility_"
                f"fingerprints must be identical (paired comparison "
                f"requires shared paired-data). Got "
                f"{len(unique_compat_fps)} unique values: "
                f"{sorted(unique_compat_fps)}"
            )
        if self.paired_compat_fingerprint != self.parent_compatibility_fingerprints[0]:
            raise ValueError(
                f"PairwiseCompareArtifact: paired_compat_fingerprint="
                f"{self.paired_compat_fingerprint!r} != "
                f"parent_compatibility_fingerprints[0]="
                f"{self.parent_compatibility_fingerprints[0]!r}"
            )
        # parent_model_config_hashes: each Optional[str] must be 64-hex if not None
        for i, h in enumerate(self.parent_model_config_hashes):
            if h is not None:
                if not (
                    isinstance(h, str)
                    and len(h) == 64
                    and all(c in "0123456789abcdef" for c in h)
                ):
                    raise ValueError(
                        f"PairwiseCompareArtifact: "
                        f"parent_model_config_hashes[{i}]={h!r} "
                        f"must be 64-char lower-case hex or None"
                    )
        # Treatment labels non-empty
        for i, label in enumerate(self.treatment_labels):
            if not label:
                raise ValueError(
                    f"PairwiseCompareArtifact: treatment_labels[{i}] is empty"
                )

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict.

        Used by ``content_hash`` (canonical-JSON input) and ``save``
        (on-disk persistence). Reverse via ``from_dict``.
        """
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "metric_name": self.metric_name,
            "block_length": self.block_length,
            "block_length_source": self.block_length_source,
            "n_bootstraps": self.n_bootstraps,
            "alpha": self.alpha,
            "seed": self.seed,
            "n_treatments": self.n_treatments,
            "n_samples_paired": self.n_samples_paired,
            "n_samples_raw": self.n_samples_raw,
            "n_dropped_nonfinite": self.n_dropped_nonfinite,
            "drop_fraction": self.drop_fraction,
            "primary_horizon_idx": self.primary_horizon_idx,
            "parent_experiment_ids": list(self.parent_experiment_ids),
            "parent_compatibility_fingerprints": list(
                self.parent_compatibility_fingerprints
            ),
            "parent_model_config_hashes": list(self.parent_model_config_hashes),
            "paired_compat_fingerprint": self.paired_compat_fingerprint,
            "paired_labels_sha256": self.paired_labels_sha256,
            "pairs": [dataclasses.asdict(p) for p in self.pairs],
            "treatment_labels": list(self.treatment_labels),
            "timestamp_utc": self.timestamp_utc,
            "method_caveats": list(self.method_caveats),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PairwiseCompareArtifact":
        """Reconstruct from a dict (e.g., loaded from JSON).

        Defensive:
          - MINOR-bump fields with ``None`` default gracefully absent
            in legacy artifacts.
          - ``PairwiseResultRecord`` entries filter unknown kwargs so
            future v2 per-pair field additions don't crash a v1 consumer.
          - Strict on required scalar fields.
        """
        record_fields = {f.name for f in dataclasses.fields(PairwiseResultRecord)}
        pairs = tuple(
            PairwiseResultRecord(
                **{k: v for k, v in p.items() if k in record_fields}
            )
            for p in data.get("pairs", [])
        )
        return cls(
            schema_version=data["schema_version"],
            method=data["method"],
            metric_name=data["metric_name"],
            block_length=int(data["block_length"]),
            block_length_source=data["block_length_source"],
            n_bootstraps=int(data["n_bootstraps"]),
            alpha=float(data["alpha"]),
            seed=int(data["seed"]),
            n_treatments=int(data["n_treatments"]),
            n_samples_paired=int(data["n_samples_paired"]),
            n_samples_raw=int(data["n_samples_raw"]),
            n_dropped_nonfinite=int(data["n_dropped_nonfinite"]),
            drop_fraction=float(data["drop_fraction"]),
            primary_horizon_idx=int(data["primary_horizon_idx"]),
            parent_experiment_ids=tuple(data["parent_experiment_ids"]),
            parent_compatibility_fingerprints=tuple(
                data["parent_compatibility_fingerprints"]
            ),
            parent_model_config_hashes=tuple(
                # Optional[str] preserve None values from legacy artifacts
                None if v is None else str(v)
                for v in data["parent_model_config_hashes"]
            ),
            paired_compat_fingerprint=data["paired_compat_fingerprint"],
            paired_labels_sha256=data["paired_labels_sha256"],
            pairs=pairs,
            treatment_labels=tuple(data["treatment_labels"]),
            timestamp_utc=data["timestamp_utc"],
            method_caveats=tuple(data.get("method_caveats", ())),
        )

    def save(self, path: Path) -> None:
        """Atomic write via ``hft_contracts.atomic_io`` SSoT.

        Mirror of FeatureImportanceArtifact.save + TestMetricsCIArtifact.save.
        """
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "PairwiseCompareArtifact":
        """Load from a JSON file. Strict — partial artifacts raise.

        Used by hft-ops ``_POST_STAGE_ARTIFACT_PATTERNS`` validator.
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
        """
        blob = canonical_json_blob(self.to_dict())
        return sha256_hex(blob)

    # -----------------------------------------------------------------
    # Lookup helpers
    # -----------------------------------------------------------------

    def get_pair(
        self,
        treatment_a_idx: int,
        treatment_b_idx: int,
    ) -> Optional[PairwiseResultRecord]:
        """Find pair by (a, b) indices. O(K^2) scan; K typically <= 10.

        Convention: treatment_a_idx < treatment_b_idx (the canonical
        ordering at production time). If you have an unordered pair,
        sort first.
        """
        if treatment_a_idx >= treatment_b_idx:
            return None
        for p in self.pairs:
            if (
                p.treatment_a_idx == treatment_a_idx
                and p.treatment_b_idx == treatment_b_idx
            ):
                return p
        return None

    def get_pair_by_labels(
        self,
        label_a: str,
        label_b: str,
    ) -> Optional[PairwiseResultRecord]:
        """Find pair by (label_a, label_b) — searches both orderings."""
        for p in self.pairs:
            if (
                p.treatment_a_label == label_a
                and p.treatment_b_label == label_b
            ):
                return p
            if (
                p.treatment_a_label == label_b
                and p.treatment_b_label == label_a
            ):
                # Return with sign-flipped delta if ordering reversed?
                # No — return the original record; caller handles sign.
                return p
        return None
