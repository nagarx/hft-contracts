"""
Experiment record: immutable, self-contained record of a completed experiment.

Phase 6 6B.1a (2026-04-17): Moved from `hft_ops.ledger.experiment_record` to
`hft_contracts.experiment_record` so the record dataclass — which embeds
`Provenance` (moved in 6B.4) and `feature_set_ref` — lives on the contract
plane. Every pipeline module that instantiates or consumes an ExperimentRecord
(hft-ops ledger writer, dashboard, comparison CLI, future trainer native
emission in 6B.1b) imports from here.

Each record captures the full configuration snapshot, provenance, results,
and metadata needed to reproduce and compare experiments. Records are
append-only — once written, they are never modified (except the `notes`
field for post-experiment observations).

I/O notes: this module provides `save` / `load` JSON helpers. I/O is lazy —
import has no side effects; callers invoke save/load explicitly.

Design reference: UNIFIED_PIPELINE_ARCHITECTURE_PLAN.md, Phase 4.

NOTE: Phase 6 scope split. Phase 6B.1a (THIS commit) moves the ExperimentRecord
dataclass only — keeps the hft-ops ledger writer and its re-exports intact.
Phase 6B.1b (DEFERRED to Phase 7) retires `lobtrainer.experiments.ExperimentRegistry`
entirely + ships the migration CLI that rewrites trainer-local `experiments/*.json`
into ExperimentRecord shapes under `hft-ops/ledger/records/`. The narrow-move
split limits risk while still establishing the contract-plane authority.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, FrozenSet, List, Optional

from hft_contracts.atomic_io import atomic_write_json
from hft_contracts.provenance import Provenance
from hft_contracts.signal_manifest import CONTENT_HASH_RE


# -----------------------------------------------------------------------------
# Phase 8B — Index-projection schema version (auto-invalidation substrate).
# -----------------------------------------------------------------------------
#
# ``hft-ops/ledger/index.json`` is a DERIVED projection produced by
# ``ExperimentRecord.index_entry()``; its contents depend on the whitelist of
# fields this module projects. When a developer extends the whitelist (new
# metric), old on-disk ``index.json`` entries silently omit the new key until
# someone manually runs ``hft-ops ledger rebuild-index``. Phase 8B introduces
# this constant + a companion envelope format on ``index.json`` so that a
# ``MAJOR.MINOR`` mismatch between the on-disk version and this code-side
# version is detected at load-time and triggers an automatic rebuild (loudly
# logged; ``--strict-index`` turns it into a hard error in CI).
#
# Bumping policy (mirrors ``contracts/pipeline_contract.toml [[changelog]]``
# discipline and matches ``packaging.version.Version`` MAJOR.MINOR.PATCH):
#   - MAJOR: a whitelist key is RENAMED or REMOVED. Bundle with explicit
#     migration notes in CHANGELOG; coordinate with CLI consumers.
#   - MINOR: a whitelist key is ADDED (additive; back-compat preserved via
#     re-projection on load). Default increment when extending ``index_entry()``.
#   - PATCH: documentation-only change inside ``index_entry()``; does NOT
#     trigger a rebuild (comparison logic is MAJOR.MINOR only).
#
# See root ``CLAUDE.md`` §Change-Coordination Checklist row
# "Extend ExperimentRecord.index_entry() whitelist" for the full workflow.
INDEX_SCHEMA_VERSION: str = "1.6.0"
# 1.5.0 (Phase X.3 / Phase D — Empirical Trust 2026-05-05): MINOR additive bump
# for ``experiment_provenance_hash`` projection — composes 4 existing fingerprints
# (data_export_fp + feature_set_content_hash + compatibility_fp + model_config_hash)
# into a single SHA-256 enabling cross-experiment reproducibility queries.
# Auto-rebuild fires on existing ledgers per the MAJOR.MINOR mismatch policy.
# 1.4.0 (Phase V.A.4 — 2026-04-21): added compatibility_fingerprint projection.
# 1.0.0 → 1.1.0 (Phase 8A.0, 2026-04-20): additive MINOR bump for the
# extraction-cache observability fields (``cache_hit``, ``cache_key``,
# ``cache_seconds_saved``) surfaced via ``ExperimentRecord.cache_info`` and
# projected into ``index_entry()``. Back-compat preserved: pre-Phase-8A.0
# records default ``cache_info={}``; ``hft-ops ledger.py::_load_index``
# auto-rebuilds on first post-bump load (loud WARN) to re-project legacy
# records under the new whitelist.
#
# 1.1.0 → 1.2.0 (Phase 8A.1, 2026-04-20): additive MINOR bump for the
# parallel-sweep failure taxonomy — ``sweep_failure_info`` dict field on
# ``ExperimentRecord`` surfaces ``{error_kind, exit_code, stderr_tail,
# attempt, transient}`` for grid-point failures under
# ``record_type=sweep_failure``. Empty dict on all non-failure records.
# Projected in ``index_entry()`` so ``hft-ops ledger list`` can filter by
# failure type without loading full records. Back-compat: pre-Phase-8A.1
# records default ``sweep_failure_info={}``; auto-rebuild on first load.
#
# 1.2.0 → 1.3.0 (Phase 8C-α Stage C.2, 2026-04-20): additive MINOR bump
# for post-training feature-importance artifacts. New
# ``ExperimentRecord.artifacts`` field (List[Dict]) carries references
# to content-addressed artifact files (currently:
# ``feature_importance_v1.json`` from the trainer's permutation-
# importance callback; future: SHAP / IntegratedGradients). Each entry:
# ``{kind, path, sha256, bytes, method}``. Empty list on pre-Phase-8C-α
# records. ``index_entry()`` projects ``artifact_kinds`` — sorted set
# of distinct ``kind`` values — for fast ``ledger list
# --has-artifact feature_importance`` queries without loading full
# record bodies. Schema for ``feature_importance`` kind is registered
# in ``contracts/pipeline_contract.toml [artifacts.feature_importance_schema]``.


class RecordType(str, Enum):
    """Type of experiment a ledger record represents.

    Introduced in Phase 1.3 to accommodate the full scope of past/future
    experiments. Not every "experiment" produces a trainer ``history.json`` —
    many analytical studies (E7-E16) produce only analyzer/evaluator output,
    and some post-hoc calibrations (E6) produce only signals.

    Each type has a reduced-fidelity schema:

    - ``training``: Full trainer run. ``training_metrics`` populated, optionally
      ``backtest_metrics`` too. The default.
    - ``analysis``: Diagnostic study (no training, no backtest). Results live
      in ``training_metrics`` as a free-form dict OR in ``notes``.
    - ``calibration``: Post-hoc calibration of an existing trained model.
      References a parent training record via ``parent_experiment_id``.
    - ``backtest``: Backtest-only experiment (pre-existing model, pre-existing
      signals). ``backtest_metrics`` populated.
    - ``evaluation``: 5-path feature-evaluator run producing a classification
      table / feature profiles. ``training_metrics`` holds the summary.
    - ``sweep_aggregate``: Aggregate record for a multi-run script (e.g.,
      e4_baselines.py that runs 5 models). Sub-results live in ``sub_records``.
    - ``sweep_failure`` (Phase 8A.1, 2026-04-20): A grid-point that failed
      under parallel sweep execution. Carries ``sweep_failure_info``
      {error_kind, exit_code, stderr_tail, attempt, transient} for
      diagnosis and retry decisions. Shares its ``fingerprint`` with the
      would-be successful record for that treatment so retries match;
      ``hft-ops/ledger/dedup.py::check_duplicate`` MUST filter this type
      out so retries re-run instead of being silently skipped as
      duplicates.
    """

    TRAINING = "training"
    ANALYSIS = "analysis"
    CALIBRATION = "calibration"
    BACKTEST = "backtest"
    EVALUATION = "evaluation"
    SWEEP_AGGREGATE = "sweep_aggregate"
    SWEEP_FAILURE = "sweep_failure"


@dataclass
class ExperimentRecord:
    """Complete record of a single experiment run.

    Attributes:
        experiment_id: Unique identifier ({name}_{timestamp}_{fingerprint[:8]}).
        name: Human-readable experiment name (from manifest).
        manifest_path: Absolute path to the source manifest YAML.
        fingerprint: SHA-256 of resolved config for dedup.

        provenance: Full provenance (git, config hashes, data hash, timestamp).
        contract_version: Pipeline contract version at time of experiment.

        extraction_config: Full extractor TOML as dict.
        training_config: Full trainer YAML as dict.
        backtest_params: Backtest parameters as dict.

        training_metrics: Training results (accuracy, f1, per-class, etc.).
        backtest_metrics: Backtest results (return, sharpe, drawdown, etc.).
        dataset_health: Key stats from dataset analysis.

        tags: User-defined tags for filtering.
        hypothesis: What the experiment aims to test.
        description: Detailed experiment description.
        notes: Post-experiment observations (mutable field).

        created_at: ISO 8601 creation timestamp.
        duration_seconds: Wall-clock time for full pipeline.
        status: completed | failed | partial.
        stages_completed: Which stages ran successfully.
    """

    experiment_id: str = ""
    name: str = ""
    manifest_path: str = ""
    fingerprint: str = ""

    # Phase 4 Batch 4c.4 (2026-04-16): optional reference to the FeatureSet
    # registry entry used at trainer time. Top-level (not nested under
    # `Provenance.config_hashes`) because it's a structured reference with
    # identity (name + content_hash), not an opaque config-hash. Query
    # pattern: `record.feature_set_ref["name"] == "momentum_v1"`.
    # None iff the trainer did not use `DataConfig.feature_set`
    # (legacy path, explicit feature_indices, or feature_preset).
    # See PA §13.4.2 for why this is NOT in Provenance.config_hashes
    # (that dict's implicit contract is values are SHA-256 hex; forcing a
    # structured reference into it would violate that).
    feature_set_ref: Optional[Dict[str, str]] = None

    # Phase V.A.4 (2026-04-21): optional reference to the CompatibilityContract
    # fingerprint that the trainer emitted into ``signal_metadata.json``.
    # The contract's 11 shape-determining fields (contract_version,
    # schema_version, feature_count, window_size, feature_layout,
    # data_source, label_strategy_hash, calibration_method,
    # primary_horizon_idx, horizons, normalization_strategy) are hashed via
    # ``hft_contracts.canonical_hash`` to produce a 64-hex SHA-256 digest.
    # This column makes the "trustworthy experiment" claim FALSIFIABLE at the
    # ledger layer: ``hft-ops ledger list --compatibility-fp <hex>`` surfaces
    # every experiment produced against a specific contract version.
    #
    # None iff: (a) trainer did not emit the compatibility block (pre-Phase-II
    # legacy signal_metadata.json format), OR (b) hft-ops harvester could not
    # read signal_metadata.json (missing file, malformed JSON), OR (c) the
    # stored value did not match the 64-hex SHA-256 regex ``CONTENT_HASH_RE``
    # (fail-loud rejection of poisoned input; same pattern as feature_set_ref
    # validation).
    #
    # Query pattern: ``record.compatibility_fingerprint == "abc123..."``.
    # Filter via ``ledger.filter(compatibility_fingerprint=...)`` or
    # ``hft-ops ledger list --compatibility-fp <hex>``.
    #
    # Fingerprint stability: this field IS an observation (set post-training
    # by the harvester). MUST NOT affect ``dedup.compute_fingerprint`` — the
    # fingerprint is a property OF the experiment, not an input TO it. Same
    # invariant as ``gate_reports`` / ``artifacts`` / ``cache_info``.
    #
    # Index projection: ``index_entry()`` projects this field with 64-hex
    # validation via ``CONTENT_HASH_RE``; malformed values surface as empty
    # string "" in the index (graceful degradation for ledger queries).
    compatibility_fingerprint: Optional[str] = None

    # Phase X.3 / Phase D Empirical Trust (2026-05-05): the long-promised
    # Phase Y trust column — composes 4 existing fingerprints into a single
    # SHA-256 enabling cross-experiment reproducibility queries:
    #
    #     experiment_provenance_hash = sha256(canonical_json_blob({
    #         "data_export_fp": provenance.data_dir_hash,
    #         "feature_set_content_hash": feature_set_ref["content_hash"],
    #         "compatibility_fp": compatibility_fingerprint,
    #         "model_config_hash": training_config["model_config_hash"],
    #     }))
    #
    # The 4 components are ALREADY computed + stored separately on the
    # record (Phase II + Phase 4.4c.4 + Phase X.1 v2 + Phase V.A.4). This
    # field composes them into a single 64-hex identity for the experiment
    # — same data + same features + same architecture + same loss-tuning
    # invariants → same hash. Different ANY of the 4 → different hash.
    #
    # USE CASES:
    #   1. ``hft-ops ledger list --provenance-hash <hex>`` — find all runs
    #      with a specific complete-state identity (e.g., re-runs of an
    #      experiment to verify reproducibility).
    #   2. Reproducibility audit — same config + same code should produce
    #      same provenance hash; drift indicates an upstream change
    #      (model_config_hash bumped, feature_set rotated, data re-extracted).
    #   3. Cross-experiment composability — researchers comparing models
    #      on the same data corpus can filter by compatibility_fingerprint
    #      AND model_config_hash separately, OR by experiment_provenance_hash
    #      for full-stack identity.
    #
    # GRACEFUL DEGRADATION: ``None`` when ANY of the 4 components is missing
    # (pre-Phase-II / pre-Phase-4.4c.4 / pre-Phase-X.1 v2 / pre-Phase-V.A.4
    # legacy records). Use ``compute_experiment_provenance_hash(record)``
    # to compute on-demand for older records that have all 4 components
    # populated.
    #
    # FINGERPRINT INVARIANT: this field IS an observation (composed from
    # other observations). MUST NOT enter ``dedup.compute_fingerprint`` —
    # same invariant as ``gate_reports`` / ``artifacts`` / ``cache_info`` /
    # ``compatibility_fingerprint`` / ``signal_export_output_dir``.
    #
    # Index projection: ``index_entry()`` projects this field with 64-hex
    # validation via ``CONTENT_HASH_RE``; malformed values surface as empty
    # string "" (graceful degradation).
    experiment_provenance_hash: Optional[str] = None

    # Phase V.1 L1.2 (2026-04-21): resolved ABSOLUTE path to the signal-export
    # output directory captured at RUN TIME (not re-resolved from the manifest
    # post-hoc). Closes Agent 2 H1 manifest-move-resilience gap surfaced by
    # the Phase V post-audit cross-cutting review.
    #
    # Without this field, ``hft_ops.ledger.statistical_compare._resolve_signal_dir``
    # must re-load the manifest YAML + re-resolve variable substitutions to
    # find the signal files — fragile if the monorepo is moved OR the
    # manifest is edited post-run OR variable-substitution context is no
    # longer resolvable at query time. With this field, consumers can trust
    # the run-time-resolved absolute path for the lifetime of the record.
    #
    # None iff: (a) signal_export stage was disabled for this experiment
    # (training-only or dry-run), OR (b) the stage ran but did not set
    # ``output_dir`` on the SignalExportStage config. The cli.py attachment
    # logic only populates this when BOTH conditions are satisfied —
    # graceful degradation, no silent corruption.
    #
    # Query pattern: ``pathlib.Path(record.signal_export_output_dir)``.
    # This field is an OBSERVATION (set post-stage) and MUST NOT affect
    # ``dedup.compute_fingerprint`` — same invariant as ``gate_reports`` /
    # ``artifacts`` / ``compatibility_fingerprint`` / ``cache_info``.
    #
    # Not projected into ``index_entry()`` — the path is an implementation
    # detail used by the statistical-compare adapter + future tooling; not
    # a user-facing filter axis. Loading the full record via
    # ``ExperimentLedger.get(exp_id)`` is the access pattern.
    signal_export_output_dir: Optional[str] = None

    provenance: Provenance = field(default_factory=Provenance)
    contract_version: str = ""

    extraction_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    backtest_params: Dict[str, Any] = field(default_factory=dict)

    training_metrics: Dict[str, Any] = field(default_factory=dict)
    backtest_metrics: Dict[str, Any] = field(default_factory=dict)
    dataset_health: Dict[str, Any] = field(default_factory=dict)

    # Phase 7 Stage 7.4 Round 4 item #2 (2026-04-20). Generic surface
    # for stage-level gate reports — keyed by runner stage_name
    # (``"validation"``, ``"post_training_gate"``, future
    # ``"post_backtest_gate"``, ...). Inner shape is the runner's own
    # ``report.to_dict()`` output — deliberately ``Dict[str, Any]``
    # (not a typed shape) because different gates emit different
    # field names (``status`` vs ``verdict`` today); a typed
    # ``GateReportBase`` contract can be introduced later once ≥3
    # gates stabilize on a common set of fields.
    #
    # ``gate_reports`` replaces the Round 1 pattern of nesting under
    # ``training_metrics["post_training_gate"]`` — that pattern
    # violated the training_metrics scalar-dict convention AND was
    # silently filtered from ``index_entry()``. Legacy records load
    # correctly via the ``from_dict`` migration shim below.
    #
    # INVARIANT: gate_reports MUST NOT affect the fingerprint
    # (``hft-ops/src/hft_ops/ledger/dedup.py::compute_fingerprint``
    # only hashes the resolved trainer config). The rationale:
    # identical inputs + identical code produce the same experiment
    # identity regardless of gate outcome — a gate is an
    # *observation*, not a *treatment*. Test coverage locked in
    # ``hft-contracts/tests/test_experiment_record.py::TestGateReportsFingerprintStability``.
    gate_reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Phase 8A.0 (2026-04-20): extraction-cache observability fields.
    # Harvested by ``hft-ops cli.py::_record_experiment`` from the
    # extraction stage's ``captured_metrics`` — flattened into record-
    # level ``cache_info`` for ledger-query ergonomics (vs reaching into
    # ``stages.extraction.captured_metrics.cache_hit``).
    #
    # Schema (all fields optional — empty dict for pre-Phase-8A.0 records):
    #   ``cache_hit: bool``       — True iff cache resolved + validated;
    #   ``cache_key: str``        — 64-char SHA-256 of the 9 inputs;
    #   ``cache_seconds_saved: float`` — 0.0 on miss; >0 on hit
    #     (``extractor_duration_seconds`` from CACHE_MANIFEST.json);
    #   ``cache_linked_files: int`` (hits only);
    #   ``cache_link_type: str``  — one of clonefile/reflink/
    #     hardlink_readonly/symlink_relative (hits only).
    #
    # Fingerprint-stability: NONE of these enter the ExperimentRecord
    # fingerprint. Cache outcome is observation, not treatment
    # (Invariant 4). Locked by
    # ``test_extraction_cache.py::TestFingerprintStabilityAcrossCache``.
    cache_info: Dict[str, Any] = field(default_factory=dict)

    # Phase 8A.1 (2026-04-20): parallel-sweep failure taxonomy.
    # Populated ONLY for ``record_type=sweep_failure`` records.
    #
    # Schema (all fields optional; empty dict for non-failure records):
    #   ``error_kind: str``      — "oom" | "validation_error" |
    #     "gpu_acquire_timeout" | "broken_process_pool" | "subprocess_nonzero" |
    #     "assertion" | "unknown"
    #   ``exit_code: int``       — subprocess exit code (137 = SIGKILL/OOM,
    #     139 = SIGSEGV, etc.); -1 for non-subprocess failures
    #   ``stderr_tail: str``     — last 4KB of stderr for diagnosis
    #   ``attempt: int``         — retry attempt number (1 = first, 2+ = retry)
    #   ``transient: bool``      — True if retryable (OOM, timeout); False
    #     for fatal (AssertionError, config invalid)
    #
    # Fingerprint invariant: sweep_failure records share their fingerprint
    # with the would-be successful record for the same treatment — so a
    # retry that completes successfully matches the same fingerprint for
    # downstream comparison. ``dedup.check_duplicate`` MUST filter out
    # record_type=sweep_failure so retries are not silently blocked.
    # Locked by ``test_scheduler_parallel.py::TestDedupSkipsSweepFailure``.
    sweep_failure_info: Dict[str, Any] = field(default_factory=dict)

    # Phase 8C-α Stage C.2 (2026-04-20): post-training artifact references.
    # Populated by hft-ops ledger routing (``_persist_post_stage_artifacts``)
    # when stages produce content-addressable artifacts. Currently:
    # ``feature_importance_v1.json`` from trainer's
    # ``PermutationImportanceCallback`` (Phase 8C-α Stage C.1).
    #
    # Schema per entry (all Dict[str, Any] — flexible for future
    # artifact kinds without schema bumps):
    #   ``kind: str``      — "feature_importance" | future: "shap",
    #     "integrated_gradients". The registry of valid kinds is
    #     documented in ``contracts/pipeline_contract.toml``.
    #   ``path: str``      — relative to pipeline_root; e.g.,
    #     "hft-ops/ledger/feature_importance/2026_04/<sha256>.json"
    #   ``sha256: str``    — content hash for integrity verification
    #   ``bytes: int``     — file size
    #   ``method: str``    — sub-classification of kind (e.g.,
    #     "permutation" for feature_importance). Optional.
    #
    # Empty list on pre-Phase-8C-α records. The ``from_dict`` shim
    # defaults to [] when the field is absent (legacy records).
    # ``index_entry()`` projects ``artifact_kinds`` — sorted list of
    # distinct kinds — for fast ``ledger list --has-artifact`` queries.
    #
    # Fingerprint-stability: ``artifacts[]`` is NOT part of
    # ``compute_fingerprint``. Post-training artifacts are observations,
    # not treatments (same treatment + different post-hoc analysis →
    # same fingerprint).
    artifacts: List[Dict[str, Any]] = field(default_factory=list)

    tags: List[str] = field(default_factory=list)
    hypothesis: str = ""
    description: str = ""
    notes: str = ""

    created_at: str = ""
    duration_seconds: float = 0.0
    status: str = "pending"
    stages_completed: List[str] = field(default_factory=list)

    # Sweep metadata (populated when this record is part of a sweep)
    sweep_id: str = ""
    """Sweep identifier linking this record to its parent sweep."""

    axis_values: Dict[str, str] = field(default_factory=dict)
    """Axis name -> selected label for this grid point (e.g., {"model": "tlob", "horizon": "H10"})."""

    # Phase 1.3 record-typing fields
    record_type: str = "training"
    """Type of record. One of: training, analysis, calibration, backtest, evaluation,
    sweep_aggregate. Use the ``RecordType`` enum's ``.value`` for type-safety.
    Default ``training`` preserves backward compat with pre-Phase-1.3 records."""

    sub_records: List[Dict[str, Any]] = field(default_factory=list)
    """For ``sweep_aggregate`` records: per-sub-experiment summaries (typically
    {"name": ..., "training_metrics": {...}, "config_diff": {...}}).
    Empty for non-aggregate types."""

    parent_experiment_id: str = ""
    """For ``calibration`` and ``backtest`` records: the experiment_id of the
    upstream record this one depends on (e.g., calibration → its trained model)."""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d = asdict(self)
        d["provenance"] = self.provenance.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentRecord:
        """Deserialize from a dict.

        Phase 6 post-validation hardening (2026-04-18): non-mutating —
        reads via ``data.get(...)`` so callers can pass the same dict
        multiple times (e.g., through a cache layer) without the second
        call finding ``provenance`` missing.

        Phase 7 Stage 7.4 Round 4 migration shim (2026-04-20, removal
        deadline 2026-08-01): records written between Round 1
        (2026-04-19) and Round 4 nested the post-training gate report
        under ``training_metrics["post_training_gate"]``. Lift it to
        the new ``gate_reports`` field so the whole ledger loads with
        uniform shape. ``training_metrics["post_training_gate_summary"]``
        was a redundant one-line projection of ``.summary()`` — drop
        rather than migrate (not part of the new contract). Fresh
        records never emit either key, so after 2026-08-01 the shim
        is a no-op and can be deleted.
        """
        prov_data = data.get("provenance", {})
        record = cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__ and k != "provenance"
        })
        record.provenance = Provenance.from_dict(dict(prov_data))

        # Migration shim — lift legacy nested gate report into gate_reports.
        # Phase 7 Stage 7.4 Round 5 (2026-04-20) hardening: shallow-copy
        # ``training_metrics`` BEFORE mutating. ``cls(**{...})`` passes the
        # caller's dict by reference, so ``.pop()`` on ``record.training_metrics``
        # would mutate ``data["training_metrics"]`` too — surprising callers
        # that hold the input (cache layers, round-trip tests, etc.). Copy
        # isolates the mutation to the record's own attribute.
        record.training_metrics = dict(record.training_metrics)
        legacy_gate = record.training_metrics.pop("post_training_gate", None)
        if isinstance(legacy_gate, dict):
            record.gate_reports.setdefault("post_training_gate", legacy_gate)
        # Drop the legacy summary projection — redundant with GateReport.summary().
        record.training_metrics.pop("post_training_gate_summary", None)

        return record

    def save(self, path: Path) -> None:
        """Save record to a JSON file atomically.

        Phase 7 Stage 7.4 Round 5 (2026-04-20): delegates to the
        canonical ``hft_contracts.atomic_io.atomic_write_json`` (tmp
        + fsync + os.replace) — unified with
        ``hft_ops.feature_sets.writer.atomic_write_json`` and
        ``hft_ops.ledger.ledger._save_index`` to prevent serialization-
        convention drift across the three sites.

        Prior to Round 4, this method used non-atomic
        ``open(w) + json.dump`` — vulnerable to silent data loss: if
        the write was interrupted (SIGKILL, ENOSPC, power failure)
        the record file would be left partial, and
        ``ExperimentLedger._rebuild_index`` would drop it on the next
        load (``ledger.py`` catches ``JSONDecodeError`` and skips).
        That is silent data loss, not corruption.

        Canonical serialization (``sort_keys=True`` + trailing
        newline) ensures byte-equal output across runs — diff-stable
        records for git history + content-addressable fingerprinting.
        """
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> ExperimentRecord:
        """Load record from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def index_entry(self) -> Dict[str, Any]:
        """Create a lightweight index entry for fast ledger queries.

        Contains enough metadata for filtering and comparison without
        loading the full record.
        """
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "fingerprint": self.fingerprint,
            "contract_version": self.contract_version,
            "tags": self.tags,
            "hypothesis": self.hypothesis,
            "status": self.status,
            "stages_completed": self.stages_completed,
            "created_at": self.created_at,
            "duration_seconds": self.duration_seconds,
            "training_metrics": {
                k: v for k, v in self.training_metrics.items()
                if k in (
                    # Classification (pre-Phase-7 baseline keys — retroactive records)
                    "accuracy", "macro_f1", "macro_precision", "macro_recall",
                    "best_val_accuracy", "best_val_macro_f1", "best_epoch",
                    # Regression test-split (Phase 7 Stage 7.4 Round 1,
                    # 2026-04-19) — emitted by TrainingRunner when
                    # test_metrics.json is present. Required by
                    # PostTrainingGateRunner for prior-best regression IC
                    # comparison.
                    "test_ic", "test_directional_accuracy", "test_r2",
                    "test_mae", "test_rmse", "test_pearson",
                    "test_profitable_accuracy",
                    # Classification test-split (Phase 7 Stage 7.4 Round 5,
                    # 2026-04-20) — Round 4 item #2 added
                    # ``scripts/train.py::_dump_test_metrics`` which
                    # unconditionally prefixes every key with ``test_``,
                    # so ClassificationMetrics.to_dict() lands as
                    # ``test_accuracy`` / ``test_macro_f1`` etc. Without
                    # whitelisting, these silently vanish from the index
                    # — identical root cause as the Round 1 regression
                    # gap Round 4 was meant to fix, shifted from
                    # regression to classification. Core scalars only
                    # (no per-class precision/recall/F1 to prevent
                    # index bloat, mirroring the regression convention).
                    "test_accuracy", "test_macro_f1",
                    "test_macro_precision", "test_macro_recall",
                    "test_loss",
                    # Per-epoch val_* best values (Phase 7 Stage 7.4
                    # Round 4, 2026-04-20) — Round 1's test_* whitelist
                    # was silently dead for every PyTorch TLOB/HMHP run
                    # that never persisted test_metrics.json. Round 4
                    # item #6 added that persistence; item #1 here
                    # surfaces the per-epoch best values extracted by
                    # TrainingRunner._capture_training_metrics so that
                    # _find_prior_best_experiment can fall back to
                    # best_val_ic when test_ic is absent (early-phase
                    # runs, in-flight experiments). Max-better (5):
                    "best_val_ic", "best_val_directional_accuracy",
                    "best_val_r2", "best_val_pearson",
                    "best_val_profitable_accuracy",
                    # Min-better (3):
                    "best_val_loss", "best_val_mae", "best_val_rmse",
                    # Classification extra (1) — signal_rate emitted by
                    # TLOB / opportunity strategies.
                    "best_val_signal_rate",
                )
            },
            "backtest_metrics": {
                k: v for k, v in self.backtest_metrics.items()
                if k in (
                    "total_return", "sharpe_ratio", "max_drawdown",
                    "win_rate", "total_trades",
                )
            },
            "model_type": self.training_config.get("model", {}).get("model_type", ""),
            "labeling_strategy": self.training_config.get("data", {}).get(
                "labeling_strategy", ""
            ),
            "sweep_id": self.sweep_id,
            "axis_values": self.axis_values,
            "record_type": self.record_type,
            "parent_experiment_id": self.parent_experiment_id,
            "retroactive": self.provenance.retroactive,
            # Phase 4 Batch 4c.4: surface feature_set_ref in index for
            # `hft-ops ledger list --feature-set <name>` filtering. Empty dict
            # (not None) when unset, matches other Dict default conventions.
            "feature_set_ref": self.feature_set_ref or {},
            # Phase V.A.4 (2026-04-21): surface compatibility_fingerprint
            # in index for `hft-ops ledger list --compatibility-fp <hex>`
            # filtering. Empty string "" (not None) when unset, matches
            # the JSON-default convention for Optional[str] fields in the
            # index projection. Malformed values (not 64-hex) are silently
            # coerced to "" — graceful degradation for poisoned records
            # (same gate as feature_set_ref.content_hash harvest).
            "compatibility_fingerprint": (
                self.compatibility_fingerprint
                if (
                    isinstance(self.compatibility_fingerprint, str)
                    and bool(CONTENT_HASH_RE.match(self.compatibility_fingerprint))
                )
                else ""
            ),
            # Phase X.3 / Phase D Empirical Trust (2026-05-05): surface
            # experiment_provenance_hash for cross-experiment reproducibility
            # filtering via ``hft-ops ledger list --provenance-hash <hex>``.
            # Same regex gate + graceful-degradation pattern as
            # compatibility_fingerprint (Phase V.A.4). Composed from 4
            # other fingerprints by ``compute_experiment_provenance_hash``;
            # see field docstring for composition formula.
            "experiment_provenance_hash": (
                self.experiment_provenance_hash
                if (
                    isinstance(self.experiment_provenance_hash, str)
                    and bool(CONTENT_HASH_RE.match(self.experiment_provenance_hash))
                )
                else ""
            ),
            # Phase Y / γ-1 LITE close-out (#PY-94, 2026-05-10): surface
            # ``model_config_hash`` as a top-level mirror in the index
            # projection. The Phase Y composer reads from
            # ``training_config["model_config_hash"]`` (per
            # ``_extract_provenance_components`` at L749) — that nested
            # value IS populated at trainer write time (sklearn at
            # ``simple_trainer.py`` sidecar; PyTorch at
            # ``_build_checkpoint_dict``). Without this projection,
            # ``hft-ops ledger list --model-config-hash <hex>`` queries
            # cannot filter. #PY-94 surfaced this gap during γ-1 LITE
            # empirical gate: 12 records had populated nested mch
            # (sklearn=``be40f8f0...``, TLOB=``de47c0ef...``) but
            # 0 top-level projection because the field exists only
            # nested in ``training_config``. Same regex gate +
            # graceful-degradation pattern as
            # ``compatibility_fingerprint`` (Phase V.A.4) and
            # ``experiment_provenance_hash`` (Phase X.3) above.
            "model_config_hash": (
                (self.training_config or {}).get("model_config_hash", "")
                if (
                    isinstance(
                        (self.training_config or {}).get("model_config_hash"),
                        str,
                    )
                    and bool(
                        CONTENT_HASH_RE.match(
                            (self.training_config or {}).get("model_config_hash")
                            or ""
                        )
                    )
                )
                else ""
            ),
            # Phase 8A.0 (2026-04-20): extraction-cache observability.
            # Empty dict (not None) for pre-Phase-8A.0 records so
            # ``ledger list --cache-hit true`` has a stable shape. Full
            # schema documented on ``ExperimentRecord.cache_info`` field.
            "cache_info": self.cache_info or {},
            # Phase 8A.1 (2026-04-20): surface parallel-sweep failure
            # taxonomy for ``ledger list --failure-kind oom`` and similar
            # filters. Empty dict on non-failure records — shape-stable
            # for index queries. Schema documented on
            # ``ExperimentRecord.sweep_failure_info`` field.
            "sweep_failure_info": self.sweep_failure_info or {},
            # Phase 8C-α Stage C.2 (2026-04-20): surface distinct artifact
            # kinds for ``ledger list --has-artifact feature_importance``
            # filtering. Sorted for deterministic projection (hft-rules §7
            # — no dict/set-ordering in externally-visible output).
            # Empty list on pre-Phase-8C-α records.
            "artifact_kinds": sorted({
                a.get("kind")
                for a in self.artifacts
                if isinstance(a, dict)
                and isinstance(a.get("kind"), str)
                and a.get("kind")
            }),
            # Phase 7 Stage 7.4 Round 4 (2026-04-20): surface gate
            # outcome per stage for fast filtering ("show me all
            # experiments where post_training_gate warned"). Project
            # only the status + a truncated summary — the full report
            # stays in the record body to keep index.json small.
            # Summary cap matches the Round 1 one-line convention;
            # a multi-paragraph summary would bloat index lookups.
            # Phase 7 Stage 7.4 Round 5 (2026-04-20): ``status`` is
            # now the canonical key per
            # ``hft_contracts.gate_report.GateReportDict``. The
            # validation stage adapter injects ``status`` (lowercased
            # verdict) before writing, so the legacy coalesce
            # ``.get("status") or .get("verdict")`` is no longer
            # needed. Removing it prevents casing inconsistency
            # (``"PASS"`` vs ``"pass"``) from leaking into
            # ``ledger list --gate-status`` queries.
            "gate_reports": {
                stage: {
                    "status": report.get("status", ""),
                    "summary": str(report.get("summary", ""))[:256],
                }
                for stage, report in (self.gate_reports or {}).items()
                if isinstance(report, dict)
            },
        }


@dataclass(frozen=True)
class ProvenanceDiagnostic:
    """Diagnostic result describing which provenance-hash components are present + valid.

    Phase X.3 / REFINED-PLUS Sub-cycle 2 (2026-05-09 night): structured diagnostic
    paired with :func:`compute_experiment_provenance_hash`. Closes PHASE_P_BACKLOG
    L891 mitigation for #PY-49 (cli.py:636-644 inline missing-list duplication
    becomes consumable from a single home).

    Component names live as the :attr:`COMPONENT_NAMES` ``ClassVar`` constant —
    the SSoT for the 4 fingerprint sources — so callers cannot typo-drift
    relative to the composer (per hft-rules §1).

    Used by:
      - :func:`compute_experiment_provenance_hash` for ``required`` validation.
      - hft-ops ``cli.py::_record_experiment`` warning diagnostic (replaces
        inline missing-list logic at cli.py:639-666; Sub-cycle 4b refactor).
      - Future fail-loud callers that want structured missing/invalid info
        without raising.

    Empty-string values count as **missing** (mirrors the composer's existing
    ``if not all(components.values())`` semantics — producer-side convention
    is that an un-populated 64-hex string surfaces as ``None`` or ``""``).

    Attributes:
        complete: True iff all 4 components are present + non-empty + valid
            lowercase 64-hex SHA-256.
        missing: ``FrozenSet`` of component names whose value is None/empty.
        invalid_format: ``FrozenSet`` of component names whose value is present
            but does not match :data:`hft_contracts.signal_manifest.CONTENT_HASH_RE`
            (lowercase 64-hex SHA-256).
    """

    COMPONENT_NAMES: ClassVar[FrozenSet[str]] = frozenset({
        "data_export_fp",
        "feature_set_content_hash",
        "compatibility_fp",
        "model_config_hash",
    })

    complete: bool
    missing: FrozenSet[str]
    invalid_format: FrozenSet[str]


def _extract_provenance_components(record: "ExperimentRecord") -> Dict[str, Optional[str]]:
    """Internal helper: extract the 4 fingerprint components from a record.

    Single source of truth for component-extraction semantics. Both
    :func:`diagnose_provenance_completeness` and
    :func:`compute_experiment_provenance_hash` consume this helper so they
    cannot disagree on what counts as "present" vs "missing".
    """
    return {
        "data_export_fp": record.provenance.data_dir_hash if record.provenance else None,
        "feature_set_content_hash": (record.feature_set_ref or {}).get("content_hash"),
        "compatibility_fp": record.compatibility_fingerprint,
        "model_config_hash": (record.training_config or {}).get("model_config_hash"),
    }


def diagnose_provenance_completeness(record: "ExperimentRecord") -> ProvenanceDiagnostic:
    """Diagnose which provenance-hash components are present + valid.

    Phase X.3 / REFINED-PLUS Sub-cycle 2 (2026-05-09 night): the SSoT validator
    for the 4 components that :func:`compute_experiment_provenance_hash`
    composes. Lives alongside the composer per PHASE_P_BACKLOG L891 mitigation
    (#PY-49 + #PY-91).

    Empty-string values are classified as **missing**, not **invalid_format** —
    matches the composer's existing ``if not all(components.values())`` semantics
    and reflects the producer-side convention that an un-populated 64-hex string
    surfaces as ``None`` or ``""``.

    Args:
        record: The :class:`ExperimentRecord` to diagnose.

    Returns:
        :class:`ProvenanceDiagnostic` with ``complete`` / ``missing`` /
        ``invalid_format`` fields.
    """
    components = _extract_provenance_components(record)

    missing = frozenset(name for name, value in components.items() if not value)
    invalid_format = frozenset(
        name
        for name, value in components.items()
        if value and not CONTENT_HASH_RE.match(value)
    )
    complete = not missing and not invalid_format

    return ProvenanceDiagnostic(
        complete=complete,
        missing=missing,
        invalid_format=invalid_format,
    )


def compute_experiment_provenance_hash(
    record: "ExperimentRecord",
    *,
    required: Optional[FrozenSet[str]] = None,
) -> Optional[str]:
    """Compose ``experiment_provenance_hash`` from 4 existing fingerprints.

    Phase X.3 / Phase D Empirical Trust (2026-05-05): the long-promised
    Phase Y trust column. Composes:

        - ``record.provenance.data_dir_hash``     (Phase 6 Provenance)
        - ``record.feature_set_ref["content_hash"]``  (Phase 4.4c.4)
        - ``record.compatibility_fingerprint``    (Phase V.A.4)
        - ``record.training_config["model_config_hash"]``  (Phase X.1 v2)

    via canonical-JSON SHA-256 (matching the ``canonical_hash`` SSoT
    convention used everywhere else — content-hashed FeatureSet,
    CompatibilityContract.fingerprint, FeatureImportanceArtifact, etc.).

    Same data + same features + same architecture + same loss-tuning
    invariants → same hash. Mutating ANY of the 4 components → different
    hash. Enables cross-experiment reproducibility queries:

        ``hft-ops ledger list --provenance-hash 43374f95...``  →
        finds all records with this exact 4-fingerprint composition.

    REFINED-PLUS Sub-cycle 2 (2026-05-09 night): added the keyword-only
    ``required`` parameter for fail-loud opt-in. When ``required`` is set,
    missing or invalid-format components in the required-set raise
    :class:`ValueError` instead of silently returning ``None``. Closes the
    :class:`lobmodels.registry.protocols.OrchestratorContract` ``requires_*``
    contract pre-committed by Sub-cycle 1a (Phase Y composability).

    Args:
        record: The :class:`ExperimentRecord` to compose from.
        required: Optional ``FrozenSet`` of component names that MUST be
            present and valid lowercase 64-hex SHA-256. When ``None``
            (default), preserves silent-None graceful-degradation back-compat.
            When provided, raises ``ValueError`` listing missing +
            invalid-format component names. Valid names are
            :attr:`ProvenanceDiagnostic.COMPONENT_NAMES`; unknown names also
            raise ``ValueError``.

    Returns:
        64-hex SHA-256 string when all 4 components are present and valid,
        OR ``None`` if ``required is None`` AND any component is
        missing/invalid (graceful degradation for pre-Phase-II /
        pre-Phase-4.4c.4 / pre-Phase-X.1 v2 / pre-Phase-V.A.4 legacy records).

    Raises:
        ValueError: If ``required`` contains unknown component names, OR if
            any required component is missing or invalid-format.

    Per hft-rules §0 (reuse-first): delegates to existing
    ``hft_contracts.canonical_hash.canonical_json_blob`` + ``sha256_hex``
    SSoT — NO new canonical-form site. Per hft-rules §1 (single source of
    truth): component names live on
    :attr:`ProvenanceDiagnostic.COMPONENT_NAMES` — never duplicated.
    """
    from hft_contracts.canonical_hash import canonical_json_blob, sha256_hex

    # Eagerly reject unknown component names BEFORE running the diagnostic —
    # input validation precedes work (mid-impl gate refinement 2026-05-09).
    # Catches caller typo / stale required-set after future component addition.
    # Per hft-rules §1 SSoT discipline (component names canonical home is
    # ProvenanceDiagnostic.COMPONENT_NAMES).
    if required is not None:
        unknown = required - ProvenanceDiagnostic.COMPONENT_NAMES
        if unknown:
            raise ValueError(
                f"Unknown component names in `required`: {sorted(unknown)}. "
                f"Valid names: {sorted(ProvenanceDiagnostic.COMPONENT_NAMES)}."
            )

    diagnostic = diagnose_provenance_completeness(record)

    if required is not None:
        required_missing = diagnostic.missing & required
        required_invalid = diagnostic.invalid_format & required
        if required_missing or required_invalid:
            raise ValueError(
                f"Cannot compose experiment_provenance_hash for required set "
                f"{sorted(required)}: "
                f"missing={sorted(required_missing) or 'none'}, "
                f"invalid_format={sorted(required_invalid) or 'none'}. "
                f"Use diagnose_provenance_completeness() for the full diagnostic."
            )

    if not diagnostic.complete:
        # Graceful degradation path: at least one of the 4 components is
        # missing or invalid AND no `required` set was passed → silent None.
        # The consumer (CLI filter, reproducibility audit) distinguishes "no
        # provenance" from "different provenance" via this null sentinel.
        return None

    components = _extract_provenance_components(record)
    return sha256_hex(canonical_json_blob(components))


__all__ = [
    "ExperimentRecord",
    "RecordType",
    "INDEX_SCHEMA_VERSION",
    "compute_experiment_provenance_hash",
    "diagnose_provenance_completeness",
    "ProvenanceDiagnostic",
]
