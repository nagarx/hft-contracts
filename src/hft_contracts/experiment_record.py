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
from typing import Any, Dict, List, Optional

from hft_contracts.atomic_io import atomic_write_json
from hft_contracts.provenance import Provenance


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
INDEX_SCHEMA_VERSION: str = "1.0.0"


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
    """

    TRAINING = "training"
    ANALYSIS = "analysis"
    CALIBRATION = "calibration"
    BACKTEST = "backtest"
    EVALUATION = "evaluation"
    SWEEP_AGGREGATE = "sweep_aggregate"


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


__all__ = [
    "ExperimentRecord",
    "RecordType",
    "INDEX_SCHEMA_VERSION",
]
