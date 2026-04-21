"""Tests for hft_contracts.experiment_record (Phase 6 6B.1a co-move).

Mirrors the contract-level subset of hft-ops tests/test_ledger.py.
Tests that exercise the full ledger writer / dedup / comparison pipeline
stay in hft-ops (they have orchestrator-specific semantics). This file
exercises the dataclass + JSON round-trip + index_entry — all of which
are contract-plane concerns.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from hft_contracts.experiment_record import (
    ExperimentRecord,
    INDEX_SCHEMA_VERSION,
    RecordType,
)
from hft_contracts.provenance import GitInfo, Provenance


class TestPackageSurface:
    """Phase 8B: lock the `INDEX_SCHEMA_VERSION` constant's presence, type,
    and SemVer format. The constant drives the auto-invalidation substrate
    for `hft-ops/ledger/index.json` — any future typo that drops the constant
    or changes its format silently would re-introduce the silent-omission
    class this phase exists to eliminate.
    """

    def test_index_schema_version_present_and_string(self):
        assert isinstance(INDEX_SCHEMA_VERSION, str), (
            f"INDEX_SCHEMA_VERSION must be str; got {type(INDEX_SCHEMA_VERSION).__name__}"
        )
        assert INDEX_SCHEMA_VERSION, "INDEX_SCHEMA_VERSION must not be empty"

    def test_index_schema_version_is_semver_major_minor_patch(self):
        # Format: MAJOR.MINOR.PATCH — three non-negative integers separated
        # by dots. Phase 8B Step B.2 (next) will parse via
        # packaging.version.Version for MAJOR.MINOR comparison. This test
        # is the stdlib-regex fallback guarding the format invariant.
        import re

        pattern = r"^\d+\.\d+\.\d+$"
        assert re.match(pattern, INDEX_SCHEMA_VERSION), (
            f"INDEX_SCHEMA_VERSION must match MAJOR.MINOR.PATCH regex {pattern!r}; "
            f"got {INDEX_SCHEMA_VERSION!r}. Extending index_entry() whitelist? "
            f"Bump MINOR. Renaming/removing whitelist keys? Bump MAJOR + migration note."
        )

    def test_index_schema_version_reexported_at_package_level(self):
        # The constant is re-exported from `hft_contracts` at package level
        # so downstream consumers (hft-ops ledger envelope writer) can import
        # it without drilling into the submodule.
        import hft_contracts

        assert hasattr(hft_contracts, "INDEX_SCHEMA_VERSION"), (
            "INDEX_SCHEMA_VERSION must be re-exported at the hft_contracts "
            "package level so `from hft_contracts import INDEX_SCHEMA_VERSION` works"
        )
        assert hft_contracts.INDEX_SCHEMA_VERSION == INDEX_SCHEMA_VERSION, (
            "Package-level re-export must be the same string as the submodule export"
        )
        assert "INDEX_SCHEMA_VERSION" in hft_contracts.__all__, (
            "INDEX_SCHEMA_VERSION must appear in hft_contracts.__all__"
        )


class TestIndexEntryCompleteness:
    """Phase 8B Step B.4: whitelist-parity golden. Every key projected by
    ``ExperimentRecord.index_entry()`` must map to a raw-field accessor on
    the source record (not a hand-coded literal or a ``.get(..., default)``
    outside the projection loop).

    This catches the R4-style silent-omission bug pattern where a developer
    adds ``self.training_metrics.get("test_NEW", 0.0)`` outside the
    comprehension loop: the key lands in the projected dict ("test_NEW")
    but it has no raw-field source, so it's always projected as ``0.0``
    for EVERY record, masking the intended extraction of the real value
    from raw ``training_metrics``.

    The check: populate a fixture record with ALL fields set to distinctive
    non-default values. Project via ``index_entry()``. For every key in
    the projection that has a scalar value (not a list/dict of derived
    stats), verify the value matches the raw-record source — i.e. the
    value came FROM the record, not from a literal default.
    """

    def test_every_projected_scalar_key_has_raw_source(self):
        # Populate a fixture record with distinctive, non-default values so
        # we can detect "the projection returns a default instead of the
        # real raw-field value" — the silent-omission pattern.
        fixture = ExperimentRecord(
            experiment_id="FIXTURE_20260420T000000_fixture1",
            name="fixture_record",
            fingerprint="f" * 64,
            contract_version="2.2",
            tags=["fixture", "b4-test"],
            status="completed",
            created_at="2026-04-20T00:00:00+00:00",
            training_metrics={
                # Classification taxonomy (R4 Round 5 fix):
                "test_accuracy": 0.5555,
                "test_macro_f1": 0.4444,
                "test_weighted_f1": 0.3333,
                "test_macro_precision": 0.2222,
                "test_macro_recall": 0.1111,
                # Regression taxonomy:
                "test_ic": 0.3333,
                "test_directional_accuracy": 0.6666,
                "test_r2": 0.1111,
                "test_mae": 0.0555,
                "test_rmse": 0.0666,
                "test_pearson": 0.3555,
                "test_profitable_accuracy": 0.6555,
                # Per-epoch best_val_*:
                "best_val_ic": 0.3777,
                "best_val_directional_accuracy": 0.6777,
                "best_val_r2": 0.1777,
                # Classification best_val_*:
                "best_val_accuracy": 0.7777,
                "best_val_macro_f1": 0.6888,
                # Loss:
                "best_val_loss": 0.8888,
                "best_val_mae": 0.0777,
                "best_val_rmse": 0.0888,
            },
        )

        projection = fixture.index_entry()
        assert isinstance(projection, dict), (
            "index_entry() must return a dict"
        )

        # Every scalar key projected from training_metrics MUST be sourced
        # from the raw training_metrics dict (not a literal default).
        projected_metrics = projection.get("training_metrics", {})
        raw_metrics = fixture.training_metrics

        for key, projected_value in projected_metrics.items():
            # The raw-source invariant: projection MUST match raw-field
            # value. If it doesn't, either the key was hand-literaled into
            # the projection (a bug), or the raw-source has a typo.
            assert key in raw_metrics, (
                f"projected key {key!r} not found in raw training_metrics — "
                f"index_entry() is either hand-literaling a default or the "
                f"whitelist has a typo. Either fix the projection OR add "
                f"{key!r} to the fixture to assert the data-flow is correct."
            )
            assert projected_value == raw_metrics[key], (
                f"projected training_metrics[{key!r}]={projected_value} "
                f"differs from raw training_metrics[{key!r}]={raw_metrics[key]}. "
                f"index_entry() appears to be returning a default instead "
                f"of the raw value — this is the silent-omission class that "
                f"Phase 8B INDEX_SCHEMA_VERSION auto-invalidation guards."
            )

    def test_index_entry_top_level_key_set_frozen(self):
        """Phase 8B Step B.5 (hardcoded-golden variant): the TOP-LEVEL
        key set of ``index_entry()`` projection is frozen. Any addition
        or removal forces this test to be updated AND the developer
        must bump ``INDEX_SCHEMA_VERSION`` per the Change-Coordination
        Checklist discipline.

        This is a deliberate friction gate: changing the projection
        surface should NEVER happen silently. The assertion here makes
        it structurally impossible to extend the whitelist without
        also updating a test that documents the schema at the current
        version.

        Future enhancement: replace the hardcoded set with a JSON
        golden file loaded via ``pytest --regen`` (plan Step B.5's
        fixture-file variant). For v1.0.0 with only one consumer
        (hft-ops ledger) the hardcoded set is sufficient; a fixture
        file becomes valuable when a second consumer of the projection
        surface lands in Phase 8C.
        """
        fixture = ExperimentRecord(
            experiment_id="GOLDEN_20260420T000000_golden01",
            name="golden_record",
            fingerprint="g" * 64,
            contract_version="2.2",
            status="completed",
            created_at="2026-04-20T00:00:00+00:00",
        )
        projection = fixture.index_entry()
        # Frozen top-level key set for INDEX_SCHEMA_VERSION="1.3.0":
        # 1.0.0 → 1.1.0 (Phase 8A.0, 2026-04-20): +cache_info (additive).
        # 1.1.0 → 1.2.0 (Phase 8A.1, 2026-04-20): +sweep_failure_info (additive).
        # 1.2.0 → 1.3.0 (Phase 8C-α C.2, 2026-04-20): +artifact_kinds (additive).
        expected_top_level = {
            "experiment_id",
            "name",
            "fingerprint",
            "contract_version",
            "tags",
            "training_metrics",
            "backtest_metrics",
            "status",
            "stages_completed",
            "duration_seconds",
            "model_type",
            "labeling_strategy",
            "hypothesis",
            "created_at",
            "retroactive",
            "sweep_id",
            "axis_values",
            "record_type",
            "parent_experiment_id",
            "feature_set_ref",
            "gate_reports",
            "cache_info",              # Phase 8A.0 — extraction-cache observability
            "sweep_failure_info",      # Phase 8A.1 — parallel-sweep failure taxonomy
            "artifact_kinds",          # Phase 8C-α C.2 — post-training artifact kinds
            "compatibility_fingerprint",  # Phase V.A.4 — Signal-boundary compatibility trust column
        }
        actual_top_level = set(projection.keys())

        added = actual_top_level - expected_top_level
        removed = expected_top_level - actual_top_level

        assert not added and not removed, (
            f"Phase 8B/8A.0/8A.1/8C-α/V.A.4: index_entry() key-set drifted from "
            f"INDEX_SCHEMA_VERSION=1.4.0 golden. Added: {sorted(added)}; "
            f"Removed: {sorted(removed)}. If this change is intentional, "
            f"(a) bump INDEX_SCHEMA_VERSION MINOR (or MAJOR for removals), "
            f"(b) update this test's expected_top_level set, "
            f"(c) update root CLAUDE.md Change-Coordination Checklist entry "
            f"for index_entry() whitelist extensions. The auto-invalidation "
            f"substrate in hft-ops/ledger/ledger.py::_load_index will then "
            f"re-project all existing records on next load."
        )

    def test_top_level_scalar_keys_traceable(self):
        """Beyond training_metrics, top-level scalar keys projected by
        index_entry() (experiment_id, name, fingerprint, status, etc.) must
        also trace back to raw-field values.
        """
        fixture = ExperimentRecord(
            experiment_id="TRACE_20260420T000000_trace123",
            name="trace_record",
            fingerprint="e" * 64,
            contract_version="2.2",
            tags=["trace"],
            status="completed",
            created_at="2026-04-20T00:00:00+00:00",
        )
        projection = fixture.index_entry()
        # Spot-check the canonical top-level keys — these are the identity
        # fields every consumer (dedup, query, comparison) relies on.
        assert projection["experiment_id"] == fixture.experiment_id
        assert projection["name"] == fixture.name
        assert projection["fingerprint"] == fixture.fingerprint
        assert projection["status"] == fixture.status
        assert projection["created_at"] == fixture.created_at


class TestRecordType:
    def test_enum_values(self):
        assert RecordType.TRAINING.value == "training"
        assert RecordType.ANALYSIS.value == "analysis"
        assert RecordType.CALIBRATION.value == "calibration"
        assert RecordType.BACKTEST.value == "backtest"
        assert RecordType.EVALUATION.value == "evaluation"
        assert RecordType.SWEEP_AGGREGATE.value == "sweep_aggregate"


class TestExperimentRecordDefaults:
    def test_created_at_auto_populates(self):
        r = ExperimentRecord(name="test")
        assert r.created_at != ""
        # ISO 8601
        assert "T" in r.created_at

    def test_empty_defaults(self):
        r = ExperimentRecord()
        assert r.experiment_id == ""
        assert r.name == ""
        assert r.fingerprint == ""
        assert r.feature_set_ref is None
        assert r.record_type == "training"
        assert r.parent_experiment_id == ""


class TestToDictFromDict:
    def test_roundtrip(self):
        r1 = ExperimentRecord(
            experiment_id="E5_2026-03_abc12345",
            name="E5_60s_huber",
            manifest_path="/abs/to/manifest.yaml",
            fingerprint="a" * 64,
            contract_version="2.2",
            training_metrics={"ic": 0.38, "r2": 0.124},
            backtest_metrics={"total_return": -0.0193, "win_rate": 0.401},
            tags=["e5", "regression"],
            status="completed",
            stages_completed=["extract", "train", "backtest"],
            provenance=Provenance(
                git=GitInfo(commit_hash="abc123", branch="main", dirty=False),
                contract_version="2.2",
            ),
        )
        r2 = ExperimentRecord.from_dict(r1.to_dict())
        assert r2.experiment_id == r1.experiment_id
        assert r2.fingerprint == r1.fingerprint
        assert r2.training_metrics == r1.training_metrics
        assert r2.backtest_metrics == r1.backtest_metrics
        assert r2.provenance.git.commit_hash == "abc123"
        assert r2.provenance.contract_version == "2.2"

    def test_roundtrip_with_feature_set_ref(self):
        r1 = ExperimentRecord(
            name="test",
            feature_set_ref={"name": "nvda_98_stable_v1", "content_hash": "a" * 64},
        )
        r2 = ExperimentRecord.from_dict(r1.to_dict())
        assert r2.feature_set_ref == {"name": "nvda_98_stable_v1", "content_hash": "a" * 64}


class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path: Path):
        r1 = ExperimentRecord(
            experiment_id="test_001",
            name="test_run",
            training_metrics={"ic": 0.5},
        )
        p = tmp_path / "record.json"
        r1.save(p)
        r2 = ExperimentRecord.load(p)
        assert r2.experiment_id == "test_001"
        assert r2.training_metrics == {"ic": 0.5}


class TestIndexEntry:
    def test_whitelisted_metrics_only(self):
        r = ExperimentRecord(
            training_metrics={
                "accuracy": 0.6,
                "macro_f1": 0.4,
                "internal_loss": 123.45,  # not whitelisted
            },
            backtest_metrics={
                "total_return": 0.05,
                "win_rate": 0.52,
                "internal_equity_series": [1, 2, 3],  # not whitelisted
            },
        )
        entry = r.index_entry()
        assert entry["training_metrics"] == {"accuracy": 0.6, "macro_f1": 0.4}
        assert entry["backtest_metrics"] == {"total_return": 0.05, "win_rate": 0.52}

    def test_feature_set_ref_in_index(self):
        r = ExperimentRecord(
            feature_set_ref={"name": "nvda_98_stable_v1", "content_hash": "a" * 64},
        )
        entry = r.index_entry()
        assert entry["feature_set_ref"] == {
            "name": "nvda_98_stable_v1",
            "content_hash": "a" * 64,
        }

    def test_feature_set_ref_empty_dict_when_unset(self):
        r = ExperimentRecord(feature_set_ref=None)
        entry = r.index_entry()
        # Empty dict (not None) for consistent schema
        assert entry["feature_set_ref"] == {}

    # -------------------------------------------------------------------------
    # Phase V.A.4 (2026-04-21): compatibility_fingerprint field + projection
    # -------------------------------------------------------------------------

    def test_compatibility_fingerprint_defaults_to_none(self):
        """Dataclass default matches other optional reference fields."""
        r = ExperimentRecord()
        assert r.compatibility_fingerprint is None

    def test_compatibility_fingerprint_round_trip(self):
        """Valid 64-hex fingerprint survives to_dict → from_dict unchanged."""
        hex_fp = "a" * 64
        r = ExperimentRecord(compatibility_fingerprint=hex_fp)
        r2 = ExperimentRecord.from_dict(r.to_dict())
        assert r2.compatibility_fingerprint == hex_fp

    def test_compatibility_fingerprint_in_index_valid(self):
        """Valid 64-hex fingerprint projects as-is into index_entry."""
        hex_fp = "b" * 64
        r = ExperimentRecord(compatibility_fingerprint=hex_fp)
        entry = r.index_entry()
        assert entry["compatibility_fingerprint"] == hex_fp

    def test_compatibility_fingerprint_empty_string_when_unset(self):
        """None default → empty string "" in index (JSON schema consistency)."""
        r = ExperimentRecord(compatibility_fingerprint=None)
        entry = r.index_entry()
        assert entry["compatibility_fingerprint"] == ""

    def test_compatibility_fingerprint_malformed_coerced_to_empty(self):
        """Non-64-hex values silently coerce to "" — graceful degradation
        for poisoned records; matches the feature_set_ref.content_hash
        gate pattern. Producer (hft-ops harvester) already validates via
        CONTENT_HASH_RE, but this is defense-in-depth for records written
        by non-standard paths (test fixtures, manual ledger edits, etc.)."""
        for bad in [
            "not-hex",                         # too short + invalid chars
            "abc",                              # too short
            "A" * 64,                           # uppercase — CONTENT_HASH_RE is lowercase-only
            "a" * 63,                           # 63 chars, off-by-one
            "a" * 65,                           # 65 chars, off-by-one
            "gg" + "a" * 62,                    # non-hex chars
            "",                                 # empty string
        ]:
            r = ExperimentRecord(compatibility_fingerprint=bad)
            entry = r.index_entry()
            assert entry["compatibility_fingerprint"] == "", (
                f"Malformed fingerprint {bad!r} should coerce to empty "
                f"string; got {entry['compatibility_fingerprint']!r}"
            )

    def test_index_schema_version_bumped_to_1_4_0(self):
        """Phase V.A.4 adds compatibility_fingerprint to the index projection —
        bumps INDEX_SCHEMA_VERSION MINOR 1.3.0 → 1.4.0 per SemVer additive
        policy (root CLAUDE.md §Change-Coordination Checklist). hft-ops ledger
        envelope auto-rebuild triggers on MAJOR.MINOR mismatch."""
        from hft_contracts.experiment_record import INDEX_SCHEMA_VERSION

        assert INDEX_SCHEMA_VERSION == "1.4.0", (
            f"Expected INDEX_SCHEMA_VERSION='1.4.0' (Phase V.A.4 bump); "
            f"got {INDEX_SCHEMA_VERSION!r}. If intentional, update this "
            f"test + root CLAUDE.md Last-verified stamp + "
            f"pipeline_contract.toml changelog entry."
        )

    def test_retroactive_surfaced(self):
        r = ExperimentRecord(
            provenance=Provenance(retroactive=True),
        )
        entry = r.index_entry()
        assert entry["retroactive"] is True

    # -------------------------------------------------------------------------
    # Phase V.1 L1.2 (2026-04-21): signal_export_output_dir field (Agent 2 H1
    # manifest-move-resilience fix). Not projected into index_entry() —
    # record-level only, accessed via ledger.get(exp_id).
    # -------------------------------------------------------------------------

    def test_signal_export_output_dir_defaults_to_none(self):
        """Dataclass default: absent signal_export OR no output_dir set →
        None. Matches the pattern of other optional reference fields."""
        r = ExperimentRecord()
        assert r.signal_export_output_dir is None

    def test_signal_export_output_dir_round_trip(self):
        """Absolute-path string survives to_dict → from_dict unchanged."""
        path_str = "/Users/knight/code_local/HFT-pipeline-v2/outputs/experiments/foo/signals/test"
        r = ExperimentRecord(signal_export_output_dir=path_str)
        r2 = ExperimentRecord.from_dict(r.to_dict())
        assert r2.signal_export_output_dir == path_str

    def test_signal_export_output_dir_not_in_index(self):
        """Record-level field only — NOT projected into index_entry()
        (path is an implementation detail, not a user-facing filter axis).
        Locks the architectural decision that consumers access via the
        full record (ledger.get), not the lightweight index."""
        r = ExperimentRecord(signal_export_output_dir="/some/path")
        entry = r.index_entry()
        assert "signal_export_output_dir" not in entry, (
            f"signal_export_output_dir should NOT be in index_entry(); "
            f"found in: {list(entry.keys())}"
        )

    def test_signal_export_output_dir_backward_compat_absent_key(self):
        """Pre-V.1.L1.2 records saved without the field deserialize
        gracefully with signal_export_output_dir=None."""
        data = ExperimentRecord(name="legacy_experiment").to_dict()
        # Simulate a pre-V.1.L1.2 JSON record by stripping the field
        data.pop("signal_export_output_dir", None)
        r = ExperimentRecord.from_dict(data)
        assert r.signal_export_output_dir is None


class TestClassificationTestMetricsWhitelist:
    """Phase 7 Stage 7.4 Round 5 item C1 (2026-04-20): whitelist must
    surface ``test_accuracy`` / ``test_macro_f1`` / ``test_macro_precision``
    / ``test_macro_recall`` / ``test_loss``.

    Round 4 shipped ``scripts/train.py::_dump_test_metrics`` with
    unconditional ``test_`` prefix. ClassificationMetrics.to_dict()
    returns {accuracy, macro_f1, macro_precision, macro_recall, loss,
    ...}; prefixed → {test_accuracy, test_macro_f1, ...}. Without this
    whitelist expansion, those keys silently vanished from the ledger
    index for every PyTorch TLOB / HMHP / LogisticLOB / DeepLOB run —
    identical failure mode to the Round 1 regression gap.
    """

    def test_classification_test_keys_surface(self):
        r = ExperimentRecord(
            training_metrics={
                "test_accuracy": 0.596,
                "test_macro_f1": 0.421,
                "test_macro_precision": 0.450,
                "test_macro_recall": 0.398,
                "test_loss": 1.024,
            },
        )
        entry = r.index_entry()
        assert entry["training_metrics"] == {
            "test_accuracy": 0.596,
            "test_macro_f1": 0.421,
            "test_macro_precision": 0.450,
            "test_macro_recall": 0.398,
            "test_loss": 1.024,
        }

    def test_per_class_and_strategy_metrics_NOT_surfaced(self):
        """Only core scalars are whitelisted — per-class precision /
        recall / F1 + strategy metrics are dropped to prevent index
        bloat (matches regression convention that drops per-horizon
        keys like test_h10_ic).
        """
        r = ExperimentRecord(
            training_metrics={
                "test_accuracy": 0.596,  # whitelisted
                "test_up_precision": 0.500,  # NOT whitelisted
                "test_down_recall": 0.450,  # NOT whitelisted
                "test_class_0_f1": 0.400,  # NOT whitelisted
                "test_signal_rate": 0.600,  # NOT whitelisted (not a scalar "core" metric)
            },
        )
        entry = r.index_entry()
        assert entry["training_metrics"] == {"test_accuracy": 0.596}


class TestGateReportsField:
    """Phase 7 Stage 7.4 Round 4 (2026-04-20): generic ``gate_reports``
    field replaces the Round 1 nested-under-training_metrics pattern.
    """

    def test_default_is_empty_dict(self):
        r = ExperimentRecord()
        assert r.gate_reports == {}

    def test_accepts_multi_stage_reports(self):
        r = ExperimentRecord(
            gate_reports={
                "validation": {
                    "verdict": "PASS",
                    "best_ic": 0.15,
                    "reason": "",
                },
                "post_training_gate": {
                    "status": "pass",
                    "primary_metric_name": "test_ic",
                    "primary_metric_value": 0.38,
                },
            },
        )
        assert r.gate_reports["validation"]["verdict"] == "PASS"
        assert r.gate_reports["post_training_gate"]["status"] == "pass"

    def test_roundtrip_preserves_gate_reports(self):
        r1 = ExperimentRecord(
            name="test",
            gate_reports={
                "post_training_gate": {
                    "status": "warn",
                    "primary_metric_value": 0.02,
                    "checks": [{"name": "floor", "status": "fail"}],
                },
            },
        )
        r2 = ExperimentRecord.from_dict(r1.to_dict())
        assert r2.gate_reports == r1.gate_reports


class TestGateReportsIndexEntry:
    """``index_entry()`` projects gate status + truncated summary per stage
    so `hft-ops ledger list` can filter by gate outcome without loading
    the full record.
    """

    def test_surfaces_status_field(self):
        r = ExperimentRecord(
            gate_reports={
                "post_training_gate": {
                    "status": "warn",
                    "summary": "test_ic=0.02 below floor 0.05",
                },
            },
        )
        entry = r.index_entry()
        assert entry["gate_reports"] == {
            "post_training_gate": {
                "status": "warn",
                "summary": "test_ic=0.02 below floor 0.05",
            },
        }

    def test_validation_gate_emits_status_natively_post_round5(self):
        """Phase 7 Stage 7.4 Round 5 (2026-04-20): the validation
        stage adapter (``hft_ops/stages/validation.py``) now injects
        a ``status`` field (lowercased from ``verdict``) before
        writing to ``captured_metrics``. The ``index_entry`` projection
        reads ``status`` directly — no more verdict coalesce.

        Legacy records written before the Round 5 adapter change may
        have ``verdict`` but no ``status``; those fall through the
        ``status: ""`` default. Still queryable via ``ledger show``
        (full record body) even if not surfaced in the index.
        """
        # Round 5+ shape: status is present (injected by adapter)
        r = ExperimentRecord(
            gate_reports={
                "validation": {
                    "verdict": "PASS",  # preserved for fast_gate consumers
                    "status": "pass",   # Round 5 adapter injection
                    "summary": "IC gate PASS",
                },
            },
        )
        entry = r.index_entry()
        assert entry["gate_reports"]["validation"]["status"] == "pass"

    def test_legacy_validation_report_without_status_falls_through(self):
        """Backward-compat: records written before the Round 5 adapter
        change (verdict-only, no status) surface an empty status in
        the index. The full report is still on disk for ``ledger show``.
        """
        r = ExperimentRecord(
            gate_reports={
                "validation": {
                    "verdict": "PASS",  # Round 4 era — no status field
                    "summary": "IC gate PASS",
                },
            },
        )
        entry = r.index_entry()
        assert entry["gate_reports"]["validation"]["status"] == ""

    def test_truncates_long_summaries(self):
        long_summary = "x" * 500
        r = ExperimentRecord(
            gate_reports={
                "post_training_gate": {
                    "status": "pass",
                    "summary": long_summary,
                },
            },
        )
        entry = r.index_entry()
        # Truncated to 256 chars to prevent index.json bloat
        assert len(entry["gate_reports"]["post_training_gate"]["summary"]) == 256

    def test_empty_gate_reports_is_empty_dict_in_index(self):
        r = ExperimentRecord()
        entry = r.index_entry()
        assert entry["gate_reports"] == {}

    def test_missing_status_or_verdict_yields_empty_string(self):
        r = ExperimentRecord(
            gate_reports={
                "post_training_gate": {
                    # neither status nor verdict present — malformed
                    "primary_metric_value": 0.38,
                },
            },
        )
        entry = r.index_entry()
        assert entry["gate_reports"]["post_training_gate"]["status"] == ""


class TestLegacyNestedGateMigration:
    """Phase 7 Stage 7.4 Round 4 migration shim: records written
    between Round 1 (2026-04-19) and Round 4 (2026-04-20) nested
    ``post_training_gate`` under ``training_metrics``. ``from_dict``
    lifts it to the new ``gate_reports`` field and drops the redundant
    ``post_training_gate_summary``. Removal deadline 2026-08-01.
    """

    def test_legacy_nested_gate_is_migrated(self):
        legacy_gate_data = {
            "status": "pass",
            "primary_metric_name": "test_ic",
            "primary_metric_value": 0.38,
        }
        legacy_payload = {
            "name": "legacy_run",
            "training_metrics": {
                "test_ic": 0.38,
                "post_training_gate": legacy_gate_data,
                "post_training_gate_summary": "post_training_gate: PASS",
            },
        }
        r = ExperimentRecord.from_dict(legacy_payload)

        # Lifted to new field
        assert r.gate_reports == {"post_training_gate": legacy_gate_data}

        # Stripped from training_metrics
        assert "post_training_gate" not in r.training_metrics
        assert "post_training_gate_summary" not in r.training_metrics
        # But actual metric keys preserved
        assert r.training_metrics == {"test_ic": 0.38}

    def test_migration_does_not_override_new_field(self):
        # If a record has BOTH the legacy nested value AND a new-shape
        # gate_reports entry (impossible in practice but defensive),
        # the new-shape value wins — setdefault leaves it alone.
        modern_gate_data = {"status": "warn"}
        legacy_gate_data = {"status": "pass"}
        payload = {
            "gate_reports": {"post_training_gate": modern_gate_data},
            "training_metrics": {
                "post_training_gate": legacy_gate_data,
            },
        }
        r = ExperimentRecord.from_dict(payload)
        # setdefault preserves the new field's value, not the legacy nested one.
        assert r.gate_reports["post_training_gate"]["status"] == "warn"

    def test_pre_round1_records_load_cleanly(self):
        # Records written BEFORE Round 1 (2026-04-19) had no gate fields
        # at all. Must load without error.
        payload = {
            "name": "pre_round1",
            "training_metrics": {"accuracy": 0.55},
        }
        r = ExperimentRecord.from_dict(payload)
        assert r.gate_reports == {}
        assert r.training_metrics == {"accuracy": 0.55}

    def test_non_dict_legacy_value_is_not_migrated(self):
        # Defensive: if someone manually mutilated training_metrics,
        # the migration must not crash.
        payload = {
            "training_metrics": {
                "post_training_gate": "bad-scalar-not-a-dict",
            },
        }
        r = ExperimentRecord.from_dict(payload)
        assert r.gate_reports == {}
        # Non-dict value still popped to prevent it contaminating metrics.
        assert "post_training_gate" not in r.training_metrics

    def test_from_dict_does_not_mutate_caller_input(self):
        """Phase 7 Stage 7.4 Round 5 (2026-04-20): migration shim must
        not mutate the caller's ``data`` dict. Previously the shim
        pop()'d from ``record.training_metrics``, which was the SAME
        reference as ``data["training_metrics"]`` (dataclass field
        passes by reference). Now shallow-copies before mutating.
        """
        legacy_gate_data = {"status": "pass", "primary_metric_value": 0.38}
        legacy_payload = {
            "name": "legacy_noimutate",
            "training_metrics": {
                "test_ic": 0.38,
                "post_training_gate": legacy_gate_data,
                "post_training_gate_summary": "PASS",
            },
        }

        # Snapshot input state BEFORE from_dict
        input_training_metrics_before = dict(legacy_payload["training_metrics"])

        r = ExperimentRecord.from_dict(legacy_payload)

        # Record has migrated shape
        assert r.gate_reports == {"post_training_gate": legacy_gate_data}
        assert "post_training_gate" not in r.training_metrics

        # Caller's input dict MUST be untouched
        assert legacy_payload["training_metrics"] == input_training_metrics_before, (
            "from_dict mutated caller's training_metrics dict — "
            "migration shim must shallow-copy before popping"
        )
        assert "post_training_gate" in legacy_payload["training_metrics"]
        assert "post_training_gate_summary" in legacy_payload["training_metrics"]

        # Round-trip stability: loading the same payload twice must
        # produce two records with identical shape (no stale state).
        r2 = ExperimentRecord.from_dict(legacy_payload)
        assert r2.gate_reports == r.gate_reports
        assert "post_training_gate" not in r2.training_metrics


class TestGateReportsFingerprintStability:
    """INVARIANT: ``gate_reports`` content MUST NOT affect identity.

    ``compute_fingerprint`` (hft_ops.ledger.dedup) only hashes the
    resolved trainer config. A gate is an observation, not a
    treatment — different gate outcomes on the same config must
    produce the same fingerprint.

    Here we exercise the contract-plane-side invariant: two records
    with identical everything EXCEPT ``gate_reports`` must serialize
    to identical byte strings if ``gate_reports`` is excluded. This is
    a tautology at the dataclass level, but codifies the rule so any
    future attempt to fold gate state into fingerprint-generation
    surfaces here.
    """

    def test_gate_reports_not_in_hash_critical_surface(self):
        from hft_contracts.canonical_hash import canonical_json_blob, sha256_hex

        r1 = ExperimentRecord(
            name="e1",
            fingerprint="f" * 64,
            training_config={"model": {"type": "tlob"}},
            gate_reports={"post_training_gate": {"status": "pass"}},
        )
        r2 = ExperimentRecord(
            name="e1",
            fingerprint="f" * 64,
            training_config={"model": {"type": "tlob"}},
            gate_reports={"post_training_gate": {"status": "warn"}},
        )

        # "Hash-critical surface" = identity fields used by dedup:
        #   training_config + extraction_config + contract_version.
        # Both records project to the same hash input.
        def _identity(rec):
            return {
                "training_config": rec.training_config,
                "extraction_config": rec.extraction_config,
                "contract_version": rec.contract_version,
            }

        assert sha256_hex(canonical_json_blob(_identity(r1))) == sha256_hex(
            canonical_json_blob(_identity(r2)),
        )


class TestAtomicWriteJsonCrashSafety:
    """Phase 7 Stage 7.4 Round 5 (2026-04-20): ``atomic_write_json`` in
    ``hft_contracts.atomic_io`` must handle BOTH ``OSError`` (disk
    issues) and non-OSError exceptions (TypeError from non-serializable
    values, KeyboardInterrupt mid-fsync) by cleaning up the tmp file.

    REV 2 pre-push (2026-04-20): tests now import from the canonical
    public ``hft_contracts.atomic_io``. Shim-identity regression test
    (deprecated ``_atomic_io`` still resolves to the same functions) is
    in ``TestAtomicIoShimCompat`` below.
    """

    def test_typeerror_cleans_up_tmp(self, tmp_path: Path):
        """json.dump on a non-serializable value raises TypeError. The
        cleanup path must unlink the tmp file — leaking orphans would
        accumulate over many runs.
        """
        from hft_contracts.atomic_io import atomic_write_json

        class NotSerializable:
            """default=str produces a string, but a circular list inside raises."""

        bad = {"circular": None}
        bad["circular"] = bad  # Force circular reference → RecursionError /
                               # ValueError from json.dump

        target = tmp_path / "bad.json"
        with pytest.raises((ValueError, RecursionError, TypeError)):
            atomic_write_json(target, bad)

        # Target was never created
        assert not target.exists()

        # No orphan tmp files left behind
        leftover = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
        assert leftover == [], f"tmp files leaked on TypeError: {leftover}"

    def test_canonical_convention_sort_keys_and_trailing_newline(
        self, tmp_path: Path,
    ):
        """Canonical convention: sort_keys=True + trailing newline.

        Locks the decision that Round 5 unified all atomic writes on
        this form (diff-stable + POSIX text-file convention).
        """
        from hft_contracts.atomic_io import atomic_write_json

        target = tmp_path / "canonical.json"
        atomic_write_json(target, {"z": 3, "a": 1, "m": 2})

        content = target.read_text()
        # Keys in sorted order
        assert content.index('"a"') < content.index('"m"') < content.index('"z"')
        # Trailing newline
        assert content.endswith("\n")

    def test_hft_ops_reexport_is_identical(self):
        """``hft_ops.feature_sets.writer.atomic_write_json`` re-exports
        the canonical hft_contracts implementation — not a copy.
        """
        hft_ops_writer = pytest.importorskip("hft_ops.feature_sets.writer")
        from hft_contracts.atomic_io import atomic_write_json as canonical
        assert hft_ops_writer.atomic_write_json is canonical

        from hft_contracts.atomic_io import AtomicWriteError as canonical_err
        assert hft_ops_writer.AtomicWriteError is canonical_err


class TestAtomicIoShimCompat:
    """REV 2 pre-push hygiene regression (2026-04-20): the renamed module
    ``hft_contracts._atomic_io`` → ``hft_contracts.atomic_io`` keeps the
    underscore-prefix name as a deprecation shim until 2026-10-31.

    These tests lock:

    - Shim resolves attribute access to the SAME objects as the canonical
      module (no copies, no divergence).
    - Shim emits ``DeprecationWarning`` on first access per symbol.
    - Non-public attribute access raises ``AttributeError`` (shim does
      not accidentally forward arbitrary names).
    """

    def test_shim_resolves_same_objects_as_canonical(self):
        """``hft_contracts._atomic_io.atomic_write_json`` IS
        ``hft_contracts.atomic_io.atomic_write_json`` (same function)."""
        import warnings

        import hft_contracts._atomic_io as shim
        import hft_contracts.atomic_io as canonical

        with warnings.catch_warnings():
            # Shim access may emit DeprecationWarning — we test that
            # separately in test_shim_emits_deprecation_warning.
            warnings.simplefilter("ignore", DeprecationWarning)
            assert shim.atomic_write_json is canonical.atomic_write_json
            assert shim.AtomicWriteError is canonical.AtomicWriteError

    def test_shim_emits_deprecation_warning_on_first_access(self):
        """Accessing a symbol via the shim triggers DeprecationWarning
        with the canonical migration path in the message.
        """
        import importlib
        import warnings

        # Re-import to reset the per-process _WARNED set — tests may run in
        # any order and another test may have already consumed the warning.
        import hft_contracts._atomic_io as shim
        importlib.reload(shim)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            _ = shim.atomic_write_json  # triggers __getattr__

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1, (
            "F2 regression: hft_contracts._atomic_io.atomic_write_json access "
            "did NOT emit DeprecationWarning. Shim telemetry broken — "
            "consumers will silently continue importing from the deprecated "
            "path past the 2026-10-31 removal date."
        )
        msg = str(dep_warnings[0].message)
        assert "hft_contracts.atomic_io" in msg
        assert "2026-10-31" in msg

    def test_shim_rejects_non_public_attributes(self):
        """Shim should NOT transparently forward arbitrary module internals
        — only the publicly-contracted names (atomic_write_json,
        AtomicWriteError). Reaching into shim.os etc. must raise
        AttributeError so callers don't accidentally couple to
        implementation details through the deprecated path.
        """
        import hft_contracts._atomic_io as shim

        with pytest.raises(AttributeError):
            _ = shim.os  # os is a stdlib import inside the canonical module,
                         # but must NOT be exposed through the shim


class TestAtomicSave:
    """Phase 7 Stage 7.4 Round 4 (2026-04-20): ``ExperimentRecord.save``
    is atomic. Crash-safety verified at the ``os.replace`` boundary.
    """

    def test_save_produces_complete_file(self, tmp_path: Path):
        r = ExperimentRecord(
            experiment_id="atomic_test",
            name="atomic",
            training_metrics={"test_ic": 0.38},
        )
        save_path = tmp_path / "records" / "atomic_test.json"
        r.save(save_path)

        # File exists + round-trips
        assert save_path.exists()
        loaded = ExperimentRecord.load(save_path)
        assert loaded.experiment_id == "atomic_test"
        assert loaded.training_metrics["test_ic"] == 0.38

    def test_save_leaves_no_tmp_files_on_success(self, tmp_path: Path):
        r = ExperimentRecord(experiment_id="no_leak_test", name="clean")
        save_path = tmp_path / "records" / "no_leak_test.json"
        r.save(save_path)

        # No residual .tmp files in the parent directory.
        leftover = [p for p in save_path.parent.iterdir() if ".tmp." in p.name]
        assert leftover == []

    def test_save_creates_parent_directory(self, tmp_path: Path):
        r = ExperimentRecord(experiment_id="mkdirp", name="x")
        deep_path = tmp_path / "records" / "nested" / "deep" / "mkdirp.json"
        # Parent does not exist yet.
        assert not deep_path.parent.exists()
        r.save(deep_path)
        assert deep_path.exists()

    def test_save_overwrites_existing_file_atomically(self, tmp_path: Path):
        save_path = tmp_path / "records" / "overwrite.json"
        r1 = ExperimentRecord(
            experiment_id="overwrite",
            training_metrics={"test_ic": 0.30},
        )
        r1.save(save_path)

        r2 = ExperimentRecord(
            experiment_id="overwrite",
            training_metrics={"test_ic": 0.38},
        )
        r2.save(save_path)

        loaded = ExperimentRecord.load(save_path)
        assert loaded.training_metrics["test_ic"] == 0.38

    def test_interrupted_write_leaves_original_intact(
        self, tmp_path: Path, monkeypatch,
    ):
        """Simulate a crash mid-write: ``os.replace`` never runs, so the
        original target file (if any) must remain untouched and the
        tmp file must be cleaned up.

        Phase 7 Stage 7.4 Round 5: patches ``hft_contracts.atomic_io``
        (canonical location) — previously patched ``experiment_record``
        when the helper lived there inline.

        REV 2 pre-push (2026-04-20): canonical module renamed from
        ``_atomic_io`` to the public ``atomic_io``.
        """
        save_path = tmp_path / "records" / "crash_safe.json"
        r1 = ExperimentRecord(
            experiment_id="crash_safe",
            training_metrics={"test_ic": 0.30},
        )
        r1.save(save_path)
        original_content = save_path.read_text()

        # Monkeypatch os.replace in the canonical atomic-io module —
        # simulates crash after fsync but before the atomic swap.
        from hft_contracts import atomic_io as io_mod

        def _fail_replace(src, dst):
            # Clean up the tmp ourselves to keep the test isolated.
            Path(src).unlink(missing_ok=True)
            raise OSError("simulated crash")

        monkeypatch.setattr(io_mod.os, "replace", _fail_replace)

        r2 = ExperimentRecord(
            experiment_id="crash_safe",
            training_metrics={"test_ic": 0.38},
        )
        with pytest.raises(OSError, match="simulated crash"):
            r2.save(save_path)

        # Original file unchanged
        assert save_path.read_text() == original_content
        loaded = ExperimentRecord.load(save_path)
        assert loaded.training_metrics["test_ic"] == 0.30

        # Clean up tmp files (monkeypatched replacement unlinks src; verify).
        leftover = [p for p in save_path.parent.iterdir() if ".tmp." in p.name]
        assert leftover == []


@pytest.mark.skipif(
    importlib.util.find_spec("hft_ops") is None,
    reason=(
        "hft_ops not installed — shim-parity regression guard skipped on "
        "fresh-clone installs of hft-contracts. Runs in authoring env."
    ),
)
class TestHftOpsShimCompat:
    """Back-compat: pre-6B.1a imports through hft_ops.ledger.experiment_record
    must still work, returning the SAME class (not a copy).

    REV 2 pre-push (2026-04-20): class-level skip marker so fresh-clone
    users (who install only hft-contracts) get `pytest -q` → all tests
    SKIP gracefully rather than ERROR with ModuleNotFoundError. Running
    `pip install hft-ops` re-enables these guards automatically.
    """

    def test_hft_ops_reexport_is_identical_class(self):
        from hft_ops.ledger.experiment_record import ExperimentRecord as FromHftOps
        from hft_ops.ledger.experiment_record import RecordType as FromHftOpsRT

        assert FromHftOps is ExperimentRecord
        assert FromHftOpsRT is RecordType

    def test_hft_ops_ledger_top_level_reexport(self):
        """ledger/__init__.py re-exports ExperimentRecord + Provenance."""
        from hft_ops.ledger import ExperimentRecord as FromLedger

        assert FromLedger is ExperimentRecord
