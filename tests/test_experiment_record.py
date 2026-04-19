"""Tests for hft_contracts.experiment_record (Phase 6 6B.1a co-move).

Mirrors the contract-level subset of hft-ops tests/test_ledger.py.
Tests that exercise the full ledger writer / dedup / comparison pipeline
stay in hft-ops (they have orchestrator-specific semantics). This file
exercises the dataclass + JSON round-trip + index_entry — all of which
are contract-plane concerns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hft_contracts.experiment_record import ExperimentRecord, RecordType
from hft_contracts.provenance import GitInfo, Provenance


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

    def test_retroactive_surfaced(self):
        r = ExperimentRecord(
            provenance=Provenance(retroactive=True),
        )
        entry = r.index_entry()
        assert entry["retroactive"] is True


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
    ``hft_contracts._atomic_io`` must handle BOTH ``OSError`` (disk
    issues) and non-OSError exceptions (TypeError from non-serializable
    values, KeyboardInterrupt mid-fsync) by cleaning up the tmp file.
    """

    def test_typeerror_cleans_up_tmp(self, tmp_path: Path):
        """json.dump on a non-serializable value raises TypeError. The
        cleanup path must unlink the tmp file — leaking orphans would
        accumulate over many runs.
        """
        from hft_contracts._atomic_io import atomic_write_json

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
        from hft_contracts._atomic_io import atomic_write_json

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
        from hft_contracts._atomic_io import atomic_write_json as canonical
        from hft_ops.feature_sets.writer import (
            atomic_write_json as reexported,
        )
        assert reexported is canonical

        from hft_contracts._atomic_io import AtomicWriteError as canonical_err
        from hft_ops.feature_sets.writer import AtomicWriteError as reexported_err
        assert reexported_err is canonical_err


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

        Phase 7 Stage 7.4 Round 5: patches ``hft_contracts._atomic_io``
        (canonical location) — previously patched ``experiment_record``
        when the helper lived there inline.
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
        from hft_contracts import _atomic_io as io_mod

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


class TestHftOpsShimCompat:
    """Back-compat: pre-6B.1a imports through hft_ops.ledger.experiment_record
    must still work, returning the SAME class (not a copy).
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
