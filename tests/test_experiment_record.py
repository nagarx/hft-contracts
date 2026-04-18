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
