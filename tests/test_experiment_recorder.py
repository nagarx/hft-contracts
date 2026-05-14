"""Tests for ``hft_contracts.experiment_recorder`` SSoT module.

Phase 8D / #PY-223 (2026-05-14): closes the R-17a-class direct-trainer
~26% invisibility class via SSoT helper consumed by BOTH hft-ops
``cli.py::_record_experiment`` AND lob-model-trainer
``scripts/train.py --register-to-ledger``.

Coverage:

* :class:`HarvestedTrustColumns` defaults + dataclass round-trip.
* :func:`harvest_trust_columns` — all 4 fields harvested + fail-loud
  on invalid format + graceful absent.
* :func:`harvest_trust_columns_from_signal_metadata` — file I/O paths
  (missing / malformed / non-dict / happy).
* :func:`record_from_artifacts` — mutually-exclusive raises, Phase Y
  composer success/graceful-None/require_complete raises, ledger I/O
  (atomic round-trip), experiment_id_override, model_config_hash
  injection into training_config.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from hft_contracts.experiment_record import ExperimentRecord
from hft_contracts.experiment_recorder import (
    HarvestedTrustColumns,
    harvest_trust_columns,
    harvest_trust_columns_from_signal_metadata,
    record_from_artifacts,
)


VALID_HASH = "a" * 64
ANOTHER_VALID_HASH = "b" * 64
THIRD_VALID_HASH = "c" * 64
FOURTH_VALID_HASH = "d" * 64


# ---------------------------------------------------------------------------
# HarvestedTrustColumns defaults
# ---------------------------------------------------------------------------


class TestHarvestedTrustColumnsDefaults:
    """Locks the dataclass defaults — all None + empty harvest_errors."""

    def test_defaults_all_none(self):
        h = HarvestedTrustColumns()
        assert h.feature_set_ref is None
        assert h.compatibility_fingerprint is None
        assert h.model_config_hash is None
        assert h.signal_export_output_dir is None
        assert h.harvest_errors == []

    def test_construct_with_fields(self):
        h = HarvestedTrustColumns(
            feature_set_ref={"name": "foo", "content_hash": VALID_HASH},
            compatibility_fingerprint=ANOTHER_VALID_HASH,
        )
        assert h.feature_set_ref == {"name": "foo", "content_hash": VALID_HASH}
        assert h.compatibility_fingerprint == ANOTHER_VALID_HASH

    def test_harvest_errors_independent_per_instance(self):
        """Regression: mutable default factory must not share state."""
        a = HarvestedTrustColumns()
        b = HarvestedTrustColumns()
        a.harvest_errors.append("error from a")
        assert b.harvest_errors == []


# ---------------------------------------------------------------------------
# harvest_trust_columns — all 4 trust columns
# ---------------------------------------------------------------------------


class TestHarvestTrustColumnsHappy:
    """Happy paths: all 4 fields valid → populated, no errors."""

    def test_all_four_fields_valid(self):
        captured = {
            "feature_set_ref": {"name": "nvda_v1", "content_hash": VALID_HASH},
            "compatibility_fingerprint": ANOTHER_VALID_HASH,
            "model_config_hash": THIRD_VALID_HASH,
            "signal_export_output_dir": "/abs/path/to/signals",
        }
        result = harvest_trust_columns(captured)
        assert result.feature_set_ref == {"name": "nvda_v1", "content_hash": VALID_HASH}
        assert result.compatibility_fingerprint == ANOTHER_VALID_HASH
        assert result.model_config_hash == THIRD_VALID_HASH
        assert result.signal_export_output_dir == "/abs/path/to/signals"
        assert result.harvest_errors == []

    def test_empty_captured_metrics(self):
        result = harvest_trust_columns({})
        assert result.feature_set_ref is None
        assert result.compatibility_fingerprint is None
        assert result.model_config_hash is None
        assert result.signal_export_output_dir is None
        assert result.harvest_errors == []

    def test_partial_fields_only(self):
        result = harvest_trust_columns({
            "compatibility_fingerprint": ANOTHER_VALID_HASH,
        })
        assert result.compatibility_fingerprint == ANOTHER_VALID_HASH
        assert result.feature_set_ref is None
        assert result.harvest_errors == []


class TestHarvestTrustColumnsInvalid:
    """Invalid input formats → field stays None, error appended to list."""

    def test_feature_set_ref_non_dict(self):
        result = harvest_trust_columns({"feature_set_ref": "not_a_dict"})
        assert result.feature_set_ref is None
        assert len(result.harvest_errors) == 1
        assert "feature_set_ref not a dict" in result.harvest_errors[0]

    def test_feature_set_ref_missing_keys(self):
        result = harvest_trust_columns({
            "feature_set_ref": {"name": "foo"},  # missing content_hash
        })
        assert result.feature_set_ref is None
        assert len(result.harvest_errors) == 1

    def test_feature_set_ref_wrong_value_types(self):
        result = harvest_trust_columns({
            "feature_set_ref": {"name": 123, "content_hash": VALID_HASH},
        })
        assert result.feature_set_ref is None
        assert len(result.harvest_errors) == 1

    def test_compatibility_fingerprint_uppercase_rejected(self):
        """CONTENT_HASH_RE requires lowercase — fail-loud on uppercase."""
        result = harvest_trust_columns({
            "compatibility_fingerprint": VALID_HASH.upper(),
        })
        assert result.compatibility_fingerprint is None
        assert len(result.harvest_errors) == 1

    def test_compatibility_fingerprint_wrong_length(self):
        result = harvest_trust_columns({
            "compatibility_fingerprint": "abc123",  # too short
        })
        assert result.compatibility_fingerprint is None
        assert len(result.harvest_errors) == 1

    def test_model_config_hash_non_string(self):
        result = harvest_trust_columns({"model_config_hash": 12345})
        assert result.model_config_hash is None
        assert len(result.harvest_errors) == 1

    def test_signal_export_output_dir_empty_string(self):
        result = harvest_trust_columns({"signal_export_output_dir": ""})
        assert result.signal_export_output_dir is None
        assert len(result.harvest_errors) == 1


# ---------------------------------------------------------------------------
# harvest_trust_columns_from_signal_metadata — file I/O paths
# ---------------------------------------------------------------------------


class TestHarvestFromSignalMetadata:
    """File-based harvest helper for direct-trainer path."""

    def test_file_missing(self, tmp_path: Path):
        missing = tmp_path / "nonexistent.json"
        result = harvest_trust_columns_from_signal_metadata(missing)
        assert result.feature_set_ref is None
        assert len(result.harvest_errors) == 1
        assert "not found" in result.harvest_errors[0]

    def test_malformed_json(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        result = harvest_trust_columns_from_signal_metadata(bad)
        assert result.feature_set_ref is None
        assert len(result.harvest_errors) == 1
        assert "failed to read" in result.harvest_errors[0]

    def test_root_not_dict(self, tmp_path: Path):
        list_root = tmp_path / "list.json"
        list_root.write_text('["array", "root"]')
        result = harvest_trust_columns_from_signal_metadata(list_root)
        assert result.feature_set_ref is None
        assert len(result.harvest_errors) == 1
        assert "root not a dict" in result.harvest_errors[0]

    def test_happy_path_post_phase_y(self, tmp_path: Path):
        """Phase Y-era signal_metadata with all trust columns present."""
        metadata = {
            "model_type": "tlob_regression",
            "feature_set_ref": {"name": "nvda_v1", "content_hash": VALID_HASH},
            "compatibility_fingerprint": ANOTHER_VALID_HASH,
            "model_config_hash": THIRD_VALID_HASH,
        }
        sig_meta = tmp_path / "signal_metadata.json"
        sig_meta.write_text(json.dumps(metadata))
        result = harvest_trust_columns_from_signal_metadata(sig_meta)
        assert result.feature_set_ref == {"name": "nvda_v1", "content_hash": VALID_HASH}
        assert result.compatibility_fingerprint == ANOTHER_VALID_HASH
        assert result.model_config_hash == THIRD_VALID_HASH
        # signal_export_output_dir NOT in producer schema — not harvested from file.
        assert result.signal_export_output_dir is None
        assert result.harvest_errors == []

    def test_legacy_pre_phase_y_metadata(self, tmp_path: Path):
        """Pre-Phase-Y signal_metadata without trust columns → all None, no errors."""
        legacy = {
            "model_type": "tlob_regression",
            "horizon_idx": 0,
            "metrics": {"ic": 0.7},
        }
        sig_meta = tmp_path / "signal_metadata.json"
        sig_meta.write_text(json.dumps(legacy))
        result = harvest_trust_columns_from_signal_metadata(sig_meta)
        assert result.feature_set_ref is None
        assert result.compatibility_fingerprint is None
        assert result.model_config_hash is None
        # Critical: absent keys are NOT errors (graceful for legacy artifacts).
        assert result.harvest_errors == []


# ---------------------------------------------------------------------------
# record_from_artifacts — core composition
# ---------------------------------------------------------------------------


class TestRecordFromArtifactsBasic:
    """Minimal direct-trainer-style invocations."""

    def test_minimal_required_only(self, tmp_path: Path):
        record = record_from_artifacts(
            name="test_exp",
            pipeline_root=tmp_path,
            contract_version="3.0",
            fingerprint=VALID_HASH,
        )
        assert record.name == "test_exp"
        assert record.contract_version == "3.0"
        assert record.fingerprint == VALID_HASH
        assert record.experiment_id.startswith("test_exp_")
        assert record.experiment_id.endswith(VALID_HASH[:8])
        # Phase Y composer returns None when sources missing.
        assert record.experiment_provenance_hash is None

    def test_experiment_id_format(self, tmp_path: Path):
        record = record_from_artifacts(
            name="myexp",
            pipeline_root=tmp_path,
            contract_version="3.0",
            fingerprint="0123456789abcdef" + "0" * 48,
        )
        # Format: {name}_{YYYYMMDDTHHMMSS}_{fp[:8]}
        parts = record.experiment_id.split("_")
        assert parts[0] == "myexp"
        # Last segment is fp[:8]
        assert parts[-1] == "01234567"
        # Middle segment is timestamp (YYYYMMDDTHHMMSS = 15 chars with T)
        assert len(parts[-2]) == 15
        assert "T" in parts[-2]

    def test_experiment_id_override(self, tmp_path: Path):
        record = record_from_artifacts(
            name="myexp",
            pipeline_root=tmp_path,
            contract_version="3.0",
            fingerprint=VALID_HASH,
            experiment_id_override="custom_id_123",
        )
        assert record.experiment_id == "custom_id_123"


class TestRecordFromArtifactsMutuallyExclusive:
    """Mutually-exclusive trust-column sources must raise."""

    def test_both_sources_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="mutually exclusive"):
            record_from_artifacts(
                name="test",
                pipeline_root=tmp_path,
                contract_version="3.0",
                fingerprint=VALID_HASH,
                signal_metadata_path=tmp_path / "fake.json",
                captured_metrics_for_trust={"x": 1},
            )


class TestRecordFromArtifactsTrustHarvest:
    """Trust column harvesting + injection paths."""

    def test_trust_from_captured_metrics(self, tmp_path: Path):
        captured = {
            "feature_set_ref": {"name": "fs1", "content_hash": VALID_HASH},
            "compatibility_fingerprint": ANOTHER_VALID_HASH,
            "model_config_hash": THIRD_VALID_HASH,
        }
        record = record_from_artifacts(
            name="test",
            pipeline_root=tmp_path,
            contract_version="3.0",
            fingerprint="f" * 64,
            captured_metrics_for_trust=captured,
        )
        assert record.feature_set_ref == {"name": "fs1", "content_hash": VALID_HASH}
        assert record.compatibility_fingerprint == ANOTHER_VALID_HASH
        # model_config_hash is INJECTED into training_config (not top-level).
        assert record.training_config.get("model_config_hash") == THIRD_VALID_HASH

    def test_trust_from_signal_metadata_path(self, tmp_path: Path):
        metadata = {
            "feature_set_ref": {"name": "fs1", "content_hash": VALID_HASH},
            "compatibility_fingerprint": ANOTHER_VALID_HASH,
            "model_config_hash": THIRD_VALID_HASH,
        }
        sig_meta = tmp_path / "signal_metadata.json"
        sig_meta.write_text(json.dumps(metadata))
        record = record_from_artifacts(
            name="test",
            pipeline_root=tmp_path,
            contract_version="3.0",
            fingerprint="f" * 64,
            signal_metadata_path=sig_meta,
        )
        assert record.feature_set_ref == {"name": "fs1", "content_hash": VALID_HASH}
        assert record.compatibility_fingerprint == ANOTHER_VALID_HASH
        assert record.training_config.get("model_config_hash") == THIRD_VALID_HASH

    def test_training_config_not_mutated_by_caller(self, tmp_path: Path):
        """Regression: caller's training_config dict must NOT be mutated."""
        caller_config = {"model": {"model_type": "tlob"}}
        original_keys = set(caller_config.keys())
        captured = {"model_config_hash": THIRD_VALID_HASH}
        record_from_artifacts(
            name="test",
            pipeline_root=tmp_path,
            contract_version="3.0",
            fingerprint="f" * 64,
            captured_metrics_for_trust=captured,
            training_config=caller_config,
        )
        # Caller's dict should be unmodified.
        assert set(caller_config.keys()) == original_keys
        assert "model_config_hash" not in caller_config

    def test_signal_export_output_dir_override_wins(self, tmp_path: Path):
        """Override takes precedence over trust harvest."""
        captured = {"signal_export_output_dir": "/from/harvest"}
        record = record_from_artifacts(
            name="test",
            pipeline_root=tmp_path,
            contract_version="3.0",
            fingerprint=VALID_HASH,
            captured_metrics_for_trust=captured,
            signal_export_output_dir_override="/from/override",
        )
        assert record.signal_export_output_dir == "/from/override"

    def test_signal_export_output_dir_override_with_signal_metadata_path(
        self, tmp_path: Path
    ):
        """Override wins over signal_metadata_path harvest too.

        Architect pre-commit HIGH-1 (2026-05-14): the existing
        test_signal_export_output_dir_override_wins covers the
        captured_metrics_for_trust source. This test locks the
        precedence rule when the OTHER (file-based) source is used.
        Per docstring: caller override > trust harvest > None.
        """
        # signal_metadata.json doesn't carry signal_export_output_dir by
        # producer-side design (it's orchestrator-side run-time-captured),
        # so the trust harvest from the file will always return None for
        # that field. Override is the ONLY source for direct-trainer path.
        metadata = {
            "feature_set_ref": {"name": "fs", "content_hash": VALID_HASH},
            "compatibility_fingerprint": ANOTHER_VALID_HASH,
            "model_config_hash": THIRD_VALID_HASH,
        }
        sig_meta = tmp_path / "signal_metadata.json"
        sig_meta.write_text(json.dumps(metadata))

        record = record_from_artifacts(
            name="test",
            pipeline_root=tmp_path,
            contract_version="3.0",
            fingerprint=FOURTH_VALID_HASH,
            signal_metadata_path=sig_meta,
            signal_export_output_dir_override="/abs/from/override",
        )
        assert record.signal_export_output_dir == "/abs/from/override"
        # Verify other trust harvest still worked alongside the override.
        assert record.compatibility_fingerprint == ANOTHER_VALID_HASH


# ---------------------------------------------------------------------------
# Fingerprint format validation (architect pre-commit MEDIUM-1)
# ---------------------------------------------------------------------------


class TestFingerprintFormatValidation:
    """Fail-loud per hft-rules §5 on malformed fingerprint inputs."""

    def test_fingerprint_too_short_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="64-hex"):
            record_from_artifacts(
                name="test",
                pipeline_root=tmp_path,
                contract_version="3.0",
                fingerprint="abc123",  # too short
            )

    def test_fingerprint_uppercase_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="lowercase"):
            record_from_artifacts(
                name="test",
                pipeline_root=tmp_path,
                contract_version="3.0",
                fingerprint=VALID_HASH.upper(),
            )

    def test_fingerprint_non_hex_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="SHA-256"):
            record_from_artifacts(
                name="test",
                pipeline_root=tmp_path,
                contract_version="3.0",
                fingerprint="z" * 64,  # 'z' not hex
            )

    def test_fingerprint_empty_string_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="64-hex"):
            record_from_artifacts(
                name="test",
                pipeline_root=tmp_path,
                contract_version="3.0",
                fingerprint="",
            )


# ---------------------------------------------------------------------------
# Phase Y composer behavior
# ---------------------------------------------------------------------------


class TestPhaseYComposer:
    """Phase Y experiment_provenance_hash composition."""

    def test_composer_success_all_four_components(self, tmp_path: Path):
        """When all 4 sources are valid, composer returns 64-hex SHA."""
        # Build a data_dir so build_provenance can hash it.
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "file1.txt").write_text("hello")

        captured = {
            "feature_set_ref": {"name": "fs", "content_hash": VALID_HASH},
            "compatibility_fingerprint": ANOTHER_VALID_HASH,
            "model_config_hash": THIRD_VALID_HASH,
        }
        record = record_from_artifacts(
            name="test",
            pipeline_root=tmp_path,
            contract_version="3.0",
            fingerprint=FOURTH_VALID_HASH,
            captured_metrics_for_trust=captured,
            data_dir=data_dir,
        )
        assert record.experiment_provenance_hash is not None
        assert len(record.experiment_provenance_hash) == 64
        # Lowercase hex per CONTENT_HASH_RE convention.
        assert record.experiment_provenance_hash == record.experiment_provenance_hash.lower()

    def test_composer_graceful_none_missing_component(self, tmp_path: Path):
        """Missing any 1 of 4 components → composer returns None (graceful)."""
        # Only 3 trust columns; data_dir omitted so data_dir_hash is missing too.
        captured = {
            "feature_set_ref": {"name": "fs", "content_hash": VALID_HASH},
            "compatibility_fingerprint": ANOTHER_VALID_HASH,
            # model_config_hash absent
        }
        record = record_from_artifacts(
            name="test",
            pipeline_root=tmp_path,
            contract_version="3.0",
            fingerprint=VALID_HASH,
            captured_metrics_for_trust=captured,
        )
        assert record.experiment_provenance_hash is None

    def test_composer_require_complete_raises(self, tmp_path: Path):
        """require_complete_provenance=True + missing source → ValueError."""
        with pytest.raises(ValueError, match="experiment_provenance_hash"):
            record_from_artifacts(
                name="test",
                pipeline_root=tmp_path,
                contract_version="3.0",
                fingerprint=VALID_HASH,
                # No trust columns supplied → multiple missing
                require_complete_provenance=True,
            )


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------


class TestLedgerWrite:
    """Atomic ledger-write path."""

    def test_ledger_write_creates_record_file(self, tmp_path: Path):
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        record = record_from_artifacts(
            name="test_lw",
            pipeline_root=tmp_path,
            contract_version="3.0",
            fingerprint=VALID_HASH,
            ledger_path=ledger_dir,
        )
        record_path = ledger_dir / "records" / f"{record.experiment_id}.json"
        assert record_path.exists()
        # Verify content round-trips.
        loaded = ExperimentRecord.load(record_path)
        assert loaded.experiment_id == record.experiment_id
        assert loaded.name == "test_lw"
        assert loaded.fingerprint == VALID_HASH

    def test_ledger_path_missing_raises(self, tmp_path: Path):
        """Non-existent ledger_path → ValueError (fail-loud)."""
        missing = tmp_path / "nonexistent_ledger"
        with pytest.raises(ValueError, match="does not exist"):
            record_from_artifacts(
                name="test",
                pipeline_root=tmp_path,
                contract_version="3.0",
                fingerprint=VALID_HASH,
                ledger_path=missing,
            )

    def test_ledger_write_atomic_round_trip(self, tmp_path: Path):
        """Verify atomic write via canonical save() + round-trip integrity."""
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        captured = {
            "feature_set_ref": {"name": "fs", "content_hash": VALID_HASH},
            "compatibility_fingerprint": ANOTHER_VALID_HASH,
            "model_config_hash": THIRD_VALID_HASH,
        }
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "f.txt").write_text("x")

        record = record_from_artifacts(
            name="rt",
            pipeline_root=tmp_path,
            contract_version="3.0",
            fingerprint=FOURTH_VALID_HASH,
            captured_metrics_for_trust=captured,
            data_dir=data_dir,
            training_metrics={"test_ic": 0.5},
            tags=["smoke"],
            hypothesis="testing round-trip",
            ledger_path=ledger_dir,
        )
        record_path = ledger_dir / "records" / f"{record.experiment_id}.json"
        loaded = ExperimentRecord.load(record_path)
        assert loaded.experiment_provenance_hash == record.experiment_provenance_hash
        assert loaded.experiment_provenance_hash is not None
        assert loaded.compatibility_fingerprint == ANOTHER_VALID_HASH
        assert loaded.training_config.get("model_config_hash") == THIRD_VALID_HASH
        assert loaded.training_metrics == {"test_ic": 0.5}
        assert loaded.tags == ["smoke"]
        assert loaded.hypothesis == "testing round-trip"

    def test_ledger_write_creates_records_subdir(self, tmp_path: Path):
        """records/ subdir auto-created if missing under existing ledger_path."""
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        # Verify records/ does NOT exist before call.
        assert not (ledger_dir / "records").exists()
        record_from_artifacts(
            name="test",
            pipeline_root=tmp_path,
            contract_version="3.0",
            fingerprint=VALID_HASH,
            ledger_path=ledger_dir,
        )
        assert (ledger_dir / "records").exists()


# ---------------------------------------------------------------------------
# Reuse-first verification (per hft-rules §0)
# ---------------------------------------------------------------------------


class TestSSoTReuseDiscipline:
    """Locks the SSoT discipline — module must not re-implement primitives."""

    def test_module_imports_from_hft_contracts_only(self):
        """All non-stdlib imports must be from hft_contracts.* SSoTs."""
        import hft_contracts.experiment_recorder as mod
        # Check the module re-uses ExperimentRecord, build_provenance, etc.
        # (Direct import check — would fail at import if names absent.)
        assert mod.ExperimentRecord is ExperimentRecord
        assert callable(mod.build_provenance)
        assert callable(mod.compute_experiment_provenance_hash)
        # CONTENT_HASH_RE consumed from signal_manifest (canonical regex).
        from hft_contracts.signal_manifest import CONTENT_HASH_RE as canonical_re
        assert mod.CONTENT_HASH_RE is canonical_re

    def test_no_new_sha256_calls(self):
        """Module must not import hashlib directly (use canonical_hash SSoT).

        Per #PY-41 / #PY-186 / #PY-188 SSoT discipline: every SHA-256 call
        in the pipeline goes through hft_contracts.canonical_hash. The
        composer at compute_experiment_provenance_hash already does this;
        this module just delegates.
        """
        import hft_contracts.experiment_recorder as mod
        # Inspect module source for raw hashlib usage.
        import inspect
        source = inspect.getsource(mod)
        # The string "hashlib" should NOT appear in the module source
        # (delegated to canonical_hash SSoT inside compute_experiment_provenance_hash).
        assert "hashlib" not in source, (
            "experiment_recorder.py must not call hashlib directly; "
            "delegate to hft_contracts.canonical_hash SSoT."
        )
