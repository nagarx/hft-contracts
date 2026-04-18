"""Tests for hft_contracts.provenance (Phase 6 6B.4 co-move).

Mirrors the contract-level subset of hft-ops tests/test_provenance.py.
Tests that are hft-ops-specific (retroactive-backfill CLI invocations,
live experiment-ledger integration) stay in hft-ops.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hft_contracts.provenance import (
    GitInfo,
    NOT_GIT_TRACKED_SENTINEL,
    PROVENANCE_SCHEMA_VERSION,
    Provenance,
    build_provenance,
    capture_git_info,
    hash_config_dict,
    hash_directory_manifest,
    hash_file,
)


class TestHashFile:
    def test_existing_file(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h = hash_file(f)
        assert len(h) == 64
        assert h == hash_file(f)

    def test_different_content(self, tmp_path: Path):
        f1 = tmp_path / "a.txt"
        f1.write_text("hello")
        f2 = tmp_path / "b.txt"
        f2.write_text("world")
        assert hash_file(f1) != hash_file(f2)

    def test_missing_file(self, tmp_path: Path):
        assert hash_file(tmp_path / "nonexistent.txt") == ""


class TestHashConfigDict:
    def test_deterministic(self):
        cfg = {"a": 1, "b": {"c": 3}}
        assert hash_config_dict(cfg) == hash_config_dict(cfg)
        assert len(hash_config_dict(cfg)) == 64

    def test_order_independent(self):
        assert hash_config_dict({"a": 1, "b": 2}) == hash_config_dict({"b": 2, "a": 1})

    def test_different_values(self):
        assert hash_config_dict({"a": 1}) != hash_config_dict({"a": 2})


class TestHashDirectoryManifest:
    def test_empty_dir(self, tmp_path: Path):
        d = tmp_path / "empty"
        d.mkdir()
        assert len(hash_directory_manifest(d)) == 64

    def test_deterministic(self, tmp_path: Path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "file.txt").write_text("content")
        assert hash_directory_manifest(d) == hash_directory_manifest(d)

    def test_detects_file_addition(self, tmp_path: Path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "a.txt").write_text("content")
        h1 = hash_directory_manifest(d)
        (d / "b.txt").write_text("new")
        assert hash_directory_manifest(d) != h1

    def test_missing_dir(self, tmp_path: Path):
        assert hash_directory_manifest(tmp_path / "nonexistent") == ""


class TestGitInfo:
    def test_roundtrip(self):
        info = GitInfo(commit_hash="abc123", branch="main", dirty=True, short_hash="abc1")
        restored = GitInfo.from_dict(info.to_dict())
        assert restored.commit_hash == "abc123"
        assert restored.branch == "main"
        assert restored.dirty is True

    def test_empty_defaults(self):
        info = GitInfo()
        assert info.commit_hash == ""
        assert info.dirty is False


class TestCaptureGitInfo:
    def test_non_repo_dir(self, tmp_path: Path):
        info = capture_git_info(tmp_path)
        assert info.commit_hash == NOT_GIT_TRACKED_SENTINEL
        assert info.short_hash == NOT_GIT_TRACKED_SENTINEL[:8]


class TestProvenanceSchemaVersion:
    def test_default_schema_version(self):
        prov = Provenance()
        assert prov.schema_version == PROVENANCE_SCHEMA_VERSION == "1.0"

    def test_default_not_retroactive(self):
        assert Provenance().retroactive is False

    def test_retroactive_true_roundtrip(self):
        prov = Provenance(
            git=GitInfo(commit_hash="not_git_tracked", short_hash="not_git_"),
            contract_version="2.2",
            retroactive=True,
        )
        restored = Provenance.from_dict(prov.to_dict())
        assert restored.retroactive is True
        assert restored.schema_version == "1.0"

    def test_old_records_default_to_schema_1_0(self):
        """Records without schema_version default to '1.0' for backward compat."""
        restored = Provenance.from_dict({
            "git": {"commit_hash": "abc", "branch": "main"},
            "config_hashes": {},
            "contract_version": "2.2",
            "timestamp_utc": "2026-03-01T00:00:00+00:00",
        })
        assert restored.schema_version == "1.0"
        assert restored.retroactive is False


class TestProvenanceRoundtrip:
    def test_roundtrip(self):
        prov = Provenance(
            git=GitInfo(commit_hash="abc", branch="main", dirty=False, short_hash="ab"),
            config_hashes={"manifest": "hash1", "extractor": "hash2"},
            data_dir_hash="datahash",
            contract_version="2.2",
            timestamp_utc="2026-03-05T12:00:00+00:00",
        )
        restored = Provenance.from_dict(prov.to_dict())
        assert restored.git.commit_hash == "abc"
        assert restored.config_hashes["manifest"] == "hash1"
        assert restored.contract_version == "2.2"


class TestBuildProvenance:
    def test_build_with_manifest(self, tmp_path: Path):
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("experiment:\n  name: test\n")
        prov = build_provenance(
            tmp_path,
            manifest_path=manifest,
            contract_version="2.2",
        )
        assert prov.config_hashes["manifest"] != ""
        assert prov.contract_version == "2.2"
        assert prov.timestamp_utc != ""

    def test_build_with_inline_trainer_config_dict(self, tmp_path: Path):
        """Phase 6 6A.3 parity: inline trainer_config dict produces a
        canonical-hash-based config_hashes['trainer'] entry (SHA-256 hex, 64)."""
        trainer_cfg = {
            "name": "E5_60s_huber_cvml",
            "model": {"model_type": "tlob", "input_size": 98},
            "train": {"batch_size": 128, "seed": 42},
        }
        prov = build_provenance(
            tmp_path,
            trainer_config_dict=trainer_cfg,
            contract_version="2.2",
        )
        assert "trainer" in prov.config_hashes
        assert len(prov.config_hashes["trainer"]) == 64
        # Deterministic across call sites via the hft_contracts canonical-hash SSoT.
        prov2 = build_provenance(
            tmp_path,
            trainer_config_dict=dict(trainer_cfg),
            contract_version="2.2",
        )
        assert prov.config_hashes["trainer"] == prov2.config_hashes["trainer"]

    def test_trainer_config_path_and_dict_mutually_exclusive(self, tmp_path: Path):
        """Phase 6 6A.3: callers cannot supply BOTH trainer_config_path
        (legacy wrapper-config) AND trainer_config_dict (inline Phase 1
        wrapper-less) — they represent mutually exclusive manifest patterns.
        """
        trainer_yaml = tmp_path / "trainer.yaml"
        trainer_yaml.write_text("model:\n  model_type: tlob\n")
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_provenance(
                tmp_path,
                trainer_config_path=trainer_yaml,
                trainer_config_dict={"model": {"model_type": "tlob"}},
                contract_version="2.2",
            )
