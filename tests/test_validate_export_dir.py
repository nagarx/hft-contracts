"""Foundation Integrity tests for ``validate_export_dir`` (2026-05-29).

Directory-level export-integrity validator: composes the per-day
``validate_day_metadata`` SSoT across an export directory and adds the
cross-day + manifest<->disk reconciliation a per-day validator cannot see.

These tests lock the contract that closes the realized-victim defects found in
the Foundation Integrity audit:
- D-1 polluted directory: mixed ``schema_version`` and/or producer ``git_commit``
  across day-files (a partial re-export layered over an old one).
- D-2 lying manifest: ``days_processed`` / split counts disagree with on-disk
  files (silent truncation OR stale extra files), reconciled via
  ``partial_failure`` + ``skipped_days``.
- CF-1: the manifest's ``total_sequences`` is the PRE-alignment generated count
  and MUST NOT be compared to the on-disk post-alignment row sum (would
  false-positive on every healthy export, e.g. v3p0's 136902 vs ~66182 on disk).
- Off-exchange precondition (2026-05-30): off-exchange exports DO carry a
  ``dataset_manifest.json`` but use the ``off_exchange_*`` contract; the MBO
  directory validator must FAIL-CLEAR with a single pointer to the off-exchange
  validator, not a wall of spurious per-day ``schema_version`` mismatches.

Mirrors the fixture + signature-lock + package-surface patterns in
``test_validation_day_metadata.py``.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from hft_contracts.validation import ContractError, validate_export_dir


# =============================================================================
# Fixtures / builders
# =============================================================================

GOOD_COMMIT = "c62a1c0d9ed1b9b75dc9dabace78bf51d78ceead"
ALT_COMMIT = "b5e746dfc4f0023d6650ca8ad12b52db797fe1fa"


def _day_meta(day, *, schema="3.0", commit=GOOD_COMMIT, n_sequences=100):
    """Minimal v3.0-compliant per-day metadata (TLOB shape known to pass
    ``validate_day_metadata`` — label_encoding without a "values" key skips the
    class-name check, per the existing day-metadata tests)."""
    return {
        "day": day,
        "n_sequences": n_sequences,
        "n_features": 98,
        "window_size": 20,
        "schema_version": schema,
        "contract_version": schema,
        "label_strategy": "tlob",
        "label_encoding": {"down": -1, "stable": 0, "up": 1, "note": "TLOB"},
        "normalization": {"strategy": "none"},
        "provenance": {
            "extractor_version": "0.1.0",
            "git_commit": commit,
            "git_dirty": False,
        },
        "horizons": [10, 60, 300],
        "export_timestamp": "2026-05-29T00:00:00Z",
    }


def _build_export(
    root: Path,
    *,
    splits: dict,
    schema="3.0",
    commit=GOOD_COMMIT,
    schema_overrides=None,
    commit_overrides=None,
    nseq=100,
    nseq_overrides=None,
    omit_meta=(),
    omit_seq=(),
    manifest=None,
) -> Path:
    """Construct an MBO export directory under ``root``.

    ``splits`` maps split-name -> list of day strings. By default every day
    gets both a (placeholder, empty) ``*_sequences.npy`` and a valid
    ``*_metadata.json``; the manifest's per-split ``days`` defaults to the
    list length, ``days_processed`` to the total, ``total_sequences`` to an
    intentionally pre-align-style value (NOT Σ n_sequences). ``manifest``
    overrides merge into the default manifest.

    The validator never reads NPY bytes (it globs by name + reads counts from
    metadata), so empty placeholder ``*_sequences.npy`` files are sufficient.
    """
    schema_overrides = schema_overrides or {}
    commit_overrides = commit_overrides or {}
    nseq_overrides = nseq_overrides or {}

    for split, days in splits.items():
        sd = root / split
        sd.mkdir(parents=True, exist_ok=True)
        for day in days:
            if day not in omit_seq:
                (sd / f"{day}_sequences.npy").write_bytes(b"")
            if day not in omit_meta:
                m = _day_meta(
                    day,
                    schema=schema_overrides.get(day, schema),
                    commit=commit_overrides.get(day, commit),
                    n_sequences=nseq_overrides.get(day, nseq),
                )
                (sd / f"{day}_metadata.json").write_text(json.dumps(m))

    default_manifest = {
        "schema_version": schema,
        "days_processed": sum(len(d) for d in splits.values()),
        # PRE-alignment value, intentionally != Σ n_sequences (CF-1).
        "total_sequences": 999_999,
        "split": {s: {"days": len(d), "sequences": 12_345} for s, d in splits.items()},
        "partial_failure": {"status": "complete", "failed_partitions": []},
        "skipped_days": 0,
    }
    if manifest is not None:
        default_manifest.update(manifest)
    (root / "dataset_manifest.json").write_text(json.dumps(default_manifest))
    return root


def _failed(day, split):
    return {
        "error_class": "Label",
        "error_msg": "Label error: short session",
        "error_phase": "export",
        "partition_key": {"day": day, "split": split},
        "stranded_files": [],
    }


# =============================================================================
# Clean directories pass (and the CF-1 pre-align regression)
# =============================================================================


class TestValidateExportDirClean:
    def test_clean_passes(self, tmp_path):
        _build_export(
            tmp_path,
            splits={"train": ["20250203", "20250204"], "val": ["20250205"], "test": ["20250206"]},
        )
        result = validate_export_dir(tmp_path)
        assert isinstance(result, list)

    def test_clean_with_partial_failure_reconciles(self, tmp_path):
        """Mirrors v3p0: 3 attempted train days, 1 fails at export (no file
        written), manifest records it in partial_failure → reconciles."""
        _build_export(
            tmp_path,
            splits={"train": ["20250203", "20250204"], "val": ["20250926"], "test": ["20251114"]},
            manifest={
                "days_processed": 5,  # 3 train attempted + 1 val + 1 test
                "split": {
                    "train": {"days": 3, "sequences": 97_563},
                    "val": {"days": 1, "sequences": 20_809},
                    "test": {"days": 1, "sequences": 18_530},
                },
                "partial_failure": {
                    "status": "partial",
                    "failed_partitions": [_failed("20250703", "train")],
                },
            },
        )
        validate_export_dir(tmp_path)  # must not raise

    def test_pre_align_total_sequences_not_asserted(self, tmp_path):
        """CF-1 regression: manifest.total_sequences (pre-align) >> Σ n_sequences
        on disk MUST NOT raise (the real v3p0 case: 136902 vs ~66182)."""
        _build_export(
            tmp_path,
            splits={"train": ["20250203", "20250204"], "val": ["20250205"], "test": ["20250206"]},
            nseq=100,  # Σ = 400
            manifest={"total_sequences": 136_902},  # pre-align — must be ignored
        )
        validate_export_dir(tmp_path)  # must not raise


# =============================================================================
# D-1: cross-day uniformity (mixed schema / commit)
# =============================================================================


class TestValidateExportDirMixedSchema:
    def test_mixed_schema_raises(self, tmp_path):
        _build_export(
            tmp_path,
            splits={"train": ["20250203", "20250204"], "val": ["20250205"], "test": ["20250206"]},
            schema_overrides={"20250204": "2.2"},
        )
        with pytest.raises(ContractError) as exc:
            validate_export_dir(tmp_path)
        assert "schema_version" in str(exc.value)

    def test_uniform_legacy_schema_caught_by_per_day(self, tmp_path):
        """A uniformly-2.2 dir (manifest also 2.2) is caught by the per-day
        contract (2.2 != current 3.0), even though it passes uniformity."""
        _build_export(
            tmp_path,
            splits={"train": ["20250203"], "val": ["20250205"], "test": ["20250206"]},
            schema="2.2",
        )
        with pytest.raises(ContractError):
            validate_export_dir(tmp_path)


class TestValidateExportDirMixedCommit:
    def test_mixed_commit_raises(self, tmp_path):
        """All days valid v3.0 (per-day passes) but two producer commits →
        uniformity-commit fails."""
        _build_export(
            tmp_path,
            splits={"train": ["20250203", "20250204"], "val": ["20250205"], "test": ["20250206"]},
            commit_overrides={"20250204": ALT_COMMIT},
        )
        with pytest.raises(ContractError, match="git_commit"):
            validate_export_dir(tmp_path)


# =============================================================================
# D-2: count reconciliation (extra stale files / silent truncation)
# =============================================================================


class TestValidateExportDirCounts:
    def test_extra_stale_file_raises(self, tmp_path):
        """Bare-dir pollution shape: a 'failed' day still has an on-disk file →
        n_disk exceeds the reconciled expected count."""
        _build_export(
            tmp_path,
            splits={"train": ["20250203", "20250204", "20250703"], "val": ["20250205"], "test": ["20250206"]},
            manifest={
                "days_processed": 5,  # total disk = 3+1+1 = 5 (total check passes)
                "split": {
                    "train": {"days": 2, "sequences": 1},  # claims 2, disk has 3
                    "val": {"days": 1, "sequences": 1},
                    "test": {"days": 1, "sequences": 1},
                },
            },
        )
        with pytest.raises(ContractError, match="count"):
            validate_export_dir(tmp_path)

    def test_missing_day_silent_truncation_raises(self, tmp_path):
        """Manifest claims 3 train days, only 2 on disk, no partial_failure."""
        _build_export(
            tmp_path,
            splits={"train": ["20250203", "20250204"], "val": ["20250205"], "test": ["20250206"]},
            manifest={
                "split": {
                    "train": {"days": 3, "sequences": 1},  # claims 3, disk has 2
                    "val": {"days": 1, "sequences": 1},
                    "test": {"days": 1, "sequences": 1},
                },
            },
        )
        with pytest.raises(ContractError, match="count"):
            validate_export_dir(tmp_path)

    def test_total_days_processed_mismatch_raises(self, tmp_path):
        _build_export(
            tmp_path,
            splits={"train": ["20250203"], "val": ["20250205"], "test": ["20250206"]},
            manifest={"days_processed": 10},  # disk has 3, no failures
        )
        with pytest.raises(ContractError, match="days_processed"):
            validate_export_dir(tmp_path)


# =============================================================================
# Pairing + structural preconditions
# =============================================================================


class TestValidateExportDirPairing:
    def test_seq_without_meta_raises(self, tmp_path):
        _build_export(
            tmp_path,
            splits={"train": ["20250203", "20250204"], "val": ["20250205"], "test": ["20250206"]},
            omit_meta=("20250204",),
        )
        with pytest.raises(ContractError, match="without matching"):
            validate_export_dir(tmp_path)

    def test_meta_without_seq_raises(self, tmp_path):
        _build_export(
            tmp_path,
            splits={"train": ["20250203", "20250204"], "val": ["20250205"], "test": ["20250206"]},
            omit_seq=("20250204",),
        )
        with pytest.raises(ContractError, match="without matching"):
            validate_export_dir(tmp_path)


class TestValidateExportDirStructural:
    def test_missing_dir_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_export_dir(tmp_path / "does_not_exist")

    def test_missing_manifest_raises(self, tmp_path):
        _build_export(
            tmp_path,
            splits={"train": ["20250203"], "val": ["20250205"], "test": ["20250206"]},
        )
        (tmp_path / "dataset_manifest.json").unlink()
        with pytest.raises(ContractError, match="dataset_manifest"):
            validate_export_dir(tmp_path)

    def test_unparseable_manifest_raises(self, tmp_path):
        _build_export(
            tmp_path,
            splits={"train": ["20250203"], "val": ["20250205"], "test": ["20250206"]},
        )
        (tmp_path / "dataset_manifest.json").write_text("{not json")
        with pytest.raises(ContractError, match="unparseable"):
            validate_export_dir(tmp_path)

    def test_no_split_subdirs_raises(self, tmp_path):
        (tmp_path / "dataset_manifest.json").write_text(json.dumps({"schema_version": "3.0"}))
        with pytest.raises(ContractError, match="split subdirs"):
            validate_export_dir(tmp_path)


# =============================================================================
# Split disjointness
# =============================================================================


class TestValidateExportDirDisjoint:
    def test_day_in_two_splits_raises(self, tmp_path):
        _build_export(
            tmp_path,
            splits={"train": ["20250203"], "val": ["20250203"], "test": ["20250206"]},  # dup day
            manifest={"days_processed": 3},
        )
        with pytest.raises(ContractError, match="multiple splits"):
            validate_export_dir(tmp_path)


# =============================================================================
# Forward-compat: honest disk-truth fields (added later by C2 writer)
# =============================================================================


class TestValidateExportDirEmittedFields:
    def test_total_sequences_emitted_match_passes(self, tmp_path):
        _build_export(
            tmp_path,
            splits={"train": ["20250203", "20250204"], "val": ["20250205"], "test": ["20250206"]},
            nseq=100,  # Σ = 400
            manifest={"total_sequences_emitted": 400},
        )
        validate_export_dir(tmp_path)  # must not raise

    def test_total_sequences_emitted_mismatch_raises(self, tmp_path):
        _build_export(
            tmp_path,
            splits={"train": ["20250203", "20250204"], "val": ["20250205"], "test": ["20250206"]},
            nseq=100,  # Σ = 400
            manifest={"total_sequences_emitted": 999},  # wrong
        )
        with pytest.raises(ContractError, match="total_sequences_emitted"):
            validate_export_dir(tmp_path)


# =============================================================================
# Manifest field-shape tolerance (real on-disk shapes)
# =============================================================================


class TestValidateExportDirManifestShapes:
    def test_partial_failure_null_validates_clean(self, tmp_path):
        """A real v3p0 export (e.g. e5_timebased_30s_v3p0,
        nvda_xnas_128feat_regression_fwd_prices_v3p0) carries
        ``partial_failure: null`` (NoneType, NOT a dict). The validator must treat
        null as 'no failed partitions' (the ``isinstance(pf, dict)`` guard) and
        validate a clean export WITHOUT raising — regression-lock for the real
        on-disk null shape that the synthetic default-manifest fixture (which uses
        a dict) would otherwise never exercise."""
        _build_export(
            tmp_path,
            splits={"train": ["20250203", "20250204"], "val": ["20250205"], "test": ["20250206"]},
            manifest={"partial_failure": None},
        )
        # strict=True: must NOT raise on null partial_failure (clean export).
        result = validate_export_dir(tmp_path, strict=True)
        assert isinstance(result, list)


# =============================================================================
# Off-exchange precondition (MBO-only validator fails CLEAR, not a 233-wall)
# =============================================================================


class TestValidateExportDirOffExchange:
    """Off-exchange exports DO carry a ``dataset_manifest.json`` (ground-truth
    2026-05-30: ``data/exports/basic_nvda_60s`` — schema_version 1.0,
    contract_version ``off_exchange_1.0``). ``validate_export_dir`` is the MBO
    directory validator: handed an off-exchange dir it must FAIL-CLEAR with a
    single pointer to the off-exchange validator (short-circuiting BEFORE the
    per-day loop), NOT emit a wall of spurious per-day ``schema_version``
    mismatches. Locks the off-exchange precondition guard."""

    def _build_off_exchange(self, root: Path) -> Path:
        # schema="1.0" → off-exchange-shaped per-day metadata + manifest;
        # WITHOUT the guard this would devolve into per-day "1.0 != 3.0" errors.
        return _build_export(
            root,
            splits={"train": ["20250203", "20250204"], "val": ["20250205"], "test": ["20250206"]},
            schema="1.0",
            manifest={"contract_version": "off_exchange_1.0"},
        )

    def test_off_exchange_raises_clear_pointer_strict(self, tmp_path):
        self._build_off_exchange(tmp_path)
        with pytest.raises(ContractError, match="off-exchange") as exc:
            validate_export_dir(tmp_path)
        s = str(exc.value)
        # names the correct validator to use instead
        assert "validate_off_exchange_export_contract" in s
        # short-circuits BEFORE the per-day loop → no per-day schema wall
        assert "schema_version" not in s

    def test_off_exchange_non_strict_single_clear_error(self, tmp_path):
        self._build_off_exchange(tmp_path)
        result = validate_export_dir(tmp_path, strict=False)
        errors = [e for e in result if e.startswith("ERROR: ")]
        assert len(errors) == 1, (
            f"expected ONE clear off-exchange error, got {len(errors)}: {errors}"
        )
        assert "off-exchange" in errors[0]
        assert "off_exchange_1.0" in errors[0]
        # did not fall through to the per-day MBO schema check (no 233-wall)
        assert not any("expected '3.0'" in e for e in result)


# =============================================================================
# strict=False audit mode
# =============================================================================


class TestValidateExportDirStrictFalse:
    def test_strict_false_returns_errors_not_raises(self, tmp_path):
        _build_export(
            tmp_path,
            splits={"train": ["20250203", "20250204"], "val": ["20250205"], "test": ["20250206"]},
            commit_overrides={"20250204": ALT_COMMIT},
        )
        result = validate_export_dir(tmp_path, strict=False)
        assert isinstance(result, list)
        assert any(w.startswith("ERROR: ") for w in result)
        assert any("git_commit" in w for w in result)


# =============================================================================
# The realized-victim scenario (e5_timebased_60s pollution, in miniature)
# =============================================================================


class TestValidateExportDirRealPollution:
    def test_bare_dir_pollution_surfaces_all_defects(self, tmp_path):
        """Miniature of the polluted e5_timebased_60s: a 'failed' day (20250703)
        carries a STALE file at a different schema_version AND producer commit
        (a partial re-export layered over an old one)."""
        _build_export(
            tmp_path,
            splits={"train": ["20250203", "20250204", "20250703"], "val": ["20250926"], "test": ["20251114"]},
            schema_overrides={"20250703": "2.2"},
            commit_overrides={"20250703": ALT_COMMIT},
            manifest={
                "days_processed": 5,
                "split": {
                    "train": {"days": 3, "sequences": 1},
                    "val": {"days": 1, "sequences": 1},
                    "test": {"days": 1, "sequences": 1},
                },
                "partial_failure": {
                    "status": "partial",
                    "failed_partitions": [_failed("20250703", "train")],
                },
            },
        )
        # strict=False to inspect ALL surfaced defects at once
        result = validate_export_dir(tmp_path, strict=False)
        joined = "\n".join(result)
        assert "schema_version" in joined  # mixed {2.2, 3.0}
        assert "git_commit" in joined      # mixed producer commit
        assert "count" in joined           # extra stale file in train
        # and strict=True must raise
        with pytest.raises(ContractError):
            validate_export_dir(tmp_path)


# =============================================================================
# Package surface + signature lock (mirror test_validation_day_metadata.py)
# =============================================================================


class TestValidateExportDirPackageSurface:
    def test_importable_from_package_root(self):
        from hft_contracts import validate_export_dir as _exported
        from hft_contracts.validation import validate_export_dir as _direct
        assert _exported is _direct

    def test_in_dunder_all(self):
        import hft_contracts as pkg
        assert "validate_export_dir" in pkg.__all__


class TestValidateExportDirSignatureLock:
    def test_signature(self):
        sig = inspect.signature(validate_export_dir)
        params = list(sig.parameters.keys())
        assert params == ["export_dir", "strict"], (
            f"validate_export_dir signature changed — consumers will break. "
            f"Got: {params}"
        )

    def test_strict_is_keyword_only_default_true(self):
        sig = inspect.signature(validate_export_dir)
        strict = sig.parameters["strict"]
        assert strict.kind is inspect.Parameter.KEYWORD_ONLY
        assert strict.default is True
