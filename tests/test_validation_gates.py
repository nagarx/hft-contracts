"""
Golden cross-boundary validation tests.

These tests simulate the exact metadata JSON that the Rust exporter writes
(per the standardized format in export_aligned.rs) and verify the full
Python-side validation chain works correctly.

This is the safety net that catches contract drift between Rust and Python.
"""

import copy

import pytest

from hft_contracts import (
    FEATURE_COUNT,
    FULL_FEATURE_COUNT,
    SCHEMA_VERSION,
    NON_NORMALIZABLE_INDICES,
    EXPORT_METADATA_REQUIRED_FIELDS,
    EXPORT_METADATA_NORMALIZATION_FIELDS,
    EXPORT_METADATA_PROVENANCE_FIELDS,
    EXPORT_MANIFEST_REQUIRED_FIELDS,
)
from hft_contracts.validation import (
    ContractError,
    validate_export_contract,
    validate_schema_version,
    validate_normalization_not_applied,
    validate_metadata_completeness,
    validate_label_encoding,
    validate_provenance_present,
    validate_off_exchange_export_contract,
    validate_any_export_contract,
)


# =============================================================================
# Fixtures: Exact metadata as produced by standardized Rust exporter
# =============================================================================


def _make_tlob_metadata(
    *,
    n_features: int = 98,
    normalization_applied: bool = False,
) -> dict:
    """Build metadata dict matching Rust single-horizon TLOB output."""
    return {
        "day": "20250203",
        "n_sequences": 500,
        "window_size": 100,
        "n_features": n_features,
        "schema_version": SCHEMA_VERSION,
        "contract_version": SCHEMA_VERSION,
        "label_strategy": "tlob",
        "tensor_format": None,
        "label_mode": "single_horizon",
        "label_distribution": {"Down": 100, "Stable": 300, "Up": 100},
        "label_encoding": {
            "format": "signed_int8",
            "values": {"-1": "Down", "0": "Stable", "1": "Up"},
        },
        "export_timestamp": "2025-09-01T12:00:00Z",
        "normalization": {
            "strategy": "none",
            "applied": normalization_applied,
            "levels": 10,
            "sample_count": 50000,
            "feature_layout": "ask_prices_10_ask_sizes_10_bid_prices_10_bid_sizes_10",
            "params_file": "20250203_normalization.json",
        },
        "provenance": {
            "extractor_version": "0.1.0",
            "git_commit": "abc123",
            "git_dirty": False,
            "config_hash": "deadbeef",
            "contract_version": SCHEMA_VERSION,
            "export_timestamp_utc": "2025-09-01T12:00:00Z",
        },
        "validation": {
            "sequences_labels_match": True,
            "label_range_valid": True,
            "no_nan_inf": True,
        },
        "processing": {
            "messages_processed": 100000,
            "features_extracted": 50000,
            "sequences_generated": 520,
            "sequences_aligned": 500,
            "sequences_dropped": 20,
            "drop_rate_percent": "3.85",
            "buffer_coverage_ok": True,
        },
    }


def _make_triple_barrier_metadata() -> dict:
    meta = _make_tlob_metadata()
    meta["label_strategy"] = "triple_barrier"
    meta["label_encoding"] = {
        "format": "class_index_int8",
        "values": {"0": "StopLoss", "1": "Timeout", "2": "ProfitTarget"},
        "note": "Ready for PyTorch CrossEntropyLoss (class indices 0, 1, 2)",
    }
    meta["labeling"] = {"strategy": "triple_barrier", "horizons": [50, 100]}
    return meta


def _make_opportunity_metadata() -> dict:
    meta = _make_tlob_metadata()
    meta["label_strategy"] = "opportunity"
    meta["label_encoding"] = {
        "format": "signed_int8",
        "values": {"-1": "BigDown", "0": "NoOpportunity", "1": "BigUp"},
        "class_index_mapping": "class_idx = label + 1",
    }
    return meta


# =============================================================================
# Tests: Full Contract Validation
# =============================================================================


class TestValidateExportContract:
    """Test the full validation chain that consumers call at load time."""

    def test_valid_tlob_metadata_passes(self):
        meta = _make_tlob_metadata()
        warnings = validate_export_contract(meta)
        assert isinstance(warnings, list)

    def test_valid_triple_barrier_metadata_passes(self):
        meta = _make_triple_barrier_metadata()
        warnings = validate_export_contract(meta)
        assert isinstance(warnings, list)

    def test_valid_opportunity_metadata_passes(self):
        meta = _make_opportunity_metadata()
        warnings = validate_export_contract(meta)
        assert isinstance(warnings, list)

    def test_valid_full_feature_count_passes(self):
        meta = _make_tlob_metadata(n_features=FULL_FEATURE_COUNT)
        warnings = validate_export_contract(meta)
        assert isinstance(warnings, list)

    def test_invalid_feature_count_raises(self):
        meta = _make_tlob_metadata(n_features=200)
        with pytest.raises(ContractError, match="Feature count 200"):
            validate_export_contract(meta)

    def test_missing_schema_version_raises(self):
        meta = _make_tlob_metadata()
        del meta["schema_version"]
        with pytest.raises(ContractError, match="schema_version"):
            validate_export_contract(meta)

    def test_wrong_schema_version_raises(self):
        meta = _make_tlob_metadata()
        meta["schema_version"] = "1.0"
        with pytest.raises(ContractError, match="schema version"):
            validate_export_contract(meta)

    def test_normalization_applied_raises(self):
        meta = _make_tlob_metadata(normalization_applied=True)
        with pytest.raises(ContractError, match="already normalized"):
            validate_export_contract(meta)


# =============================================================================
# Tests: Individual Validators
# =============================================================================


class TestValidateSchemaVersion:
    def test_correct_version(self):
        validate_schema_version({"schema_version": SCHEMA_VERSION})

    def test_missing_version(self):
        with pytest.raises(ContractError, match="missing"):
            validate_schema_version({})

    def test_wrong_version(self):
        with pytest.raises(ContractError, match="!="):
            validate_schema_version({"schema_version": "0.1"})


class TestValidateNormalization:
    def test_not_applied_passes(self):
        validate_normalization_not_applied(
            {"normalization": {"applied": False}}
        )

    def test_applied_raises(self):
        with pytest.raises(ContractError, match="already normalized"):
            validate_normalization_not_applied(
                {"normalization": {"applied": True}}
            )

    def test_legacy_key_applied_raises(self):
        with pytest.raises(ContractError, match="already normalized"):
            validate_normalization_not_applied(
                {"normalization": {"normalization_applied": True}}
            )

    def test_missing_normalization_passes(self):
        validate_normalization_not_applied({})


class TestValidateMetadataCompleteness:
    def test_complete_metadata_no_warnings(self):
        meta = _make_tlob_metadata()
        warnings = validate_metadata_completeness(meta, strict=True)
        assert len(warnings) == 0

    def test_missing_critical_field_strict(self):
        meta = _make_tlob_metadata()
        del meta["n_features"]
        with pytest.raises(ContractError, match="n_features"):
            validate_metadata_completeness(meta, strict=True)

    def test_missing_optional_field_returns_warning(self):
        meta = _make_tlob_metadata()
        del meta["provenance"]
        warnings = validate_metadata_completeness(meta, strict=False)
        assert any("provenance" in w for w in warnings)


class TestValidateLabelEncoding:
    def test_tlob_encoding_valid(self):
        meta = _make_tlob_metadata()
        validate_label_encoding(meta)

    def test_triple_barrier_encoding_valid(self):
        meta = _make_triple_barrier_metadata()
        validate_label_encoding(meta)

    def test_unknown_strategy_raises(self):
        meta = _make_tlob_metadata()
        meta["label_strategy"] = "unknown_strategy"
        with pytest.raises(ContractError, match="Unknown label strategy"):
            validate_label_encoding(meta)

    def test_regression_strategy_passes(self):
        meta = _make_tlob_metadata()
        meta["label_strategy"] = "regression"
        meta["label_encoding"] = {
            "format": "continuous_bps",
            "dtype": "float64",
            "unit": "basis_points",
        }
        validate_label_encoding(meta)

    def test_regression_strategy_no_encoding_passes(self):
        meta = _make_tlob_metadata()
        meta["label_strategy"] = "regression"
        del meta["label_encoding"]
        validate_label_encoding(meta)

    def test_regression_wrong_dtype_raises(self):
        meta = _make_tlob_metadata()
        meta["label_strategy"] = "regression"
        meta["label_encoding"] = {"dtype": "int8"}
        with pytest.raises(ContractError, match="dtype mismatch"):
            validate_label_encoding(meta)

    def test_regression_full_contract_validation_passes(self):
        meta = _make_tlob_metadata()
        meta["label_strategy"] = "regression"
        meta["label_dtype"] = "float64"
        meta["labeling"] = {
            "label_mode": "regression",
            "label_encoding": {
                "format": "continuous_bps",
                "dtype": "float64",
                "unit": "basis_points",
            },
        }
        del meta["label_encoding"]
        warnings = validate_export_contract(meta)
        assert isinstance(warnings, list)


class TestValidateProvenance:
    def test_complete_provenance_no_warnings(self):
        meta = _make_tlob_metadata()
        warnings = validate_provenance_present(meta)
        assert len(warnings) == 0

    def test_missing_provenance(self):
        meta = _make_tlob_metadata()
        del meta["provenance"]
        warnings = validate_provenance_present(meta)
        assert any("provenance" in w for w in warnings)

    def test_incomplete_provenance(self):
        meta = _make_tlob_metadata()
        del meta["provenance"]["git_commit"]
        warnings = validate_provenance_present(meta)
        assert any("git_commit" in w for w in warnings)


# =============================================================================
# Tests: Constants Consistency
# =============================================================================


class TestContractConstants:
    def test_non_normalizable_superset_of_categorical(self):
        from hft_contracts import CATEGORICAL_INDICES

        assert CATEGORICAL_INDICES.issubset(NON_NORMALIZABLE_INDICES), (
            f"CATEGORICAL_INDICES {CATEGORICAL_INDICES} must be a subset of "
            f"NON_NORMALIZABLE_INDICES {NON_NORMALIZABLE_INDICES}"
        )

    def test_export_metadata_required_fields_not_empty(self):
        assert len(EXPORT_METADATA_REQUIRED_FIELDS) > 0

    def test_export_provenance_fields_not_empty(self):
        assert len(EXPORT_METADATA_PROVENANCE_FIELDS) > 0

    def test_export_normalization_fields_not_empty(self):
        assert len(EXPORT_METADATA_NORMALIZATION_FIELDS) > 0

    def test_standard_feature_count(self):
        assert FEATURE_COUNT == 98

    def test_full_feature_count(self):
        assert FULL_FEATURE_COUNT == 148


# =============================================================================
# Off-Exchange Validation Tests
# =============================================================================


def _make_off_exchange_metadata(**overrides) -> dict:
    """Factory for off-exchange metadata matching basic-quote-processor output."""
    meta = {
        "day": "2025-02-03",
        "n_sequences": 308,
        "window_size": 20,
        "n_features": 34,
        "schema_version": "1.0",
        "contract_version": "off_exchange_1.0",
        "label_strategy": "point_return",
        "label_encoding": "continuous_bps",
        "horizons": [1, 2, 3, 5, 10, 20, 30, 60],
        "bin_size_seconds": 60,
        "normalization": {
            "strategy": "per_day_zscore",
            "applied": False,
            "params_file": "2025-02-03_normalization.json",
        },
        "provenance": {
            "processor_version": "0.1.0",
            "export_timestamp_utc": "2026-03-23T21:27:13.984633+00:00",
        },
        "export_timestamp": "2026-03-23T21:27:13.984633+00:00",
    }
    meta.update(overrides)
    return meta


class TestOffExchangeValidation:
    """Tests for validate_off_exchange_export_contract()."""

    def test_valid_off_exchange_metadata(self):
        meta = _make_off_exchange_metadata()
        warnings = validate_off_exchange_export_contract(meta)
        assert isinstance(warnings, list)
        # No warnings for complete metadata
        assert len(warnings) == 0

    def test_wrong_feature_count_raises(self):
        meta = _make_off_exchange_metadata(n_features=50)
        with pytest.raises(ContractError, match="Feature count 50"):
            validate_off_exchange_export_contract(meta)

    def test_wrong_schema_version_raises(self):
        meta = _make_off_exchange_metadata(schema_version="2.2")
        with pytest.raises(ContractError, match="schema_version mismatch"):
            validate_off_exchange_export_contract(meta)

    def test_missing_schema_version_raises(self):
        meta = _make_off_exchange_metadata()
        del meta["schema_version"]
        with pytest.raises(ContractError, match="Missing"):
            validate_off_exchange_export_contract(meta)

    def test_wrong_contract_version_raises(self):
        meta = _make_off_exchange_metadata(contract_version="mbo_2.2")
        with pytest.raises(ContractError, match="contract_version mismatch"):
            validate_off_exchange_export_contract(meta)

    def test_normalization_applied_raises(self):
        meta = _make_off_exchange_metadata()
        meta["normalization"]["applied"] = True
        with pytest.raises(ContractError, match="normalization"):
            validate_off_exchange_export_contract(meta)

    def test_missing_optional_fields_warns(self):
        meta = _make_off_exchange_metadata()
        del meta["bin_size_seconds"]
        warnings = validate_off_exchange_export_contract(meta)
        assert len(warnings) > 0
        assert any("bin_size_seconds" in w for w in warnings)


class TestAutoDetectValidation:
    """Tests for validate_any_export_contract()."""

    def test_auto_detect_off_exchange(self):
        meta = _make_off_exchange_metadata()
        warnings = validate_any_export_contract(meta)
        assert isinstance(warnings, list)

    def test_auto_detect_mbo(self):
        """MBO metadata should route to MBO validator."""
        meta = _make_tlob_metadata()
        warnings = validate_any_export_contract(meta)
        assert isinstance(warnings, list)

    def test_auto_detect_no_contract_version(self):
        """Missing contract_version → defaults to MBO."""
        meta = _make_tlob_metadata()
        del meta["contract_version"]
        # Should not raise — defaults to MBO validation
        warnings = validate_any_export_contract(meta)
        assert isinstance(warnings, list)

    def test_off_exchange_wrong_features_via_auto(self):
        meta = _make_off_exchange_metadata(n_features=100)
        with pytest.raises(ContractError):
            validate_any_export_contract(meta)

    def test_mbo_rejects_34_features(self):
        """MBO validator should reject 34 features (out of [98, 148] range)."""
        meta = _make_tlob_metadata(n_features=34)
        with pytest.raises(ContractError, match="Feature count 34"):
            validate_export_contract(meta)
