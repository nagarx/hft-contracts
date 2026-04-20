"""Contract tests for ``hft_contracts.compatibility.CompatibilityContract``.

Locks Phase II (plan v2.0) invariants:
    1. Determinism: same fields → same fingerprint across invocations.
    2. Completeness: varying ANY of the 11 fields changes the fingerprint.
    3. Equality: ``c1 == c2`` ⇔ ``c1.fingerprint() == c2.fingerprint()``.
    4. diff() reports exactly the fields that differ.
    5. fingerprint format: 64-char lowercase hex.
    6. label_strategy_hash helper is deterministic over dataclasses + dicts.
    7. sanitize handles tuple/list/NaN per canonical_hash SSoT.
"""

from __future__ import annotations

import pytest

from hft_contracts.compatibility import (
    COMPATIBILITY_CONTRACT_SCHEMA_VERSION,
    CompatibilityContract,
    compute_label_strategy_hash,
)


def _base_contract(**overrides) -> CompatibilityContract:
    """Factory for a plausible ``CompatibilityContract`` — overridable per-test."""
    defaults = dict(
        contract_version="2.2",
        schema_version="2.2",
        feature_count=98,
        window_size=100,
        feature_layout="default",
        data_source="mbo_lob",
        label_strategy_hash="a" * 64,
        calibration_method=None,
        primary_horizon_idx=0,
        horizons=(10, 60, 300),
        normalization_strategy="none",
    )
    defaults.update(overrides)
    return CompatibilityContract(**defaults)


class TestFingerprintDeterminism:
    def test_same_inputs_same_fingerprint(self):
        c1 = _base_contract()
        c2 = _base_contract()
        assert c1.fingerprint() == c2.fingerprint()

    def test_repeat_computation_stable(self):
        """Re-computing on the SAME instance multiple times yields the same fingerprint."""
        c = _base_contract()
        fps = [c.fingerprint() for _ in range(10)]
        assert len(set(fps)) == 1

    def test_fingerprint_format(self):
        fp = _base_contract().fingerprint()
        assert isinstance(fp, str)
        assert len(fp) == 64
        assert all(ch in "0123456789abcdef" for ch in fp), f"Not lowercase hex: {fp!r}"


class TestFingerprintSensitivity:
    """Each of the 11 fields must affect the fingerprint (completeness check).

    Parametrized over every shape-determining key; failure = field is silently
    dropped from canonicalization.
    """

    @pytest.mark.parametrize("field_name,new_value", [
        ("contract_version", "2.3"),
        ("schema_version", "2.3"),
        ("feature_count", 148),
        ("window_size", 50),
        ("feature_layout", "abc12345" * 8),
        ("data_source", "off_exchange"),
        ("label_strategy_hash", "b" * 64),
        ("calibration_method", "variance_match"),
        ("primary_horizon_idx", 1),
        ("horizons", (5, 30, 120)),
        ("normalization_strategy", "zscore"),
    ])
    def test_varying_field_changes_fingerprint(self, field_name, new_value):
        base = _base_contract()
        changed = _base_contract(**{field_name: new_value})
        assert base.fingerprint() != changed.fingerprint(), (
            f"Field {field_name!r} did NOT participate in the fingerprint. "
            f"Base value: {getattr(base, field_name)!r}; Changed value: {new_value!r}. "
            f"Consumers would silently accept different contracts as equivalent — "
            f"fix compatibility.py::CompatibilityContract.to_canonical_dict()."
        )


class TestEquality:
    def test_equal_instances_fingerprints_match(self):
        c1 = _base_contract()
        c2 = _base_contract()
        assert c1 == c2
        assert c1.fingerprint() == c2.fingerprint()

    def test_unequal_instances_fingerprints_differ(self):
        c1 = _base_contract(feature_count=98)
        c2 = _base_contract(feature_count=148)
        assert c1 != c2
        assert c1.fingerprint() != c2.fingerprint()


class TestDiff:
    def test_identical_contracts_empty_diff(self):
        c1 = _base_contract()
        c2 = _base_contract()
        assert c1.diff(c2) == {}

    def test_diff_reports_differing_field(self):
        c1 = _base_contract(feature_count=98)
        c2 = _base_contract(feature_count=148)
        diff = c1.diff(c2)
        assert "feature_count" in diff
        assert diff["feature_count"] == (98, 148)
        assert len(diff) == 1

    def test_diff_reports_multiple_fields(self):
        c1 = _base_contract(feature_count=98, calibration_method=None)
        c2 = _base_contract(feature_count=148, calibration_method="variance_match")
        diff = c1.diff(c2)
        assert set(diff.keys()) == {"feature_count", "calibration_method"}


class TestKeyFields:
    def test_key_fields_has_11_entries(self):
        """The contract surface is fixed at 11 shape-determining keys."""
        keys = _base_contract().key_fields()
        assert len(keys) == 11, f"Expected 11 fields in CompatibilityContract, got {len(keys)}: {keys}"

    def test_key_fields_names_stable(self):
        """The canonical field names must not drift silently."""
        expected = {
            "contract_version", "schema_version",
            "feature_count", "window_size",
            "feature_layout", "data_source",
            "label_strategy_hash", "calibration_method",
            "primary_horizon_idx", "horizons", "normalization_strategy",
        }
        actual = set(_base_contract().key_fields())
        assert actual == expected, f"Unexpected field set: extra={actual - expected}, missing={expected - actual}"


class TestCanonicalDict:
    def test_tuples_become_lists(self):
        """Canonical form uses JSON-native types (list not tuple) for diff stability."""
        c = _base_contract(horizons=(10, 60, 300))
        canonical = c.to_canonical_dict()
        assert canonical["horizons"] == [10, 60, 300]

    def test_none_values_preserved(self):
        c = _base_contract(calibration_method=None, primary_horizon_idx=None, horizons=None)
        canonical = c.to_canonical_dict()
        assert canonical["calibration_method"] is None
        assert canonical["primary_horizon_idx"] is None
        assert canonical["horizons"] is None


class TestLabelStrategyHash:
    def test_dict_deterministic(self):
        cfg = {"strategy": "tlob", "horizon": 10, "smoothing_window": 5, "threshold": 0.0008}
        assert compute_label_strategy_hash(cfg) == compute_label_strategy_hash(cfg)

    def test_different_params_different_hash(self):
        """Parameter changes produce different hashes (granularity preserved)."""
        a = {"strategy": "tlob", "horizon": 10, "smoothing_window": 5}
        b = {"strategy": "tlob", "horizon": 10, "smoothing_window": 10}
        assert compute_label_strategy_hash(a) != compute_label_strategy_hash(b)

    def test_key_order_insensitive(self):
        """Canonical JSON sorts keys — order in source dict doesn't affect hash."""
        a = {"alpha": 1, "beta": 2}
        b = {"beta": 2, "alpha": 1}
        assert compute_label_strategy_hash(a) == compute_label_strategy_hash(b)

    def test_dataclass_supported(self):
        from dataclasses import dataclass

        @dataclass
        class DummyLabels:
            strategy: str = "tlob"
            horizon: int = 10
            smoothing_window: int = 5

        cfg = DummyLabels()
        h = compute_label_strategy_hash(cfg)
        assert len(h) == 64
        assert isinstance(h, str)

    def test_returns_64_char_hex(self):
        h = compute_label_strategy_hash({"x": 1})
        assert len(h) == 64
        assert all(ch in "0123456789abcdef" for ch in h)


class TestSchemaVersionConstant:
    def test_schema_version_string(self):
        """The schema_version module constant is exported and well-formed."""
        assert isinstance(COMPATIBILITY_CONTRACT_SCHEMA_VERSION, str)
        # semver-ish
        parts = COMPATIBILITY_CONTRACT_SCHEMA_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts), f"Expected semver, got {COMPATIBILITY_CONTRACT_SCHEMA_VERSION!r}"


# =============================================================================
# Phase II hardening (2026-04-20): defensive __post_init__ validators.
# hft-rules §5 "If a config option exists but is not fully supported, it must
# fail fast with a precise error — never silently degrade." Prior to this
# hardening, feature_count=0 / empty strings / primary_horizon_idx out-of-range
# produced valid-looking fingerprints that poisoned downstream comparisons.
# =============================================================================


class TestDefensiveValidators:
    def test_feature_count_zero_raises(self):
        with pytest.raises(ValueError, match="feature_count"):
            _base_contract(feature_count=0)

    def test_feature_count_negative_raises(self):
        with pytest.raises(ValueError, match="feature_count"):
            _base_contract(feature_count=-1)

    def test_window_size_zero_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            _base_contract(window_size=0)

    def test_window_size_negative_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            _base_contract(window_size=-5)

    @pytest.mark.parametrize("field_name", [
        "contract_version", "schema_version", "feature_layout", "data_source",
        "label_strategy_hash", "normalization_strategy",
    ])
    def test_empty_string_raises(self, field_name):
        """Every required-string field must reject empty strings."""
        with pytest.raises(ValueError, match=field_name):
            _base_contract(**{field_name: ""})

    def test_empty_calibration_method_raises(self):
        """calibration_method=None is valid; empty string is not."""
        # None: valid
        _base_contract(calibration_method=None)
        # Empty string: invalid
        with pytest.raises(ValueError, match="calibration_method"):
            _base_contract(calibration_method="")

    def test_primary_horizon_idx_negative_raises(self):
        with pytest.raises(ValueError, match="primary_horizon_idx"):
            _base_contract(primary_horizon_idx=-1)

    def test_primary_horizon_idx_out_of_range_raises(self):
        """primary_horizon_idx must be a valid index into horizons."""
        with pytest.raises(ValueError, match="out of range"):
            _base_contract(horizons=(10, 60, 300), primary_horizon_idx=3)

    def test_primary_horizon_idx_none_with_horizons_allowed(self):
        """None is valid (non-HMHP models) regardless of horizons."""
        c = _base_contract(horizons=(10, 60, 300), primary_horizon_idx=None)
        assert c.primary_horizon_idx is None

    def test_empty_horizons_raises(self):
        with pytest.raises(ValueError, match="horizons"):
            _base_contract(horizons=())
        with pytest.raises(ValueError, match="horizons"):
            _base_contract(horizons=[])

    def test_non_positive_horizon_value_raises(self):
        with pytest.raises(ValueError, match="positive ints"):
            _base_contract(horizons=(10, 0, 300))
        with pytest.raises(ValueError, match="positive ints"):
            _base_contract(horizons=(10, -60, 300))

    def test_horizons_list_coerced_to_tuple(self):
        """horizons passed as list → stored as tuple for fingerprint stability."""
        c = _base_contract(horizons=[10, 60, 300])
        assert isinstance(c.horizons, tuple)
        assert c.horizons == (10, 60, 300)

    def test_horizons_list_vs_tuple_same_fingerprint(self):
        """Tuple coercion guarantees fingerprint stability across JSON round-trips."""
        c_tuple = _base_contract(horizons=(10, 60, 300))
        c_list = _base_contract(horizons=[10, 60, 300])
        assert c_tuple.fingerprint() == c_list.fingerprint()

    def test_horizons_none_allowed(self):
        c = _base_contract(horizons=None, primary_horizon_idx=None)
        assert c.horizons is None

    def test_non_int_feature_count_raises(self):
        """float feature_count (e.g., from JSON-parse) must be rejected."""
        with pytest.raises(ValueError, match="feature_count"):
            _base_contract(feature_count=98.0)

    def test_bool_horizon_value_raises(self):
        """Booleans are Python-int subclasses; must NOT pass positive-int check."""
        with pytest.raises(ValueError, match="positive ints"):
            _base_contract(horizons=(10, True, 300))

    def test_bool_primary_horizon_idx_raises(self):
        with pytest.raises(ValueError, match="primary_horizon_idx"):
            _base_contract(primary_horizon_idx=True)
