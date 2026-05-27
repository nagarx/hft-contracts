"""Tests for hft_contracts._validators internal validation primitives.

Locks the validator behavior used by FeatureImportanceArtifact.__post_init__
and future artifact migrations.
"""

from __future__ import annotations

import math

import pytest

from hft_contracts._validators import (
    validate_ci_ordering,
    validate_closed_unit_interval,
    validate_feature_set_ref,
    validate_finite_float,
    validate_min_int,
    validate_non_empty_string,
    validate_non_negative_int,
    validate_open_unit_interval,
    validate_optional_sha256_hex,
    validate_positive_int,
    validate_sha256_hex,
)


class TestValidateFiniteFloat:
    def test_accepts_normal_float(self):
        validate_finite_float(1.5, "x")

    def test_accepts_zero(self):
        validate_finite_float(0.0, "x")

    def test_accepts_negative(self):
        validate_finite_float(-3.14, "x")

    def test_accepts_int(self):
        validate_finite_float(42, "x")

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="not finite"):
            validate_finite_float(float("nan"), "x")

    def test_rejects_inf(self):
        with pytest.raises(ValueError, match="not finite"):
            validate_finite_float(float("inf"), "x")

    def test_rejects_neg_inf(self):
        with pytest.raises(ValueError, match="not finite"):
            validate_finite_float(float("-inf"), "x")

    def test_rejects_bool(self):
        with pytest.raises(ValueError, match="must be a number"):
            validate_finite_float(True, "x")

    def test_context_in_message(self):
        with pytest.raises(ValueError, match="MyCtx: x="):
            validate_finite_float(float("nan"), "x", context="MyCtx")


class TestValidatePositiveInt:
    def test_accepts_positive(self):
        validate_positive_int(1, "x")

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="must be > 0"):
            validate_positive_int(0, "x")

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="must be > 0"):
            validate_positive_int(-5, "x")

    def test_rejects_bool(self):
        with pytest.raises(ValueError, match="must be int"):
            validate_positive_int(True, "x")

    def test_rejects_float(self):
        with pytest.raises(ValueError, match="must be int"):
            validate_positive_int(1.0, "x")


class TestValidateNonNegativeInt:
    def test_accepts_zero(self):
        validate_non_negative_int(0, "x")

    def test_accepts_positive(self):
        validate_non_negative_int(5, "x")

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            validate_non_negative_int(-1, "x")


class TestValidateMinInt:
    def test_accepts_at_minimum(self):
        validate_min_int(100, "x", 100)

    def test_accepts_above_minimum(self):
        validate_min_int(101, "x", 100)

    def test_rejects_below_minimum(self):
        with pytest.raises(ValueError, match="must be >= 100"):
            validate_min_int(99, "x", 100)


class TestValidateOpenUnitInterval:
    def test_accepts_mid(self):
        validate_open_unit_interval(0.5, "x")

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="must be in"):
            validate_open_unit_interval(0.0, "x")

    def test_rejects_one(self):
        with pytest.raises(ValueError, match="must be in"):
            validate_open_unit_interval(1.0, "x")

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            validate_open_unit_interval(-0.1, "x")


class TestValidateClosedUnitInterval:
    def test_accepts_zero(self):
        validate_closed_unit_interval(0.0, "x")

    def test_accepts_one(self):
        validate_closed_unit_interval(1.0, "x")

    def test_accepts_mid(self):
        validate_closed_unit_interval(0.5, "x")

    def test_rejects_above(self):
        with pytest.raises(ValueError, match="must be in"):
            validate_closed_unit_interval(1.01, "x")

    def test_rejects_below(self):
        with pytest.raises(ValueError):
            validate_closed_unit_interval(-0.01, "x")


class TestValidateSha256Hex:
    def test_accepts_valid(self):
        validate_sha256_hex("a" * 64, "x")

    def test_accepts_mixed_hex(self):
        validate_sha256_hex("0123456789abcdef" * 4, "x")

    def test_rejects_uppercase(self):
        with pytest.raises(ValueError, match="SHA-256"):
            validate_sha256_hex("A" * 64, "x")

    def test_rejects_short(self):
        with pytest.raises(ValueError):
            validate_sha256_hex("a" * 63, "x")

    def test_rejects_long(self):
        with pytest.raises(ValueError):
            validate_sha256_hex("a" * 65, "x")

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            validate_sha256_hex("", "x")

    def test_rejects_non_string(self):
        with pytest.raises(ValueError):
            validate_sha256_hex(42, "x")  # type: ignore[arg-type]


class TestValidateOptionalSha256Hex:
    def test_none_is_accepted(self):
        validate_optional_sha256_hex(None, "x")

    def test_valid_hex_accepted(self):
        validate_optional_sha256_hex("a" * 64, "x")

    def test_invalid_hex_rejected(self):
        with pytest.raises(ValueError):
            validate_optional_sha256_hex("not_hex", "x")


class TestValidateNonEmptyString:
    def test_accepts_normal(self):
        validate_non_empty_string("hello", "x")

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_non_empty_string("", "x")

    def test_rejects_none(self):
        with pytest.raises(ValueError):
            validate_non_empty_string(None, "x")  # type: ignore[arg-type]

    def test_rejects_int(self):
        with pytest.raises(ValueError):
            validate_non_empty_string(42, "x")  # type: ignore[arg-type]


class TestValidateCiOrdering:
    def test_accepts_ordered(self):
        validate_ci_ordering(0.1, 0.9)

    def test_accepts_equal(self):
        validate_ci_ordering(0.5, 0.5)

    def test_rejects_inverted(self):
        with pytest.raises(ValueError, match="inverted"):
            validate_ci_ordering(0.9, 0.1)


class TestValidateFeatureSetRef:
    def test_none_accepted(self):
        validate_feature_set_ref(None, "x")

    def test_valid_ref_accepted(self):
        validate_feature_set_ref(
            {"name": "test", "content_hash": "a" * 64}, "x",
        )

    def test_non_dict_rejected(self):
        with pytest.raises(ValueError, match="dict or None"):
            validate_feature_set_ref("not_a_dict", "x")  # type: ignore[arg-type]

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            validate_feature_set_ref(
                {"name": "", "content_hash": "a" * 64}, "x",
            )

    def test_missing_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            validate_feature_set_ref({"content_hash": "a" * 64}, "x")

    def test_invalid_content_hash_rejected(self):
        with pytest.raises(ValueError, match="SHA-256"):
            validate_feature_set_ref(
                {"name": "test", "content_hash": "not_a_hash"}, "x",
            )

    def test_non_string_content_hash_rejected(self):
        with pytest.raises(ValueError, match="string"):
            validate_feature_set_ref(
                {"name": "test", "content_hash": 42}, "x",
            )
