"""Tests for hft_contracts.canonical_hash (Phase 4 Batch 4c hardening).

This module is the SINGLE SOURCE OF TRUTH for the canonical JSON + SHA-256
convention used across hft-ops, hft-feature-evaluator, lob-model-trainer
(inline copy + parity test), and any future cross-module hashing site.

If these tests fail, every dependent module's hash output is at risk —
treat as P0.
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from hft_contracts.canonical_hash import (
    canonical_json_blob,
    sanitize_for_hash,
    sha256_hex,
)


# ---------------------------------------------------------------------------
# Frozen golden-hash fixtures
# ---------------------------------------------------------------------------


class TestGoldenHashes:
    """These hashes MUST NEVER CHANGE without a deliberate contract bump
    (new symbol name). Drift here silently invalidates every downstream
    fingerprint/ledger-record/feature-set-hash in the monorepo."""

    def test_empty_dict(self):
        # json.dumps({}, sort_keys=True, default=str) → "{}"
        # sha256("{}") → 44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a
        blob = canonical_json_blob({})
        assert blob == b"{}"
        assert sha256_hex(blob) == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"

    def test_simple_dict(self):
        blob = canonical_json_blob({"a": 1, "b": "x"})
        assert blob == b'{"a": 1, "b": "x"}'
        # Locked golden — regenerate via:
        #   hashlib.sha256(b'{"a": 1, "b": "x"}').hexdigest()
        assert sha256_hex(blob) == hashlib.sha256(blob).hexdigest()

    def test_sort_keys_enforced(self):
        h1 = sha256_hex(canonical_json_blob({"a": 1, "b": 2}))
        h2 = sha256_hex(canonical_json_blob({"b": 2, "a": 1}))
        assert h1 == h2

    def test_nested_dict_sort_keys(self):
        obj = {"outer": {"z": 1, "a": 2}, "alpha": 1}
        blob = canonical_json_blob(obj)
        # Nested dicts must also be sort_key'd — they are, because
        # json.dumps(sort_keys=True) applies recursively.
        assert blob == b'{"alpha": 1, "outer": {"a": 2, "z": 1}}'


# ---------------------------------------------------------------------------
# canonical_json_blob contract
# ---------------------------------------------------------------------------


class TestCanonicalJsonBlob:
    def test_returns_bytes(self):
        blob = canonical_json_blob({"k": "v"})
        assert isinstance(blob, bytes)

    def test_utf8_encoded(self):
        blob = canonical_json_blob({"k": "v"})
        assert blob.decode("utf-8") == '{"k": "v"}'

    def test_default_no_sanitize(self):
        # By default, NaN serializes as the non-strict JSON token 'NaN'
        # (deterministic across CPython but rejected by strict consumers).
        blob = canonical_json_blob({"x": float("nan")})
        assert b"NaN" in blob

    def test_sanitize_true_replaces_nan_with_null(self):
        # With sanitize=True, NaN/Inf → None → 'null' in JSON.
        blob = canonical_json_blob({"x": float("nan")}, sanitize=True)
        assert b"null" in blob
        assert b"NaN" not in blob

    def test_sanitize_true_preserves_finite(self):
        blob = canonical_json_blob({"x": 1.5, "y": 0.0}, sanitize=True)
        assert blob == b'{"x": 1.5, "y": 0.0}'

    def test_default_str_fallback_for_path(self):
        from pathlib import Path
        blob = canonical_json_blob({"p": Path("/tmp/foo")})
        # default=str stringifies the Path
        assert b"/tmp/foo" in blob

    def test_primitives_pass_through(self):
        assert canonical_json_blob(42) == b"42"
        assert canonical_json_blob("hello") == b'"hello"'
        assert canonical_json_blob(True) == b"true"
        assert canonical_json_blob(None) == b"null"

    def test_list_sorts_keys_inside_dict_elements(self):
        blob = canonical_json_blob([{"z": 1, "a": 2}, {"b": 3, "y": 4}])
        assert blob == b'[{"a": 2, "z": 1}, {"b": 3, "y": 4}]'


# ---------------------------------------------------------------------------
# sanitize_for_hash contract
# ---------------------------------------------------------------------------


class TestSanitizeForHash:
    def test_nan_to_none(self):
        assert sanitize_for_hash(float("nan")) is None

    def test_pos_inf_to_none(self):
        assert sanitize_for_hash(float("inf")) is None

    def test_neg_inf_to_none(self):
        assert sanitize_for_hash(float("-inf")) is None

    def test_finite_float_preserved(self):
        assert sanitize_for_hash(1.5) == 1.5
        assert sanitize_for_hash(-3.14) == -3.14
        assert sanitize_for_hash(0.0) == 0.0

    def test_int_preserved(self):
        assert sanitize_for_hash(42) == 42
        assert sanitize_for_hash(-1) == -1

    def test_bool_preserved(self):
        # bool is a subclass of int; sanitizer checks float first so
        # bool passes through as True/False (valid JSON).
        assert sanitize_for_hash(True) is True
        assert sanitize_for_hash(False) is False

    def test_none_preserved(self):
        assert sanitize_for_hash(None) is None

    def test_string_preserved(self):
        assert sanitize_for_hash("hello") == "hello"

    def test_dict_recurses(self):
        assert sanitize_for_hash({"x": float("nan"), "y": 1}) == {"x": None, "y": 1}

    def test_nested_dict(self):
        obj = {"outer": {"inner": float("inf"), "keep": 2.0}}
        assert sanitize_for_hash(obj) == {"outer": {"inner": None, "keep": 2.0}}

    def test_list_recurses(self):
        assert sanitize_for_hash([1.0, float("nan"), 2.0]) == [1.0, None, 2.0]

    def test_tuple_canonicalizes_to_list(self):
        # Tuples canonicalize to lists — JSON-identical representation.
        assert sanitize_for_hash((1, 2, 3)) == [1, 2, 3]

    def test_tuple_with_nan_becomes_list_with_none(self):
        assert sanitize_for_hash((float("nan"), 1)) == [None, 1]


# ---------------------------------------------------------------------------
# sha256_hex contract
# ---------------------------------------------------------------------------


class TestSha256Hex:
    def test_format(self):
        h = sha256_hex(b"hello")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
        assert h == h.lower()

    def test_no_prefix(self):
        h = sha256_hex(b"hello")
        assert not h.startswith("sha256:")

    def test_empty_bytes_golden(self):
        # sha256(b"") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        assert sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_deterministic(self):
        assert sha256_hex(b"x") == sha256_hex(b"x")


# ---------------------------------------------------------------------------
# Integration: the whole pipeline produces the same hash as the monorepo
# convention (json.dumps(sort_keys=True, default=str) + sha256 hex)
# ---------------------------------------------------------------------------


class TestMonorepoConventionAlignment:
    """Lock parity with the pre-extraction convention used by dedup.py,
    lineage.py, evaluator pipeline, and the 2 feature_sets hashing
    modules. These five sites must all produce byte-identical output
    for the same input — otherwise existing Phase 3 fingerprints would
    change, invalidating the ledger."""

    def test_matches_pre_extraction_convention(self):
        # This is the EXACT form used by dedup.py:398 and lineage.py:153
        # pre-extraction. Any change here breaks existing fingerprints.
        obj = {"a": 1, "b": [2, 3], "c": "hello"}
        expected = hashlib.sha256(
            json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        assert sha256_hex(canonical_json_blob(obj)) == expected

    def test_matches_sanitize_convention(self):
        # Pre-extraction form used by compute_profile_hash (evaluator).
        profiles = {
            "a": {"value": float("nan"), "ok": 1.0},
            "b": {"value": 0.5, "ok": float("inf")},
        }
        sanitized = {
            k: {k2: (v2 if isinstance(v2, float) and math.isfinite(v2) else (
                     v2 if not isinstance(v2, float) else None))
                for k2, v2 in v.items()}
            for k, v in profiles.items()
        }
        expected = hashlib.sha256(
            json.dumps(sanitized, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        assert sha256_hex(canonical_json_blob(profiles, sanitize=True)) == expected
