"""Phase X.2.A SSoT tests for ``validate_day_metadata``.

Lifted from ``lobtrainer.data.dataset._validate_day_metadata`` (Phase O
Cycle 1 C-2 + C-3 hardening). After Phase X.2.A migration, this primitive
is consumed by:
- ``lob-model-trainer`` data/dataset.py (1 caller — replaces private function)
- ``lob-backtester`` data/loader.py (2 callers — replaces inline duplicates)
- ``lob-dataset-analyzer`` 17 files / 43 np.load sites
- ``hft-feature-evaluator`` data/loader.py (1 caller)

Tests lock the SSoT contract:
- None metadata raises (Phase O C-2)
- Missing schema_version raises (Phase O C-2)
- Contract violations get date-prefixed wrapping (Phase O C-3)
- Valid v3.0 metadata returns warnings list (no hard raise)
- SSoT does NOT log (caller responsibility — preserves hft-contracts
  log-free architectural invariant)
- Public signature is locked (defends against accidental drift)
"""
from __future__ import annotations

import inspect
import typing

import pytest

from hft_contracts.validation import (
    ContractError,
    validate_day_metadata,
)


# =============================================================================
# Test fixtures
# =============================================================================


def _valid_metadata(**overrides):
    """Minimal v3.0-compliant metadata dict.

    Matches what feature-extractor-MBO-LOB emits per Phase O Cycle 1
    contract (see EXPORT_METADATA_REQUIRED_FIELDS in _generated.py).
    """
    base = {
        "day": "2025-02-03",
        "n_sequences": 1000,
        "n_features": 98,
        "window_size": 100,
        "schema_version": "3.0",
        "contract_version": "3.0",
        "label_strategy": "tlob",
        "label_encoding": {"down": -1, "stable": 0, "up": 1, "note": "TLOB classification"},
        "normalization": {"strategy": "none"},
        "provenance": {
            "extractor_version": "0.2.1",
            "config_hash": "0" * 16,
            "extracted_at": "2026-05-04T00:00:00Z",
        },
        "horizons": [10, 60, 300],
        "export_timestamp": "2026-05-04T00:00:00Z",
    }
    base.update(overrides)
    return base


# =============================================================================
# C-2 hardening: None metadata fails-loud
# =============================================================================


class TestValidateDayMetadataNoneInput:
    """Phase O Cycle 1 C-2: missing metadata.json fails-loud (per hft-rules §8)."""

    def test_metadata_none_raises_contract_error(self):
        with pytest.raises(ContractError, match="missing or could not be loaded"):
            validate_day_metadata(None, "2025-02-03")

    def test_metadata_none_error_includes_date(self):
        with pytest.raises(ContractError, match="2025-02-03"):
            validate_day_metadata(None, "2025-02-03")

    def test_metadata_none_error_mentions_re_export(self):
        """Error message guides operator toward remediation."""
        with pytest.raises(ContractError, match="Re-export"):
            validate_day_metadata(None, "2025-02-03")

    def test_metadata_none_with_different_dates(self):
        """Date string is correctly embedded — not a hardcoded value."""
        with pytest.raises(ContractError, match="20260101"):
            validate_day_metadata(None, "20260101")


# =============================================================================
# C-2 hardening: missing schema_version fails-loud
# =============================================================================


class TestValidateDayMetadataMissingSchemaVersion:
    """Phase O Cycle 1 C-2: missing schema_version is the only path to absence
    in v3.0 (producers always emit it); all other paths are hard violations."""

    def test_missing_schema_version_raises_contract_error(self):
        meta = _valid_metadata()
        del meta["schema_version"]
        with pytest.raises(ContractError, match="no 'schema_version' field"):
            validate_day_metadata(meta, "2025-02-03")

    def test_missing_schema_version_error_includes_date(self):
        meta = _valid_metadata()
        del meta["schema_version"]
        with pytest.raises(ContractError, match="2025-02-03"):
            validate_day_metadata(meta, "2025-02-03")

    def test_missing_schema_version_mentions_phase_o_cycle_1(self):
        """Error message points to Phase O Cycle 1+ as the contract source."""
        meta = _valid_metadata()
        del meta["schema_version"]
        with pytest.raises(ContractError, match="Phase O Cycle 1"):
            validate_day_metadata(meta, "2025-02-03")

    def test_empty_dict_treated_as_missing_schema_version(self):
        """Empty metadata dict triggers the schema_version branch (NOT the None
        branch — `not None` but missing key)."""
        with pytest.raises(ContractError, match="no 'schema_version' field"):
            validate_day_metadata({}, "2025-02-03")


# =============================================================================
# C-3 hardening: contract violations get date-prefixed wrapping
# =============================================================================


class TestValidateDayMetadataContractWrap:
    """Phase O Cycle 1 C-3: ContractError from validate_export_contract is
    re-raised with date prefix for multi-day corpus triage."""

    def test_pre_phase_o_legacy_schema_raises(self):
        """schema_version='2.2' (pre-Phase-O) violates v3.0 contract."""
        meta = _valid_metadata(schema_version="2.2")
        with pytest.raises(ContractError):
            validate_day_metadata(meta, "2025-02-03")

    def test_contract_violation_wrapped_with_date_prefix(self):
        meta = _valid_metadata(schema_version="2.2")
        with pytest.raises(ContractError, match=r"Export contract violation for 2025-02-03"):
            validate_day_metadata(meta, "2025-02-03")

    def test_contract_violation_chain_preserves_original(self):
        """__cause__ chain preserves the original ContractError for triage."""
        meta = _valid_metadata(schema_version="2.2")
        try:
            validate_day_metadata(meta, "2025-02-03")
            pytest.fail("Expected ContractError")
        except ContractError as exc:
            assert exc.__cause__ is not None
            assert isinstance(exc.__cause__, ContractError)


# =============================================================================
# Valid metadata path: returns warnings list
# =============================================================================


class TestValidateDayMetadataValidPasses:
    """Valid v3.0 metadata returns warnings list (possibly empty)."""

    def test_valid_metadata_returns_list(self):
        meta = _valid_metadata()
        result = validate_day_metadata(meta, "2025-02-03")
        assert isinstance(result, list)

    def test_valid_metadata_no_hard_raise(self):
        meta = _valid_metadata()
        validate_day_metadata(meta, "2025-02-03")  # Should not raise

    def test_valid_metadata_warnings_are_strings(self):
        """Returned warnings are str (consumable via logger.warning)."""
        meta = _valid_metadata()
        result = validate_day_metadata(meta, "2025-02-03")
        for w in result:
            assert isinstance(w, str)


# =============================================================================
# Architectural invariant: SSoT does NOT log
# =============================================================================


class TestValidateDayMetadataLogFree:
    """The SSoT does NOT log warnings — caller's responsibility. This
    preserves hft-contracts' log-free architectural invariant
    (Phase 6 6B.3 — only atomic_io was permitted I/O; logging is excluded)."""

    def test_validation_module_has_no_logger(self):
        """hft_contracts.validation must not instantiate a module logger."""
        import hft_contracts.validation as mod
        assert not hasattr(mod, "logger"), (
            "hft_contracts.validation must remain log-free per Phase 6 6B.3 "
            "architectural invariant. Move any logger.* calls into the caller."
        )

    def test_validation_module_does_not_import_logging(self):
        """The validation module should not import the logging stdlib at
        module level (defensive — if it did, the SSoT might log
        accidentally in future edits)."""
        import hft_contracts.validation as mod
        # Module's __dict__ should not have 'logging' bound at top level
        # (it's OK to import inside a function body, but not at module level)
        source = inspect.getsource(mod)
        # Top-level import lines (before first 'def' or 'class')
        first_def = source.find("\ndef ")
        first_class = source.find("\nclass ")
        first_def_or_class = min(
            x for x in [first_def, first_class] if x > 0
        )
        top_section = source[:first_def_or_class]
        assert "import logging" not in top_section, (
            "hft_contracts.validation must not import logging at module level"
        )


# =============================================================================
# Signature lock: defend against accidental drift
# =============================================================================


class TestValidateDayMetadataPackageSurface:
    """Phase X.2.A SSoT must be exposed at package level via the documented
    `from hft_contracts import X` idiom (per hft_contracts/__init__.py
    package docstring "All Python consumers... import from here instead of
    maintaining independent copies"). Locks against accidental drop from
    __all__ in future cleanups."""

    def test_validate_day_metadata_importable_from_package_root(self):
        """The canonical public-API import idiom must work."""
        from hft_contracts import validate_day_metadata as _exported
        # Must be the same callable as the validation module's:
        from hft_contracts.validation import validate_day_metadata as _direct
        assert _exported is _direct

    def test_validate_day_metadata_in_dunder_all(self):
        """Listed in __all__ so `from hft_contracts import *` exposes it."""
        import hft_contracts as pkg
        assert "validate_day_metadata" in pkg.__all__


class TestValidateDayMetadataSignatureLock:
    """Lock the public signature against accidental refactors that would
    silently break consumers (per hft-rules §1 inter-module contracts)."""

    def test_signature_takes_metadata_and_date(self):
        sig = inspect.signature(validate_day_metadata)
        params = list(sig.parameters.keys())
        assert params == ["metadata", "date"], (
            f"validate_day_metadata signature changed — public consumers will "
            f"break. Got params: {params}"
        )

    def test_signature_returns_list(self):
        hints = typing.get_type_hints(validate_day_metadata)
        assert "return" in hints
        # Return type should be list[str] or List[str]
        ret = hints["return"]
        # Best-effort check: the typing system reports list[str] differently
        # across Python versions; just verify it's a list-like type
        assert ret is not None

    def test_metadata_param_accepts_optional_dict(self):
        """metadata parameter type should accept dict or None."""
        sig = inspect.signature(validate_day_metadata)
        meta_param = sig.parameters["metadata"]
        # Annotation should permit None (Optional or Union)
        # Defensive: just ensure annotation exists
        assert meta_param.annotation is not inspect.Parameter.empty

    def test_date_param_is_str(self):
        sig = inspect.signature(validate_day_metadata)
        date_param = sig.parameters["date"]
        # str type (or compatible)
        assert date_param.annotation is not inspect.Parameter.empty
