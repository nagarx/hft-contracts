"""Unit tests for `hft_contracts.validation.assert_finite_array`.

#PY-63 (2026-05-07): SSoT helper added to consolidate 7+ duplicated
inline NaN/Inf checks across hft-contracts, lob-model-trainer, and
lob-backtester. Per hft-rules §6 ("Tests document behavior and expose
implementation correctness"): these tests lock the helper's contract
against future drift.

Per hft-rules §0 reuse-first: all consumers MUST call this helper
rather than re-implementing the `np.all(np.isfinite(arr))` + count
+ raise idiom inline. These tests verify the helper's:
  - Raise type (always ValueError)
  - Diagnostic content (NaN count, Inf count, total size)
  - extra_diagnostic kwarg appending
  - Edge cases (empty array, 0-d scalar, multi-dim)
"""

import numpy as np
import pytest

from hft_contracts.validation import assert_finite_array


class TestAssertFiniteArray:
    """Golden tests for the SSoT helper at hft_contracts.validation."""

    # ----------------------------- HAPPY PATHS ------------------------------

    def test_clean_1d_array_passes(self):
        """Finite 1-D array does not raise."""
        assert_finite_array(np.array([1.0, 2.0, 3.0]), name="clean")

    def test_clean_multi_dim_array_passes(self):
        """Finite multi-D array does not raise."""
        arr = np.zeros((4, 3, 2))
        assert_finite_array(arr, name="clean3d")

    def test_empty_array_passes_vacuously(self):
        """Empty array has no non-finite values → np.all of empty mask = True."""
        assert_finite_array(np.array([]), name="empty")

    def test_zero_d_scalar_finite_passes(self):
        """0-d scalar (np.float64 wrapped) passes when finite."""
        assert_finite_array(np.array(3.14), name="scalar")

    def test_int_array_passes(self):
        """Integer arrays are always finite — should pass."""
        assert_finite_array(np.array([1, 2, 3], dtype=np.int64), name="ints")

    # ---------------------------- FAIL-LOUD PATHS ---------------------------

    def test_nan_raises_with_correct_count(self):
        """NaN-containing array raises with diagnostic 'N NaN, 0 Inf out of M'."""
        arr = np.array([1.0, np.nan, 3.0, 4.0])
        with pytest.raises(ValueError, match=r"contains 1 NaN, 0 Inf out of 4"):
            assert_finite_array(arr, name="x")

    def test_positive_inf_raises_with_correct_count(self):
        """+Inf-containing array raises."""
        arr = np.array([1.0, 2.0, np.inf])
        with pytest.raises(ValueError, match=r"contains 0 NaN, 1 Inf out of 3"):
            assert_finite_array(arr, name="y")

    def test_negative_inf_counted_as_inf(self):
        """-Inf is counted in the Inf count (np.isinf detects both signs)."""
        arr = np.array([-np.inf, 2.0, 3.0])
        with pytest.raises(ValueError, match=r"contains 0 NaN, 1 Inf"):
            assert_finite_array(arr, name="negfp")

    def test_mixed_nan_inf_raises_with_both_counts(self):
        """Mixed NaN + Inf produce both counts in the message."""
        arr = np.array([1.0, np.nan, np.inf, np.nan, 5.0])
        with pytest.raises(ValueError, match=r"contains 2 NaN, 1 Inf out of 5"):
            assert_finite_array(arr, name="mixed")

    def test_all_nan_array_raises(self):
        """All-NaN array reports the full count."""
        arr = np.array([np.nan, np.nan, np.nan])
        with pytest.raises(ValueError, match=r"contains 3 NaN, 0 Inf out of 3"):
            assert_finite_array(arr, name="all_nan")

    def test_zero_d_nan_scalar_raises(self):
        """0-d NaN scalar raises (size=1, n_nan=1)."""
        with pytest.raises(ValueError, match=r"contains 1 NaN, 0 Inf out of 1"):
            assert_finite_array(np.array(np.nan), name="bad_scalar")

    def test_multi_dim_reports_total_size(self):
        """Multi-dim array reports total .size, NOT just len (first dim)."""
        # Shape (2,2) = 4 cells; 1 NaN; should report `out of 4`.
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(ValueError, match=r"contains 1 NaN, 0 Inf out of 4"):
            assert_finite_array(arr, name="2d")

    # --------------------------- KWARG SEMANTICS ----------------------------

    def test_name_appears_at_message_start(self):
        """The `name` kwarg is the message prefix (used by all 7 migrated callers)."""
        arr = np.array([np.nan])
        with pytest.raises(ValueError, match=r"^my_descriptive_name:"):
            assert_finite_array(arr, name="my_descriptive_name")

    def test_extra_diagnostic_appended(self):
        """`extra_diagnostic` text appears AFTER 'input invariant violation.'"""
        arr = np.array([1.0, np.nan])
        with pytest.raises(ValueError, match=r"input invariant violation\. Investigate upstream\."):
            assert_finite_array(arr, name="x", extra_diagnostic="Investigate upstream.")

    def test_no_extra_diagnostic_when_none(self):
        """When extra_diagnostic is None (default), message ends after 'violation.'"""
        arr = np.array([np.nan])
        with pytest.raises(ValueError) as excinfo:
            assert_finite_array(arr, name="x")
        # Should end with '.' from "violation." — no trailing whitespace+text
        assert excinfo.value.args[0].rstrip().endswith("input invariant violation.")

    def test_empty_extra_diagnostic_treated_as_falsy(self):
        """Empty string extra_diagnostic is falsy → not appended (no trailing space)."""
        arr = np.array([np.nan])
        with pytest.raises(ValueError) as excinfo:
            assert_finite_array(arr, name="x", extra_diagnostic="")
        # Empty string is falsy → message should NOT have a trailing space + empty append
        assert excinfo.value.args[0].rstrip().endswith("input invariant violation.")

    # ---------------------- CONSISTENCY WITH MIGRATIONS ---------------------

    def test_message_format_matches_migration_callers(self):
        """The message format is exactly the contract that 7 migration sites
        across hft-contracts/label_factory.py + lob-model-trainer +
        lob-backtester rely on. Format: '<name>: array contains <N> NaN,
        <M> Inf out of <T> total — input invariant violation.[ <extra>]'.
        """
        arr = np.array([np.nan, np.inf])
        with pytest.raises(ValueError) as excinfo:
            assert_finite_array(arr, name="X")
        msg = excinfo.value.args[0]
        # Locked contract — break this assertion to break the 7 callers.
        assert "X: array contains 1 NaN, 1 Inf out of 2 total — input invariant violation." in msg
