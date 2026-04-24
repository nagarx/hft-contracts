"""Tests for ``hft_contracts.timestamp_utils`` (Phase A.5.1, 2026-04-24).

Locks the ISO-8601 UTC-aware parsing + cutoff-comparison SSoT contract.
Prevents recurrence of the silent-wrong-result bug that motivated A.5.1:
``hft_ops.stages.signal_export`` compared ISO-8601 strings lexicographically,
producing wrong cutoff decisions for non-UTC timestamps.

12 tests covering:
    1.  Naive timestamps interpreted as UTC.
    2.  Explicit UTC offset crossing midnight (the exact bug scenario).
    3.  ``Z`` suffix normalization (Python < 3.11 compat path).
    4.  Malformed input raises ValueError.
    5.  Non-string input raises TypeError.
    6.  None raises TypeError (not silently None-checked).
    7.  Empty string raises.
    8.  Fractional seconds preserved.
    9-11.  is_after_cutoff: pre-cutoff, post-cutoff, exact boundary.
    12.  Epoch edge (1970-01-01T00:00:00).
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from hft_contracts.timestamp_utils import is_after_cutoff, parse_iso8601_utc


class TestParseIso8601Utc:
    def test_naive_timestamp_interpreted_as_utc(self):
        """Naive timestamps (no offset) are interpreted as UTC per pipeline convention.

        This is the common case for JSON artifacts written by Python callers
        using ``datetime.utcnow().isoformat()`` — the output has no offset.
        hft-rules §3 canonical convention is UTC; this locks it explicitly.
        """
        dt = parse_iso8601_utc("2026-04-24T12:00:00")
        assert dt.tzinfo == timezone.utc
        assert dt == datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc)

    def test_non_utc_offset_converts_across_midnight(self):
        """The exact bug scenario that motivated this module.

        ``2026-04-22T23:59:00-05:00`` is ``2026-04-23T04:59:00+00:00`` in UTC
        — strictly after the cutoff ``2026-04-23``. Lexicographic comparison
        would produce ``False``; proper UTC conversion yields ``True``.
        """
        dt = parse_iso8601_utc("2026-04-22T23:59:00-05:00")
        # 23:59 CDT (Central Daylight) = 04:59 UTC NEXT DAY
        expected = datetime(2026, 4, 23, 4, 59, 0, tzinfo=timezone.utc)
        assert dt == expected
        assert dt.tzinfo == timezone.utc

    def test_z_suffix_normalized_to_plus_zero(self):
        """``Z`` → ``+00:00`` defensively (Python < 3.11 compat).

        Python 3.11+ accepts ``Z`` directly in ``datetime.fromisoformat``,
        but the replace is a zero-cost defensive step for older targets.
        """
        dt = parse_iso8601_utc("2026-04-24T12:00:00Z")
        assert dt.tzinfo == timezone.utc
        assert dt == datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc)

    def test_malformed_string_raises_value_error(self):
        """Malformed ISO-8601 input is an operator bug → fail-loud."""
        with pytest.raises(ValueError, match="malformed timestamp"):
            parse_iso8601_utc("not-a-timestamp")

    def test_none_input_raises_type_error(self):
        """``None`` is a programming error — reject with TypeError, not silent handling."""
        with pytest.raises(TypeError, match="expects str"):
            parse_iso8601_utc(None)  # type: ignore[arg-type]

    def test_bytes_input_raises_type_error(self):
        """Bytes are a programming error — don't silently decode()."""
        with pytest.raises(TypeError, match="expects str"):
            parse_iso8601_utc(b"2026-04-24T12:00:00")  # type: ignore[arg-type]

    def test_int_input_raises_type_error(self):
        """Int epoch seconds are a programming error — wrong function."""
        with pytest.raises(TypeError, match="expects str"):
            parse_iso8601_utc(1_714_000_000)  # type: ignore[arg-type]

    def test_empty_string_raises_value_error(self):
        """Empty string is malformed."""
        with pytest.raises(ValueError, match="malformed timestamp"):
            parse_iso8601_utc("")

    def test_fractional_seconds_preserved(self):
        """Microsecond precision must survive the round-trip."""
        dt = parse_iso8601_utc("2026-04-24T12:00:00.123456")
        assert dt.microsecond == 123456
        assert dt.tzinfo == timezone.utc

    def test_epoch_edge(self):
        """Unix epoch boundary — historically a common off-by-one trap."""
        dt = parse_iso8601_utc("1970-01-01T00:00:00Z")
        assert dt == datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert dt.timestamp() == 0.0


class TestIsAfterCutoff:
    def test_pre_cutoff_returns_false(self):
        """Timestamp before cutoff → False."""
        assert is_after_cutoff("2026-04-22T12:00:00Z", "2026-04-23") is False

    def test_post_cutoff_returns_true(self):
        """Timestamp after cutoff → True. The CDT-crossing-midnight case
        that lexicographic comparison would have gotten WRONG."""
        # 23:59 on April 22 in CDT = 04:59 April 23 UTC (post-cutoff).
        assert (
            is_after_cutoff("2026-04-22T23:59:00-05:00", "2026-04-23") is True
        )
        # Lexicographic comparison on the raw string would have been False:
        assert ("2026-04-22T23:59:00-05:00" >= "2026-04-23") is False
        # Demonstrates the bug-class that is_after_cutoff retires.

    def test_exact_boundary_returns_true(self):
        """Cutoff semantics: ``>=`` (inclusive) — matches the V.A.4
        ``FINGERPRINT_REQUIRED_AFTER_ISO`` intent."""
        assert is_after_cutoff("2026-04-23T00:00:00Z", "2026-04-23") is True

    def test_malformed_timestamp_propagates(self):
        """Helper propagates parse_iso8601_utc errors — no silent False."""
        with pytest.raises(ValueError, match="malformed timestamp"):
            is_after_cutoff("not-a-timestamp", "2026-04-23")

    def test_malformed_cutoff_propagates(self):
        """Cutoff malformed → operator bug, raise."""
        with pytest.raises(ValueError, match="malformed timestamp"):
            is_after_cutoff("2026-04-24T00:00:00Z", "garbage")
