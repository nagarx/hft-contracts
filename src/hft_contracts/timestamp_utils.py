"""ISO-8601 UTC-aware timestamp parsing + cutoff comparison.

**Single source of truth** for cross-module timestamp comparisons in the HFT
pipeline. Before Phase A.5 (2026-04-24), `hft_ops.stages.signal_export`
compared ISO-8601 strings lexicographically (`exported_at >= cutoff`) — safe
ONLY when both sides are in UTC with no offset suffix (``+00:00`` / ``Z``).

The lexicographic comparison fails silently for timestamps with non-UTC
offsets::

    >>> "2026-04-22T23:59:00-05:00" >= "2026-04-23"
    False

…even though the first timestamp is ``2026-04-23T04:59:00+00:00`` in UTC
(strictly AFTER the cutoff). Operators shipping from a non-UTC JVM /
Node / Rust producer would see post-cutoff manifests silently fall into
the pre-cutoff branch.

Fix: go through ``parse_iso8601_utc`` which normalizes every input to a
timezone-aware ``datetime`` in UTC. ``is_after_cutoff`` wraps the common
comparison pattern.

Design principles (hft-rules):
    §1 Single source of truth: every timestamp-comparison call site in the
       monorepo routes through these helpers.
    §2 Exact representation: no silent float conversion; ``datetime``
       preserves microsecond precision.
    §3 Explicit units + timezone: return value is ALWAYS timezone-aware
       UTC. No naive datetimes escape this module.
    §5 Fail-fast: malformed input raises ``ValueError`` with a diagnostic
       citing the offending bytes, not a silent ``None``.
    §8 Never silently drop / clamp / "fix": if an input is malformed we
       raise; if it's ambiguous (naive) we document the assumption (UTC,
       per pipeline convention) and re-raise on type mismatch.

Consumers (added in Phase A.5 cycle):
    - ``hft_ops.stages.signal_export.FINGERPRINT_REQUIRED_AFTER_ISO`` cutoff
      check (replaces the lexicographic comparison).
    - Future: ``hft_ops.ledger`` date-range queries, backtest date gates,
      sweep date-range filters — all route through here.

Not consumed by::

    - Rust extractor / profiler (cargo deps are separate).
    - Databento downloader (uses its own ``datetime.fromisoformat`` because
      it only ever sees UTC timestamps from the API).

Frozen contract (never change without a new symbol name):

.. code-block:: python

    parse_iso8601_utc(ts: str) -> datetime   # always tzinfo=timezone.utc
    is_after_cutoff(ts: str, cutoff_iso: str) -> bool  # strict >= in UTC
"""

from __future__ import annotations

from datetime import datetime, timezone

__all__ = ["parse_iso8601_utc", "is_after_cutoff"]


def parse_iso8601_utc(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp to a timezone-aware UTC ``datetime``.

    Handles:
        - Naive timestamps (no offset): interpreted as UTC per the pipeline's
          canonical convention (hft-rules §3). A WARN log is NOT emitted —
          naive timestamps are common in JSON artifacts written by Python
          `datetime.utcnow()` callers, and flipping to strict would break
          cross-cycle manifest compatibility. The assumption is deliberate
          and documented here.
        - ``Z`` suffix (e.g., ``"2026-04-24T12:00:00Z"``): normalized to
          ``"+00:00"`` defensively for Python < 3.11 compat
          (``datetime.fromisoformat`` gained direct ``Z`` support in 3.11).
        - Explicit offsets (e.g., ``"2026-04-24T08:00:00-04:00"``):
          converted to UTC via ``.astimezone(timezone.utc)``.
        - Fractional seconds (``.123456``): preserved to microsecond precision.

    Args:
        ts: ISO-8601 timestamp string. Must be a ``str`` — no coercion
            from ``bytes`` / ``int`` / ``None``.

    Returns:
        Timezone-aware ``datetime`` with ``tzinfo == timezone.utc``.

    Raises:
        TypeError: when ``ts`` is not a ``str`` (e.g., ``None``, ``bytes``,
            ``int``). Prevents silent-coercion bugs.
        ValueError: when ``ts`` is not parseable as ISO-8601. The error
            message cites the offending input and the expected format
            for operator triage.
    """
    if not isinstance(ts, str):
        raise TypeError(
            f"parse_iso8601_utc expects str; got {type(ts).__name__} "
            f"(value={ts!r}). Coerce at the boundary; do not pass None/bytes/int."
        )
    normalized = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"parse_iso8601_utc: malformed timestamp {ts!r} — "
            f"expected ISO-8601 (e.g. '2026-04-24T12:00:00+00:00', "
            f"'2026-04-24T12:00:00Z', or naive 'YYYY-MM-DDTHH:MM:SS'). "
            f"Underlying error: {exc}"
        ) from exc
    if dt.tzinfo is None:
        # Naive → UTC (pipeline canonical convention per hft-rules §3).
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        # Aware → convert to UTC.
        dt = dt.astimezone(timezone.utc)
    return dt


def is_after_cutoff(timestamp: str, cutoff_iso: str) -> bool:
    """Return True iff ``timestamp`` is on or after ``cutoff_iso`` in UTC.

    Use this helper when you would otherwise write
    ``timestamp >= cutoff_iso`` — lexicographic string comparison silently
    fails on timestamps with non-UTC offsets (see module docstring).

    Args:
        timestamp: ISO-8601 timestamp string. Parsed via
            ``parse_iso8601_utc``; TypeError / ValueError propagate.
        cutoff_iso: Cutoff ISO-8601 timestamp string. Must parse via
            ``parse_iso8601_utc``. Typically a date-only string like
            ``"2026-04-23"`` (interpreted as ``"2026-04-23T00:00:00+00:00"``).

    Returns:
        ``True`` iff ``timestamp >= cutoff_iso`` after both are normalized
        to UTC. Comparison uses ``datetime.__ge__`` (microsecond precision).

    Raises:
        TypeError, ValueError: propagate from ``parse_iso8601_utc`` for
            either argument. A malformed cutoff is a programming error,
            NOT a runtime input — the caller should have a compile-time
            constant.
    """
    return parse_iso8601_utc(timestamp) >= parse_iso8601_utc(cutoff_iso)
