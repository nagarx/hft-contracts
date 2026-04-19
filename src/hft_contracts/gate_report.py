"""Gate-report dict contract — convention for cross-stage gate outputs.

Phase 7 Stage 7.4 Round 5 (2026-04-20). Every stage runner that emits a
regression / correctness / performance gate into
``StageResult.captured_metrics["gate_report"]`` MUST conform to the
``GateReportDict`` shape below. The dict lands under
``ExperimentRecord.gate_reports[stage_name]`` and is projected into
``index_entry()["gate_reports"][stage_name]["status"]`` for fast ledger
queries.

## Convention

- **``status: str``** (required) — one of :data:`GATE_STATUS_VALUES`
  (lower-case). Consumers filter on this key.
- **``summary: str``** (optional) — human-readable one-liner, ≤256
  chars after truncation in ``index_entry``.
- **Additional fields** — stage-specific. Document them in the
  emitting runner's own ``GateReport`` dataclass docstring.

## Why a TypedDict, not a Protocol (yet)

At Round 5 the pipeline has 2 gate types: ``validation`` (IC gate) and
``post_training_gate`` (regression gate). A TypedDict documents the
convention without forcing a runtime refactor in the evaluator's
existing ``fast_gate.GateReport`` dataclass (which has a ``verdict``
field, not ``status``). The ``validation`` stage adapter
(``hft_ops.stages.validation`` around line 229) lowercases
``verdict`` → ``status`` before writing to ``captured_metrics``.

When a third gate ships (Phase 8 ``post_backtest_gate``), upgrade to a
full ``Protocol`` or ABC in this module + rename every dataclass field
to match. Until then, the TypedDict is the minimum-invasive contract.

## Post-Round-5 state

Validator audits converged on three findings driving this module:

1. **Convention drift was LIVE** — ``validation`` emitted
   ``"verdict": "PASS"/"FAIL"`` (upper-case); ``post_training_gate``
   emitted ``"status": "pass"/"warn"/"abort"`` (lower-case). The
   ``index_entry()`` projection coalesced both via
   ``report.get("status") or report.get("verdict")`` — proving the
   drift was already a maintenance cost, not hypothetical.
2. **Casing inconsistency leaked** into ``ledger list --gate-status
   warn`` queries: post-training matches but validation does not.
3. **"Defer until 3 gates"** was a weak bet — drift accumulated
   after 2 gates shipped, so adopting a lightweight contract now
   prevents a third gate from reinforcing the pattern.

The TypedDict does NOT enforce at runtime (TypedDicts are pure type
hints). Future hardening could add a ``validate_gate_report(d) ->
None`` function that raises on missing ``status`` or invalid value.
"""

from __future__ import annotations

from typing import Any, Dict, TypedDict


# Lower-case gate-status values. Every stage runner must emit one of
# these under the ``status`` key when writing to
# ``captured_metrics["gate_report"]``.
GATE_STATUS_VALUES: frozenset = frozenset({"pass", "warn", "fail", "abort"})


class GateReportDict(TypedDict, total=False):
    """Required shape for ``captured_metrics["gate_report"]`` dicts.

    Only ``status`` is required in practice; ``summary`` is recommended.
    Additional fields are stage-specific and documented in the emitting
    runner's own ``GateReport`` dataclass docstring.

    ``TypedDict`` with ``total=False`` allows additional keys; the
    consumer (cli.py harvest + index_entry projection) is tolerant of
    extra fields. The EMITTER is responsible for including ``status``.
    """

    status: str      # one of GATE_STATUS_VALUES (lower-case)
    summary: str     # human-readable one-liner, ≤256 chars in index


__all__ = ["GATE_STATUS_VALUES", "GateReportDict"]
