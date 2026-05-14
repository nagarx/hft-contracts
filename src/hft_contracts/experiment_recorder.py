"""
SSoT for composing ExperimentRecord from on-disk training artifacts.

Phase 8D / #PY-223 (2026-05-14):
=================================

Closes the R-17a-class direct-trainer ~26% invisibility class by providing
a SINGLE function consumed by BOTH:

1. **hft-ops orchestrator path** (cli.py::_record_experiment, Phase 2 of cycle):
   Orchestrator harvests ``StageResult.captured_metrics`` + ``manifest`` +
   ``paths`` into the helper's inputs, then delegates to
   :func:`record_from_artifacts`.

2. **lob-model-trainer direct path** (scripts/train.py --register-to-ledger,
   Phase 3 of cycle): Trainer harvests its OWN artifacts (``signal_metadata.json``,
   ``test_metrics.json``, resolved config dict) into the helper's inputs, then
   delegates to :func:`record_from_artifacts`.

Architectural fit:

- Lives in hft-contracts (leaf SSoT) — both hft-ops AND lob-model-trainer
  already depend on hft-contracts. NO circular deps; NO new repo edges.
- Reuses existing SSoTs: :class:`ExperimentRecord` + :func:`build_provenance`
  + :func:`compute_experiment_provenance_hash` +
  :data:`atomic_write_json` + :data:`CONTENT_HASH_RE`.
- Migrates the trust-column harvester logic (previously cli-local at
  ``hft-ops/cli.py:104-208`` ``_HarvestedTrustColumns`` + ``_harvest_trust_columns``,
  Cluster Z Closure C 2026-05-11) to this module — same semantics, now
  reusable from train.py.

Zero class-of-divergence risk: single ExperimentRecord construction site;
single Phase Y composer call site; single ledger-write atomic discipline.

Per hft-rules §0 (reuse-first): every dependency is already a hft-contracts
SSoT — no new primitive created.

DESIGN NOTES:

- **Trust-column harvest sources**: caller passes EITHER ``signal_metadata_path``
  (load from disk; train.py path) OR ``captured_metrics_for_trust`` (in-memory
  dict; hft-ops orchestrator path). Mutually exclusive — raises ValueError if both.

- **model_config_hash injection**: the Phase Y composer reads from
  ``record.training_config["model_config_hash"]`` (nested, NOT top-level on
  the record). The harvester mutates a local copy of the caller's
  ``training_config`` dict to inject the harvested SHA before record
  construction — preserves Phase Y composability.

- **Phase Y composer behavior**: ``compute_experiment_provenance_hash``
  returns None when ANY of the 4 components is missing (graceful degradation
  for pre-Phase-II/4.4c.4/X.1v2/V.A.4 legacy records). Set
  ``require_complete_provenance=True`` to opt-into fail-loud — raises
  ValueError listing missing + invalid-format component names. Mirrors the
  REFINED-PLUS Sub-cycle 2 ``required`` parameter convention.

- **Atomic ledger write**: when ``ledger_path`` is set, writes to
  ``<ledger_path>/records/<experiment_id>.json`` via
  :meth:`ExperimentRecord.save` (which uses :data:`atomic_write_json` SSoT
  internally — tmp + fsync + os.replace). NEVER mutates ``index.json``;
  hft-ops ledger.py's ``_save_index`` auto-rebuilds on next access via the
  Phase 8B INDEX_SCHEMA_VERSION envelope.

- **Cross-repo path resolution**: caller provides ``ledger_path`` as an
  absolute Path. NO implicit ``../hft-ops/`` discovery — fail-loud if the
  path doesn't exist (per hft-rules §5).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hft_contracts.experiment_record import (
    ExperimentRecord,
    ProvenanceDiagnostic,
    compute_experiment_provenance_hash,
    diagnose_provenance_completeness,
)
from hft_contracts.provenance import build_provenance
from hft_contracts.signal_manifest import CONTENT_HASH_RE

_logger = logging.getLogger(__name__)


@dataclass
class HarvestedTrustColumns:
    """Phase Y composer trust-column harvest results.

    Migrated from ``hft-ops/cli.py::_HarvestedTrustColumns`` (Cluster Z
    Closure C 2026-05-11) as part of #PY-223 SSoT extraction (Phase 8D
    2026-05-14).

    Distinguishes three input states for each key:

    * **absent** — field stays ``None`` silently (valid: stage
      skipped/disabled/legacy artifact).
    * **present + valid** — field populated.
    * **present + invalid format** — field stays ``None``, error appended
      to ``harvest_errors``. Caller MUST emit a WARN log; record still
      persists (observation-tier failure).

    Validators (mirrors the cli-local original — same regexes + same
    structural rules):

    * ``feature_set_ref``: dict with string ``name`` AND ``content_hash``
      fields (mirrors Phase 4 4c.4 signal_metadata schema).
    * ``compatibility_fingerprint`` + ``model_config_hash``: 64-hex
      SHA-256 strings via :data:`CONTENT_HASH_RE` (canonical hft-contracts
      regex, single source of truth — reused, NOT re-implemented).
    * ``signal_export_output_dir``: non-empty string.
    """

    feature_set_ref: Optional[Dict[str, str]] = None
    compatibility_fingerprint: Optional[str] = None
    model_config_hash: Optional[str] = None
    signal_export_output_dir: Optional[str] = None
    harvest_errors: List[str] = field(default_factory=list)


def harvest_trust_columns(
    captured_metrics: Dict[str, Any],
) -> HarvestedTrustColumns:
    """Validate-and-harvest 4 Phase Y trust columns from a captured_metrics dict.

    For hft-ops consumer: ``captured_metrics`` = ``result.captured_metrics``
    from the signal_export StageResult. For train.py consumer: caller
    constructs a dict from ``signal_metadata.json`` top-level keys (helper
    :func:`harvest_trust_columns_from_signal_metadata` automates this).

    Args:
        captured_metrics: dict mapping the 4 trust-column field names to
            their raw values. Missing keys → None on the result; invalid
            formats → harvest_errors entry.

    Returns:
        :class:`HarvestedTrustColumns` with populated fields + harvest_errors
        list. Never raises — observation-tier failures degrade gracefully
        with diagnostic in ``harvest_errors``.
    """
    out = HarvestedTrustColumns()

    # feature_set_ref: nested dict {name, content_hash} (Phase 4 4c.4).
    raw_ref = captured_metrics.get("feature_set_ref")
    if raw_ref is not None:
        if isinstance(raw_ref, dict):
            name = raw_ref.get("name")
            content_hash = raw_ref.get("content_hash")
            if isinstance(name, str) and isinstance(content_hash, str):
                out.feature_set_ref = {
                    "name": name,
                    "content_hash": content_hash,
                }
            else:
                out.harvest_errors.append(
                    f"feature_set_ref dict has non-string name "
                    f"({type(name).__name__}) or content_hash "
                    f"({type(content_hash).__name__})"
                )
        else:
            out.harvest_errors.append(
                f"feature_set_ref not a dict (got "
                f"{type(raw_ref).__name__})"
            )

    # compatibility_fingerprint + model_config_hash: 64-hex SHA-256.
    # CONTENT_HASH_RE is the canonical hft-contracts regex.
    for fld in ("compatibility_fingerprint", "model_config_hash"):
        raw = captured_metrics.get(fld)
        if raw is not None:
            if isinstance(raw, str) and CONTENT_HASH_RE.match(raw):
                setattr(out, fld, raw)
            else:
                out.harvest_errors.append(
                    f"{fld} not a 64-hex SHA-256 string (got "
                    f"{type(raw).__name__}, value={raw!r})"
                )

    # signal_export_output_dir: non-empty string (Phase V.1 L1.2).
    raw_dir = captured_metrics.get("signal_export_output_dir")
    if raw_dir is not None:
        if isinstance(raw_dir, str) and raw_dir:
            out.signal_export_output_dir = raw_dir
        else:
            out.harvest_errors.append(
                f"signal_export_output_dir invalid (got "
                f"{type(raw_dir).__name__}, value={raw_dir!r})"
            )

    return out


def harvest_trust_columns_from_signal_metadata(
    signal_metadata_path: Path,
) -> HarvestedTrustColumns:
    """Read signal_metadata.json + harvest the 4 trust columns.

    Convenience helper for the train.py consumer path. The producer-side
    ``signal_metadata.json`` schema (post-Phase-II + Phase 4.4c.4 + Phase
    Y deployment) places the 4 trust-column fields at top level:

    * ``feature_set_ref``: ``{"name": str, "content_hash": str}`` or None
    * ``compatibility_fingerprint``: 64-hex string or None
    * ``model_config_hash``: 64-hex string or None

    Note: ``signal_export_output_dir`` is NOT in the producer schema (it's
    a run-time-captured absolute path harvested orchestrator-side from
    ``SignalExportStage`` output_dir). Direct-trainer callers can pass it
    via ``record_from_artifacts(signal_export_output_dir_override=...)``.

    Graceful degradation:

    * File missing → empty result + 1 harvest_error.
    * File malformed JSON → empty result + 1 harvest_error.
    * File present + valid JSON but root not dict → empty result + 1 harvest_error.
    * File present + valid + missing some trust fields → fields stay None
      (no harvest_error — absent is valid for legacy artifacts).
    """
    if not signal_metadata_path.exists():
        return HarvestedTrustColumns(
            harvest_errors=[
                f"signal_metadata.json not found at {signal_metadata_path}"
            ]
        )

    try:
        with signal_metadata_path.open("r") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return HarvestedTrustColumns(
            harvest_errors=[
                f"failed to read signal_metadata.json at {signal_metadata_path}: "
                f"{type(exc).__name__}: {exc}"
            ]
        )

    if not isinstance(metadata, dict):
        return HarvestedTrustColumns(
            harvest_errors=[
                f"signal_metadata.json root not a dict (got "
                f"{type(metadata).__name__}) at {signal_metadata_path}"
            ]
        )

    # Project signal_metadata top-level keys into the same shape that
    # ``harvest_trust_columns`` expects from captured_metrics. By Phase II
    # + Phase 4.4c.4 + Phase Y producer-side design, the key names match.
    captured_shape = {
        "feature_set_ref": metadata.get("feature_set_ref"),
        "compatibility_fingerprint": metadata.get("compatibility_fingerprint"),
        "model_config_hash": metadata.get("model_config_hash"),
        # signal_export_output_dir NOT in producer schema — caller passes via override.
        "signal_export_output_dir": None,
    }
    return harvest_trust_columns(captured_shape)


def record_from_artifacts(
    *,
    # Required identity fields
    name: str,
    pipeline_root: Path,
    contract_version: str,
    fingerprint: str,
    # Trust-column source (mutually exclusive)
    signal_metadata_path: Optional[Path] = None,
    captured_metrics_for_trust: Optional[Dict[str, Any]] = None,
    # Trainer artifacts
    training_metrics: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None,
    # Provenance inputs (passed through to build_provenance)
    manifest_path: Optional[Path] = None,
    extractor_config_path: Optional[Path] = None,
    trainer_config_path: Optional[Path] = None,
    trainer_config_dict: Optional[Dict[str, Any]] = None,
    data_dir: Optional[Path] = None,
    # Orchestrator-only fields (None for direct-trainer path)
    gate_reports: Optional[Dict[str, Dict[str, Any]]] = None,
    cache_info: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    stages_completed: Optional[List[str]] = None,
    # Status + metadata
    status: str = "completed",
    duration_seconds: float = 0.0,
    signal_export_output_dir_override: Optional[str] = None,
    tags: Optional[List[str]] = None,
    hypothesis: str = "",
    description: str = "",
    record_type: str = "training",
    # Persistence
    ledger_path: Optional[Path] = None,
    experiment_id_override: Optional[str] = None,
    # Phase Y behavior
    require_complete_provenance: bool = False,
) -> ExperimentRecord:
    """Compose :class:`ExperimentRecord` from on-disk training artifacts.

    SSoT helper consumed by BOTH:

    * **hft-ops orchestrator** (``cli.py::_record_experiment``)
    * **lob-model-trainer direct path** (``scripts/train.py
      --register-to-ledger``)

    Closes #PY-223 (R-17a-class direct-trainer ~26% invisibility).

    Args:
        name: Human-readable experiment name (from config.name or
            manifest.experiment.name). Combined with timestamp +
            fingerprint[:8] to form ``experiment_id``.
        pipeline_root: Path to monorepo root (used for git capture
            during provenance build).
        contract_version: Pipeline schema_version at time of run
            (typically ``hft_contracts.SCHEMA_VERSION``).
        fingerprint: SHA-256 of resolved config for dedup. Caller computes
            via ``hft_ops.ledger.dedup`` (orchestrator) OR via
            ``hft_contracts.canonical_hash`` on the trainer-config dict
            (direct-trainer). Both produce valid 64-hex SHA-256.
        signal_metadata_path: Optional path to ``signal_metadata.json``
            for trust-column harvest. Mutually exclusive with
            ``captured_metrics_for_trust``.
        captured_metrics_for_trust: Optional in-memory dict (e.g.,
            ``StageResult.captured_metrics`` from signal_export stage)
            for trust-column harvest. Mutually exclusive with
            ``signal_metadata_path``.
        training_metrics: Flat dict of training/test metrics (final
            scalar values; per-class arrays are dropped by the
            ``index_entry()`` whitelist anyway).
        training_config: Resolved trainer config dict. The harvested
            ``model_config_hash`` is INJECTED into this dict (mutated copy)
            before record construction so the Phase Y composer reads it
            from the canonical nested location.
        manifest_path/extractor_config_path/trainer_config_path/
            trainer_config_dict/data_dir: Passed through to
            :func:`build_provenance`. Mutually exclusive: caller supplies
            EITHER ``trainer_config_path`` (legacy wrapper) OR
            ``trainer_config_dict`` (Phase 1 inline pattern), not both.
        gate_reports: Orchestrator-only — stage-name → gate report dict.
            Defaults to empty for direct-trainer path.
        cache_info: Orchestrator-only — extraction cache observability.
            Defaults to empty for direct-trainer path.
        artifacts: Optional list of content-addressed artifact refs.
            Used by hft-ops Phase 8C-α post-stage routing.
        stages_completed: List of completed stage names. Direct-trainer
            path typically supplies ``["training"]`` (and optionally
            ``"signal_export"``).
        status: ``"completed"`` | ``"failed"`` | ``"partial"``.
        duration_seconds: Wall-clock duration of the run.
        signal_export_output_dir_override: Phase V.1 L1.2 — absolute path
            captured at run-time. Caller-override (orchestrator passes
            from SignalExportStage; direct-trainer passes resolved
            export-output-dir if it ran signal export).
        tags/hypothesis/description: Experiment metadata.
        record_type: One of :class:`RecordType` values (default
            ``"training"``).
        ledger_path: If set, atomic-write the record to
            ``<ledger_path>/records/<experiment_id>.json`` via the
            :data:`atomic_write_json` SSoT. NEVER touches
            ``<ledger_path>/index.json`` — hft-ops auto-rebuilds via
            Phase 8B envelope mismatch.
        experiment_id_override: If set, used as-is instead of the default
            ``{name}_{timestamp}_{fingerprint[:8]}``. Useful for retros.
        require_complete_provenance: If True, Phase Y composer raises
            :class:`ValueError` instead of returning None on missing or
            invalid-format components. Mirrors the REFINED-PLUS Sub-cycle 2
            ``required`` parameter convention.

    Returns:
        Fully composed :class:`ExperimentRecord` with
        ``experiment_provenance_hash`` set (or None if
        ``require_complete_provenance=False`` AND any of the 4 sources is
        missing/invalid — graceful degradation, WARN logged).

    Raises:
        ValueError: If ``signal_metadata_path`` and
            ``captured_metrics_for_trust`` are both provided, OR if
            ``require_complete_provenance=True`` and Phase Y sources are
            incomplete.

    Per hft-rules §0 (reuse-first): every dependency is already a
    hft-contracts SSoT — no new primitive created.
    """
    # Mutually exclusive trust-column source — fail-loud per hft-rules §5.
    if signal_metadata_path is not None and captured_metrics_for_trust is not None:
        raise ValueError(
            "record_from_artifacts: signal_metadata_path and "
            "captured_metrics_for_trust are mutually exclusive — harvest "
            "trust columns from ONE source per call, not both."
        )

    # Fail-loud fingerprint format validation (per hft-rules §5 + architect
    # pre-commit MEDIUM-1 review 2026-05-14): the fingerprint is required to
    # be a 64-hex lowercase SHA-256 string (CONTENT_HASH_RE convention).
    # Without this gate, an upstream caller bug could silently land
    # malformed fingerprints in the ledger — invisible to downstream
    # ``hft-ops ledger list`` filter queries. Test
    # ``test_minimal_required_only`` etc. use VALID_HASH = "a" * 64 so this
    # assertion does not affect existing tests.
    if not CONTENT_HASH_RE.match(fingerprint):
        raise ValueError(
            f"record_from_artifacts: fingerprint must be a 64-hex lowercase "
            f"SHA-256 string (CONTENT_HASH_RE convention); got "
            f"{fingerprint!r}. Compute via hft_ops.ledger.dedup (orchestrator) "
            f"OR hft_contracts.canonical_hash.sha256_hex on the resolved "
            f"trainer config dict (direct-trainer path)."
        )

    # Resolve experiment_id
    now = datetime.now(timezone.utc)
    if experiment_id_override is not None:
        experiment_id = experiment_id_override
    else:
        timestamp = now.strftime("%Y%m%dT%H%M%S")
        experiment_id = f"{name}_{timestamp}_{fingerprint[:8]}"

    # Build provenance via SSoT (handles git capture, file hashing,
    # data_dir manifest hashing). Mutually-exclusive trainer arg validation
    # happens inside build_provenance.
    provenance = build_provenance(
        pipeline_root,
        manifest_path=manifest_path,
        extractor_config_path=extractor_config_path,
        trainer_config_path=trainer_config_path,
        trainer_config_dict=trainer_config_dict,
        data_dir=data_dir,
        contract_version=contract_version,
    )

    # Harvest trust columns from one of two sources (or default empty).
    if signal_metadata_path is not None:
        trust = harvest_trust_columns_from_signal_metadata(signal_metadata_path)
    elif captured_metrics_for_trust is not None:
        trust = harvest_trust_columns(captured_metrics_for_trust)
    else:
        trust = HarvestedTrustColumns()

    if trust.harvest_errors:
        _logger.warning(
            "Trust-column harvest errors on experiment %s: %s",
            experiment_id,
            "; ".join(trust.harvest_errors),
        )

    # Inject model_config_hash into training_config (nested location is
    # what Phase Y composer reads — see compute_experiment_provenance_hash).
    # Always shallow-copy first to avoid mutating caller's dict.
    training_config = dict(training_config) if training_config is not None else {}
    if trust.model_config_hash is not None:
        training_config["model_config_hash"] = trust.model_config_hash

    # Resolve signal_export_output_dir: caller override > trust harvest > None.
    sed = signal_export_output_dir_override or trust.signal_export_output_dir

    # Construct ExperimentRecord using hft-contracts SSoT dataclass.
    record = ExperimentRecord(
        experiment_id=experiment_id,
        name=name,
        manifest_path=str(manifest_path) if manifest_path else "",
        fingerprint=fingerprint,
        feature_set_ref=trust.feature_set_ref,
        compatibility_fingerprint=trust.compatibility_fingerprint,
        signal_export_output_dir=sed,
        provenance=provenance,
        contract_version=contract_version,
        training_config=training_config,
        training_metrics=training_metrics or {},
        gate_reports=gate_reports or {},
        cache_info=cache_info or {},
        artifacts=artifacts or [],
        tags=tags or [],
        hypothesis=hypothesis,
        description=description,
        created_at=now.isoformat(),
        duration_seconds=duration_seconds,
        status=status,
        stages_completed=stages_completed or [],
        record_type=record_type,
    )

    # Phase Y composer — post-construction mutation per the cli.py:783-785
    # canonical pattern. Composer reads the 4 fields off the constructed
    # record; cannot be passed at construction time.
    required_set = None
    if require_complete_provenance:
        required_set = ProvenanceDiagnostic.COMPONENT_NAMES

    provenance_hash = compute_experiment_provenance_hash(
        record,
        required=required_set,
    )
    if provenance_hash is not None:
        record.experiment_provenance_hash = provenance_hash
    else:
        # require_complete_provenance=False path with missing sources.
        # Diagnose for operator visibility per hft-rules §8.
        diagnostic = diagnose_provenance_completeness(record)
        _logger.warning(
            "experiment_provenance_hash composition skipped for %s — "
            "missing=%s, invalid_format=%s. Record persists with "
            "experiment_provenance_hash=None (graceful degradation). "
            "Common causes: legacy signal_metadata.json (pre-Phase-Y deployment), "
            "training-only run (no signal_export), or operator-set fingerprint "
            "from non-canonical source.",
            experiment_id,
            sorted(diagnostic.missing) or "none",
            sorted(diagnostic.invalid_format) or "none",
        )

    # Atomic ledger write — caller-controlled.
    if ledger_path is not None:
        if not ledger_path.exists():
            raise ValueError(
                f"record_from_artifacts: ledger_path {ledger_path} does not "
                f"exist. Caller must provide an existing directory; this "
                f"helper does NOT auto-create the parent (would silently "
                f"hide operator typos per hft-rules §5)."
            )
        records_dir = ledger_path / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        record_path = records_dir / f"{experiment_id}.json"
        record.save(record_path)
        _logger.info(
            "Registered ExperimentRecord to ledger: %s (epH=%s)",
            record_path,
            record.experiment_provenance_hash or "None",
        )

    return record


__all__ = [
    "HarvestedTrustColumns",
    "harvest_trust_columns",
    "harvest_trust_columns_from_signal_metadata",
    "record_from_artifacts",
]
