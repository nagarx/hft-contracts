"""
Cross-module validation utilities for the HFT pipeline contract.

These functions are designed to be called at system boundaries (dataset load,
export verification) per RULE.md §8: "Validate inputs at system boundaries.
Trust internal code."

Validation hierarchy:
    validate_export_contract()      — full boundary check (calls all below)
    validate_schema_version()       — schema_version matches contract
    validate_normalization_not_applied() — Rust-side norm is disabled
    validate_metadata_completeness() — all required fields present
    validate_label_encoding()       — label strategy matches contract
    validate_provenance_present()   — provenance block is present
    validate_day_metadata()         — per-day export check (wraps the contract)
    validate_export_dir()           — directory-level manifest<->disk + cross-day
                                       schema/commit uniformity (composes
                                       validate_day_metadata; 2026-05-29)
    assert_finite_array()           — fail-loud on NaN/Inf in numpy arrays
                                       (#PY-63 SSoT extraction 2026-05-07)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from hft_contracts._generated import (
    EXPORT_METADATA_NORMALIZATION_FIELDS,
    EXPORT_METADATA_PROVENANCE_FIELDS,
    EXPORT_METADATA_REQUIRED_FIELDS,
    FEATURE_COUNT,
    FULL_FEATURE_COUNT,
    SCHEMA_VERSION,
    SCHEMA_VERSION_FLOAT,
    OFF_EXCHANGE_FEATURE_COUNT,
    OFF_EXCHANGE_SCHEMA_VERSION,
)
from hft_contracts.labels import get_contract, is_regression_strategy, RegressionLabelContract


class ContractError(Exception):
    """Raised when exported data violates the pipeline contract."""
    pass


def validate_feature_indices(
    indices: Sequence[int],
    source_feature_count: int,
    name: str = "custom",
) -> None:
    """
    Validate that feature indices are valid for a given feature count.

    Args:
        indices: Sequence of feature indices to validate.
        source_feature_count: Number of features in source data.
        name: Name for error messages.

    Raises:
        ValueError: If any index is invalid (negative, out of bounds, or duplicated).

    Contract (RULE.md §1):
        - indices must be non-empty
        - all indices must be in [0, source_feature_count)
        - no duplicates allowed
    """
    if not indices:
        raise ValueError(f"Feature indices '{name}' cannot be empty")

    idx_set = set(indices)
    if len(indices) != len(idx_set):
        duplicates = [i for i in indices if list(indices).count(i) > 1]
        raise ValueError(
            f"Feature indices '{name}' contains duplicates: {set(duplicates)}"
        )

    min_idx = min(indices)
    max_idx = max(indices)

    if min_idx < 0:
        raise ValueError(
            f"Feature indices '{name}' contains negative index: {min_idx}"
        )

    if max_idx >= source_feature_count:
        raise ValueError(
            f"Feature indices '{name}' contains index {max_idx} but source "
            f"only has {source_feature_count} features "
            f"(max valid: {source_feature_count - 1})"
        )


def validate_schema_version(
    metadata: dict,
    *,
    expected_version: str | None = None,
) -> None:
    """
    Validate that export metadata matches the expected schema version.

    Checks both the string field ``schema_version`` and the legacy float
    field at feature index 97 (if present as a top-level field).

    Args:
        metadata: Metadata dict from export (typically from *_metadata.json).
        expected_version: Override expected version. Defaults to contract SCHEMA_VERSION.

    Raises:
        ContractError: If schema version is missing or mismatched.
    """
    expected = expected_version or SCHEMA_VERSION

    exported = metadata.get("schema_version")
    if exported is None:
        raise ContractError(
            "Export metadata missing 'schema_version' field. "
            "Cannot verify contract compatibility. "
            "Re-export data with the latest feature extractor."
        )

    exported_str = str(exported)
    if exported_str != expected:
        raise ContractError(
            f"Export schema version '{exported_str}' != expected '{expected}'. "
            f"Re-export data or update consumer."
        )


def validate_normalization_not_applied(metadata: dict) -> None:
    """
    Verify that Rust-side normalization was NOT applied, so Python-side
    normalization is safe.

    Handles both Rust's ``normalization.applied`` key and the legacy
    ``normalization.normalization_applied`` key for backward compatibility.

    Args:
        metadata: Metadata dict from *_metadata.json

    Raises:
        ContractError: If normalization was already applied upstream.
    """
    norm_info = metadata.get("normalization", {})
    if not isinstance(norm_info, dict):
        return

    applied = norm_info.get("applied", norm_info.get("normalization_applied", False))
    if applied:
        raise ContractError(
            "Data was already normalized by Rust exporter. "
            "Python-side normalization would double-normalize. "
            "Set normalization.strategy='none' in the extractor config."
        )


def validate_metadata_completeness(
    metadata: dict,
    *,
    strict: bool = True,
) -> list[str]:
    """
    Validate that all required fields are present in export metadata.

    Args:
        metadata: Metadata dict from *_metadata.json
        strict: If True, raise ContractError on missing critical fields.
                If False, return list of warnings.

    Returns:
        List of warning messages for missing optional fields.

    Raises:
        ContractError: If strict=True and critical fields are missing.
    """
    warnings: list[str] = []
    critical_missing: list[str] = []

    critical_fields = ("schema_version", "n_features", "n_sequences", "window_size")

    for field in EXPORT_METADATA_REQUIRED_FIELDS:
        if field not in metadata:
            if field in critical_fields:
                critical_missing.append(field)
            else:
                warnings.append(f"Missing recommended metadata field: '{field}'")

    norm_info = metadata.get("normalization", {})
    if isinstance(norm_info, dict):
        for field in EXPORT_METADATA_NORMALIZATION_FIELDS:
            if field not in norm_info:
                warnings.append(f"Missing normalization sub-field: '{field}'")
    elif "normalization" in metadata:
        warnings.append("'normalization' field is not a dict")

    if strict and critical_missing:
        raise ContractError(
            f"Export metadata missing critical fields: {critical_missing}. "
            f"These are required for safe data loading."
        )

    return warnings


def validate_label_encoding(
    metadata: dict,
    expected_strategy: str | None = None,
) -> None:
    """
    Validate that the label encoding in metadata matches the pipeline contract.

    Args:
        metadata: Metadata dict from *_metadata.json
        expected_strategy: Strategy name to validate against. If None,
            reads from metadata's label_strategy field.

    Raises:
        ContractError: If label encoding doesn't match the contract.
    """
    strategy = expected_strategy
    if strategy is None:
        strategy = metadata.get("label_strategy")
        if strategy is None:
            labeling = metadata.get("labeling", {})
            if isinstance(labeling, dict):
                strategy = labeling.get("strategy")

    if strategy is None:
        return

    try:
        contract = get_contract(strategy)
    except ValueError:
        raise ContractError(
            f"Unknown label strategy '{strategy}' in metadata. "
            f"Valid strategies: tlob, triple_barrier, opportunity, regression"
        )

    # Regression: validate dtype, skip classification-specific checks
    if isinstance(contract, RegressionLabelContract):
        encoding = metadata.get("label_encoding", {})
        if not isinstance(encoding, dict):
            labeling = metadata.get("labeling", {})
            if isinstance(labeling, dict):
                encoding = labeling.get("label_encoding", {})
        if isinstance(encoding, dict) and "dtype" in encoding:
            if encoding["dtype"] != contract.dtype:
                raise ContractError(
                    f"Regression label dtype mismatch: "
                    f"metadata has '{encoding['dtype']}', contract requires '{contract.dtype}'"
                )
        return

    # Classification: validate class names match contract
    #
    # Phase X.2.A.2 / #PY-218 (2026-05-14): nested-form fallback mirrors the
    # regression branch at L237-240. When the top-level `label_encoding.values`
    # is not a dict (e.g., the Rust producer at
    # `feature-extractor-MBO-LOB/crates/hft-export-pipeline/src/types.rs:117-131`
    # emits a LIST `[0, 1, 2]` for all 3 classification `LabelEncoding`
    # variants), fall back to the nested `labeling.label_encoding.values`
    # which strategy-specific producers (e.g.,
    # `feature-extractor-MBO-LOB/crates/hft-export-pipeline/src/strategies/triple_barrier.rs:156-164`)
    # correctly emit as a dict. This adds backward-compat AND defends future
    # producer drift. Producer-side cleanup tracked as #PY-218.
    encoding = metadata.get("label_encoding", {})
    values_map = None
    if isinstance(encoding, dict) and "values" in encoding:
        candidate = encoding["values"]
        if isinstance(candidate, dict):
            values_map = candidate

    if values_map is None:
        labeling = metadata.get("labeling", {})
        if isinstance(labeling, dict):
            nested_enc = labeling.get("label_encoding", {})
            if isinstance(nested_enc, dict) and "values" in nested_enc:
                candidate = nested_enc["values"]
                if isinstance(candidate, dict):
                    values_map = candidate

    if values_map is not None:
        contract_names = {str(k): v for k, v in contract.class_names.items()}
        if values_map != contract_names:
            raise ContractError(
                f"Label encoding mismatch for strategy '{strategy}'. "
                f"Metadata: {values_map}, Contract: {contract_names}"
            )


def validate_provenance_present(metadata: dict) -> list[str]:
    """
    Check that the provenance block is present and has required fields.

    Args:
        metadata: Metadata dict from *_metadata.json

    Returns:
        List of warning messages for missing provenance fields.
        Empty list means provenance is complete.
    """
    warnings: list[str] = []
    provenance = metadata.get("provenance")

    if provenance is None:
        warnings.append("No 'provenance' block in metadata — cannot trace data lineage")
        return warnings

    if not isinstance(provenance, dict):
        warnings.append("'provenance' field is not a dict")
        return warnings

    for field in EXPORT_METADATA_PROVENANCE_FIELDS:
        if field not in provenance:
            warnings.append(f"Missing provenance field: '{field}'")

    return warnings


def validate_idx_97_reserved(
    sequences_path,
    *,
    strict: bool = False,
) -> list[str]:
    """Verify that feature index 97 is RESERVED 0.0 in the first sample.

    Per CLAUDE.md root: "schema_version=3.0; idx 97 RESERVED 0.0 forever
    post-G.1". Pre-Phase-O exports stored ``schema_version`` as a numeric
    value at idx 97 (e.g., 2.2). Phase O Cycle 1 promoted schema_version
    to JSON metadata STRING and reserved idx 97 to a constant 0.0 forever.

    A buggy producer that emits ``sequences[:, :, 97] = 1.5`` would pass
    ``validate_export_contract`` (which only inspects metadata.json, never
    the NPY values). This validator closes that gap by spot-sampling the
    first row's idx 97.

    Args:
        sequences_path: Path to a *_sequences.npy file (the NPY array,
            not metadata).
        strict: If True, raise ``ContractError`` on idx-97-not-zero.
            Default False — emits warnings only (per hft-rules §8 +
            back-compat with pre-Phase-O archive consumption where
            idx 97 was the schema_version value).

    Returns:
        List of warnings (empty if all OK).

    Raises:
        ContractError: If ``strict=True`` and idx 97 != 0.0.
        FileNotFoundError: If the sequences file doesn't exist.
        IndexError: If the array has fewer than 98 features (caller bug).

    Phase X.3 Empirical Trust (2026-05-05) — Phase C.3.
    """
    import numpy as np
    from pathlib import Path

    sequences_path = Path(sequences_path)
    if not sequences_path.exists():
        raise FileNotFoundError(
            f"validate_idx_97_reserved: sequences file not found: {sequences_path}"
        )

    # mmap_mode='r' gives O(1) header read without loading the full array
    arr = np.load(sequences_path, mmap_mode="r")

    if arr.ndim != 3:
        return [
            f"Cannot validate idx 97 on {sequences_path.name}: expected 3D "
            f"(N, T, F) array, got shape {arr.shape}"
        ]

    n_features = arr.shape[2]
    if n_features < 98:
        # Pre-stable-features regime; idx 97 doesn't exist
        return []

    # Spot-sample first row of first sequence
    first_value = float(arr[0, 0, 97])

    if first_value == 0.0:
        return []  # Correct — idx 97 is RESERVED 0.0 per Phase O Cycle 1

    msg = (
        f"Idx 97 RESERVED 0.0 violation in {sequences_path.name}: "
        f"first sample has feature[97]={first_value!r} (expected 0.0). "
        f"Pre-Phase-O exports may have stored schema_version here (legacy "
        f"value). Post-Phase-O, idx 97 is RESERVED 0.0 forever."
    )

    if strict:
        raise ContractError(msg)

    return [msg]


def validate_export_contract(
    metadata: dict,
    *,
    strict_completeness: bool = False,
) -> list[str]:
    """
    Full contract validation for exported data. Call once at dataset load time.

    Checks:
        1. Schema version matches
        2. Feature count is in the valid set {FEATURE_COUNT, FULL_FEATURE_COUNT}
        3. Normalization not applied (safe for Python normalization)
        4. Metadata completeness (warnings for missing optional fields)
        5. Label encoding matches contract (if label_strategy present)
        6. Provenance present (warnings only)

    Args:
        metadata: Metadata dict from *_metadata.json
        strict_completeness: If True, missing critical metadata fields raise errors.

    Returns:
        List of non-fatal warnings (missing optional fields, provenance, etc.)

    Raises:
        ContractError: On any hard contract violation.
    """
    warnings: list[str] = []

    validate_schema_version(metadata)

    n_features = metadata.get("n_features")
    if n_features is not None:
        if n_features < FEATURE_COUNT or n_features > FULL_FEATURE_COUNT:
            raise ContractError(
                f"Feature count {n_features} outside valid range "
                f"[{FEATURE_COUNT}, {FULL_FEATURE_COUNT}]. "
                f"Check FeatureConfig and experimental groups."
            )

    validate_normalization_not_applied(metadata)

    warnings.extend(
        validate_metadata_completeness(metadata, strict=strict_completeness)
    )

    validate_label_encoding(metadata)

    warnings.extend(validate_provenance_present(metadata))

    return warnings


def validate_day_metadata(metadata: dict | None, date: str) -> list[str]:
    """Validate export metadata for a single day at the load boundary.

    Phase X.2.A SSoT (2026-05-04): lifted from
    ``lobtrainer.data.dataset._validate_day_metadata``. All consumers
    (trainer, backtester, lob-dataset-analyzer, hft-feature-evaluator)
    MUST call this primitive at NPY-load time to enforce Phase O Cycle 1
    C-2 + C-3 + backtester C-4 hardening uniformly across modules — per
    hft-rules §0 "consume the SSoT, never re-implement".

    Pre-X.2.A: ``_validate_day_metadata`` was trainer-private + open-coded
    duplicated in 2 backtester sites + bypassed at 4 trainer + 17 analyzer
    + 1 evaluator NPY-load sites. This SSoT lift unifies the 20+ sites.

    Returns the warnings list (caller is responsible for logging) — preserves
    hft-contracts' log-free architectural invariant. Raises ContractError
    with date prefix on hard violations (per hft-rules §8 "never silently
    drop, clamp, or fix data").

    Args:
        metadata: Loaded metadata dict from ``*_metadata.json``. ``None``
            signals a missing file, which IS a hard contract violation
            (v3.0 producers always emit metadata.json).
        date: Date string used for error/warning context. Embedded in
            error messages so operators can locate the offending day in
            a multi-day corpus.

    Returns:
        List of non-fatal contract warnings (provenance gaps, optional
        field absences). Empty if export is fully contract-compliant.
        Caller logs these via ``logger.warning("(%s) %s", date, w)`` or
        equivalent — the SSoT does NOT log to preserve module purity.

    Raises:
        ContractError: If metadata is None, lacks ``schema_version``, or
            fails any branch of ``validate_export_contract``. Date prefix
            added for triage. ``__cause__`` chain preserves the underlying
            ContractError for diagnostic reading.
    """
    if metadata is None:
        raise ContractError(
            f"Export metadata for {date} is missing or could not be loaded. "
            f"v3.0 contract requires every day to have a *_metadata.json file. "
            f"Re-export this day or remove from corpus."
        )

    if "schema_version" not in metadata:
        raise ContractError(
            f"Export metadata for {date} has no 'schema_version' field. "
            f"Cannot verify contract compatibility. "
            f"Re-export with the latest feature extractor (Phase O Cycle 1+)."
        )

    # Wrap downstream raises with the date so operators can locate the
    # offending day in a multi-day corpus (per hft-rules §8 + Phase O C-3).
    try:
        warnings = validate_export_contract(metadata, strict_completeness=False)
    except ContractError as exc:
        raise ContractError(
            f"Export contract violation for {date}: {exc}"
        ) from exc

    return warnings


def validate_export_dir(export_dir, *, strict: bool = True) -> list[str]:
    """Directory-level export-integrity validator for MBO NPY exports.

    Composes the per-day :func:`validate_day_metadata` SSoT across an entire
    export directory and adds the cross-day + manifest<->disk reconciliation
    that a per-day metadata validator structurally cannot see (its
    "metadata-only gap"):

      1. Every ``{split}/{day}_metadata.json`` passes ``validate_day_metadata``.
      2. Every ``*_sequences.npy`` has a matching ``*_metadata.json`` (and v.v.).
      3. Cross-day **uniform** ``schema_version`` == ``manifest.schema_version``
         (catches a re-export layered over an old one — e.g. mixed day-files
         ``{"2.2", "3.0"}``).
      4. Cross-day **single** producer ``provenance.git_commit`` (catches a
         partial re-export under a different commit — e.g. ``{b5e746d, c5e9d64}``).
      5. Per-split + total on-disk ``*_sequences.npy`` count reconciles with the
         manifest's attempted-day accounting *minus* recorded failures:
         ``#files[split] == manifest.split[split].days
         − #failed_partitions[split] − #skipped_days[split]`` (and the total
         ``#files == days_processed − #failed − #skipped``). Catches BOTH
         missing days (silent truncation) AND **extra** stale day-files left
         behind by a re-export layered over an old export.
      6. Split day-sets are disjoint (no day appears in two splits — leakage).

    **What this deliberately does NOT assert** (CF-1, ground-truth 2026-05-29):
    the manifest's ``total_sequences`` is the **pre-alignment** generated count
    (``output.total_sequences()`` = ``sequences_generated()`` in the extractor),
    whereas the per-day ``n_sequences`` and the on-disk NPY rows are the
    **post-alignment** emitted count. They differ by construction (the label
    ``2*k + max_h + 1`` alignment trim), so ``total_sequences`` is NOT compared
    to the on-disk sum — doing so would false-positive on every healthy export
    (e.g. the v3p0 export legitimately has ``total_sequences=136902`` vs
    ~66182 on disk). The honest disk-truth field ``total_sequences_emitted``
    (added separately by the manifest writer) IS reconciled here when present
    (forward-compatible — skipped if absent). (A parallel ``days_emitted`` field
    was deliberately NOT added — redundant: emitted-day count is already
    reconciled via ``days_processed − failed − skipped`` vs the on-disk
    ``*_sequences.npy`` count, and equals ``len(diagnostics_files)``. Ground-truth
    verdict 2026-05-29.)

    Reuses :func:`validate_day_metadata` per day (the per-day SSoT) — no
    duplicated per-day contract logic. Remains log-free (returns warnings; the
    caller logs) per the module invariant. **MBO exports only.** Off-exchange
    exports DO carry a ``dataset_manifest.json`` (ground-truth 2026-05-30:
    ``data/exports/basic_nvda_60s``), but use the off-exchange contract
    (``schema_version`` 1.0 / ``contract_version`` ``off_exchange_*``); this
    validator fail-fast rejects them with a single clear pointer to
    :func:`validate_off_exchange_export_contract` /
    :func:`validate_any_export_contract` rather than emitting a wall of
    spurious per-day ``schema_version`` mismatches.

    Args:
        export_dir: Path to an MBO export directory containing
            ``dataset_manifest.json`` and ``{train,val,test}/`` split subdirs.
        strict: If True (default — fail-loud per hft-rules §8), any HARD
            integrity violation raises ``ContractError`` aggregating ALL
            detected hard violations. If False, hard violations are returned in
            the result list prefixed ``"ERROR: "`` (for audit/CLI that wants the
            full picture without raising).

    Returns:
        List of non-fatal warnings (per-day provenance gaps, optional-field
        absences, non-attributable skipped-day notes). When ``strict=False``,
        ``"ERROR: "``-prefixed hard violations are prepended.

    Raises:
        FileNotFoundError: If ``export_dir`` does not exist.
        ContractError: If ``strict=True`` and any hard integrity violation is
            found (or the manifest is missing/unparseable, regardless of
            ``strict`` for the missing/unparseable case under ``strict=True``;
            under ``strict=False`` those return an ``"ERROR: "`` entry).

    Origin: Foundation Integrity cluster (2026-05-29) — closes the
    manifest<->disk + cross-day-uniformity gap that allowed a polluted export
    directory (mixed ``schema_version`` + producer commit, manifest counts
    disagreeing with disk) to pass the per-day validators undetected.
    """
    import json
    from pathlib import Path

    export_dir = Path(export_dir)
    if not export_dir.exists():
        raise FileNotFoundError(
            f"validate_export_dir: export directory not found: {export_dir}"
        )
    if not export_dir.is_dir():
        raise ContractError(f"validate_export_dir: not a directory: {export_dir}")

    # -- Manifest (required for MBO exports) — structural precondition --
    manifest_path = export_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        msg = (
            f"missing dataset_manifest.json in {export_dir} — an MBO export "
            f"directory must carry a manifest."
        )
        if strict:
            raise ContractError(f"validate_export_dir: {msg}")
        return [f"ERROR: {msg}"]
    try:
        manifest = json.loads(manifest_path.read_text())
    except (ValueError, OSError) as exc:
        msg = f"dataset_manifest.json unparseable ({manifest_path}): {exc}"
        if strict:
            raise ContractError(f"validate_export_dir: {msg}") from exc
        return [f"ERROR: {msg}"]

    # -- Off-exchange precondition (this is the MBO directory validator). --
    # Off-exchange exports DO carry a manifest (ground-truth 2026-05-30:
    # basic_nvda_60s carries contract_version="off_exchange_1.0"), but use the
    # off_exchange_* contract (schema_version 1.0). Applying the MBO per-day
    # contract to them would emit a wall of spurious per-day schema_version
    # mismatches; fail-fast with a single clear pointer instead. Discriminator
    # mirrors validate_any_export_contract's cv.startswith("off_exchange").
    # Mirrors the missing/unparseable manifest preconditions above (strict
    # raises; non-strict returns one "ERROR: " entry).
    manifest_cv = str(manifest.get("contract_version") or "")
    if manifest_cv.startswith("off_exchange"):
        msg = (
            f"{export_dir} is an off-exchange export "
            f"(contract_version={manifest_cv!r}); validate_export_dir applies the "
            f"MBO export contract — use validate_off_exchange_export_contract "
            f"(per-day) or validate_any_export_contract (auto-routing) instead."
        )
        if strict:
            raise ContractError(f"validate_export_dir: {msg}")
        return [f"ERROR: {msg}"]

    errors: list[str] = []
    warnings: list[str] = []

    manifest_schema = manifest.get("schema_version")
    manifest_schema = None if manifest_schema is None else str(manifest_schema)

    # The writer emits "split" (singular); the contract field list says
    # "splits" — tolerate both (split-vs-splits discrepancy, 2026-05-29).
    split_block = manifest.get("split")
    if not isinstance(split_block, dict):
        split_block = manifest.get("splits")
    if not isinstance(split_block, dict):
        split_block = {}

    # Failed-partition accounting: partial_failure.failed_partitions[].
    # partition_key.{day,split}. These days were attempted but NOT emitted.
    pf = manifest.get("partial_failure")
    failed_partitions = pf.get("failed_partitions", []) if isinstance(pf, dict) else []
    failed_by_split: dict[str, set] = {}
    failed_total = 0
    for fp in failed_partitions:
        if not isinstance(fp, dict):
            continue
        failed_total += 1
        pk = fp.get("partition_key", {})
        if isinstance(pk, dict) and pk.get("split") is not None:
            failed_by_split.setdefault(str(pk["split"]), set()).add(str(pk.get("day")))

    # Skipped-days accounting: may be an int count or a list of {day,split}.
    skipped_raw = manifest.get("skipped_days", 0)
    skipped_by_split: dict[str, set] = {}
    skipped_total = 0
    skipped_attributable = True
    if isinstance(skipped_raw, bool):
        pass  # ignore stray bool
    elif isinstance(skipped_raw, int):
        skipped_total = skipped_raw
        if skipped_raw > 0:
            skipped_attributable = False  # int count → no per-split breakdown
    elif isinstance(skipped_raw, list):
        for sk in skipped_raw:
            skipped_total += 1
            if isinstance(sk, dict) and sk.get("split") is not None:
                skipped_by_split.setdefault(str(sk["split"]), set()).add(
                    str(sk.get("day"))
                )
            else:
                skipped_attributable = False

    # -- Walk split subdirs --
    SPLITS = ("train", "val", "test")
    present_splits = [s for s in SPLITS if (export_dir / s).is_dir()]
    if not present_splits:
        msg = f"no train/val/test split subdirs found in {export_dir}."
        if strict:
            raise ContractError(f"validate_export_dir: {msg}")
        return [f"ERROR: {msg}"]

    all_schema_versions: set = set()
    all_commits: set = set()
    day_to_splits: dict[str, set] = {}
    total_disk_files = 0
    total_n_sequences = 0  # Σ per-day n_sequences (post-align == on-disk rows)

    for split in present_splits:
        split_dir = export_dir / split
        seq_files = sorted(split_dir.glob("*_sequences.npy"))
        meta_files = sorted(split_dir.glob("*_metadata.json"))
        seq_days = {f.name[: -len("_sequences.npy")] for f in seq_files}
        meta_days = {f.name[: -len("_metadata.json")] for f in meta_files}

        # (2) pairing
        for d in sorted(seq_days - meta_days):
            errors.append(
                f"{split}/{d}: *_sequences.npy without matching *_metadata.json"
            )
        for d in sorted(meta_days - seq_days):
            errors.append(
                f"{split}/{d}: *_metadata.json without matching *_sequences.npy"
            )

        n_disk = len(seq_days)
        total_disk_files += n_disk

        # per-day contract + collect cross-day uniformity inputs
        for mf in meta_files:
            day = mf.name[: -len("_metadata.json")]
            day_to_splits.setdefault(day, set()).add(split)
            try:
                meta = json.loads(mf.read_text())
            except (ValueError, OSError) as exc:
                errors.append(f"{split}/{day}: metadata unparseable: {exc}")
                continue
            try:
                day_warnings = validate_day_metadata(meta, f"{split}/{day}")
            except ContractError as exc:
                errors.append(str(exc))
            else:
                warnings.extend(f"({split}/{day}) {w}" for w in day_warnings)
            sv = meta.get("schema_version")
            if sv is not None:
                all_schema_versions.add(str(sv))
            prov = meta.get("provenance")
            if isinstance(prov, dict) and prov.get("git_commit") is not None:
                all_commits.add(str(prov["git_commit"]))
            ns = meta.get("n_sequences")
            if isinstance(ns, int) and not isinstance(ns, bool):
                total_n_sequences += ns

        # (5) per-split count reconciliation
        claimed = split_block.get(split)
        claimed_days = claimed.get("days") if isinstance(claimed, dict) else None
        if isinstance(claimed_days, int) and not isinstance(claimed_days, bool):
            failed_here = len(failed_by_split.get(split, set()))
            if skipped_attributable:
                skipped_here = len(skipped_by_split.get(split, set()))
                expected = claimed_days - failed_here - skipped_here
                if n_disk != expected:
                    errors.append(
                        f"{split}: on-disk *_sequences.npy count {n_disk} != "
                        f"manifest.split.{split}.days {claimed_days} "
                        f"− {failed_here} failed − {skipped_here} skipped "
                        f"= {expected} (manifest disagrees with disk: silent "
                        f"truncation or stale re-export leftover)."
                    )
            else:
                warnings.append(
                    f"{split}: per-split count check skipped — skipped_days is a "
                    f"non-attributable count ({skipped_total}); relying on the "
                    f"total reconciliation."
                )

    # (3) cross-day schema_version uniformity
    if len(all_schema_versions) > 1:
        errors.append(
            f"Mixed schema_version across day-files: {sorted(all_schema_versions)} "
            f"— a single export directory must be uniform (a partial re-export "
            f"was layered over an older export)."
        )
    elif all_schema_versions and manifest_schema is not None:
        only = next(iter(all_schema_versions))
        if only != manifest_schema:
            errors.append(
                f"Day-file schema_version {only!r} != manifest.schema_version "
                f"{manifest_schema!r}."
            )

    # (4) cross-day producer-commit uniformity
    if len(all_commits) > 1:
        errors.append(
            f"Mixed producer git_commit across day-files: {sorted(all_commits)} "
            f"— a single export directory must come from one producer commit "
            f"(a partial re-export under a different commit was detected)."
        )

    # (5-total) total count reconciliation
    days_processed = manifest.get("days_processed")
    if isinstance(days_processed, int) and not isinstance(days_processed, bool):
        expected_total = days_processed - failed_total - skipped_total
        if total_disk_files != expected_total:
            errors.append(
                f"Total on-disk *_sequences.npy count {total_disk_files} != "
                f"manifest.days_processed {days_processed} − {failed_total} "
                f"failed − {skipped_total} skipped = {expected_total}."
            )

    # (6) split disjointness
    for day, splits_for_day in sorted(day_to_splits.items()):
        if len(splits_for_day) > 1:
            errors.append(
                f"Day {day} appears in multiple splits "
                f"{sorted(splits_for_day)} — split day-sets must be disjoint "
                f"(leakage risk)."
            )

    # (forward-compat) honest disk-truth sequence total, when the writer emits it.
    # No ``days_emitted`` check: emitted-day count is already reconciled above via
    # ``days_processed − failed − skipped`` vs the on-disk file count (and equals
    # ``len(diagnostics_files)``) — a separate scalar would be redundant.
    tse = manifest.get("total_sequences_emitted")
    if isinstance(tse, int) and not isinstance(tse, bool) and tse != total_n_sequences:
        errors.append(
            f"manifest.total_sequences_emitted {tse} != Σ per-day n_sequences "
            f"{total_n_sequences} (post-alignment on-disk truth)."
        )

    if errors:
        if strict:
            raise ContractError(
                f"Export directory integrity violations in {export_dir}:\n- "
                + "\n- ".join(errors)
            )
        warnings = [f"ERROR: {e}" for e in errors] + warnings

    return warnings


def validate_off_exchange_export_contract(
    metadata: dict,
) -> list[str]:
    """Validate an off-exchange export metadata dict.

    Parallel to validate_export_contract() but with off-exchange-specific rules:
    - n_features must be OFF_EXCHANGE_FEATURE_COUNT (34)
    - schema_version must be OFF_EXCHANGE_SCHEMA_VERSION ("1.0")
    - contract_version must start with "off_exchange"
    - Provenance uses processor_version (not extractor_version)

    Args:
        metadata: Metadata dict from basic-quote-processor export.

    Returns:
        List of non-fatal warnings.

    Raises:
        ContractError: On any hard contract violation.
    """
    warnings: list[str] = []

    # Schema version check (off-exchange uses "1.0", not "2.2")
    sv = metadata.get("schema_version")
    if sv is None:
        raise ContractError("Missing 'schema_version' in off-exchange metadata.")
    if str(sv) != str(OFF_EXCHANGE_SCHEMA_VERSION):
        raise ContractError(
            f"Off-exchange schema_version mismatch: got '{sv}', "
            f"expected '{OFF_EXCHANGE_SCHEMA_VERSION}'."
        )

    # Contract version check
    cv = metadata.get("contract_version", "")
    if not cv.startswith("off_exchange"):
        raise ContractError(
            f"Off-exchange contract_version mismatch: got '{cv}', "
            f"expected 'off_exchange_*'."
        )

    # Feature count check
    n_features = metadata.get("n_features")
    if n_features is not None:
        if n_features != OFF_EXCHANGE_FEATURE_COUNT:
            raise ContractError(
                f"Feature count {n_features} != OFF_EXCHANGE_FEATURE_COUNT "
                f"({OFF_EXCHANGE_FEATURE_COUNT})."
            )

    # Normalization not applied
    validate_normalization_not_applied(metadata)

    # Required fields (off-exchange specific)
    offex_required = {
        "day", "n_sequences", "window_size", "n_features",
        "schema_version", "contract_version",
        "label_strategy", "label_encoding", "horizons",
        "bin_size_seconds", "normalization", "provenance", "export_timestamp",
    }
    present = set(metadata.keys())
    missing = offex_required - present
    if missing:
        warnings.append(f"Off-exchange metadata missing optional fields: {sorted(missing)}")

    # Provenance (off-exchange uses processor_version, not extractor_version)
    prov = metadata.get("provenance")
    if prov is None:
        warnings.append("Missing 'provenance' block in off-exchange metadata.")
    elif not isinstance(prov, dict):
        warnings.append("Provenance is not a dict in off-exchange metadata.")
    else:
        if "processor_version" not in prov and "export_timestamp_utc" not in prov:
            warnings.append(
                "Off-exchange provenance missing processor_version and export_timestamp_utc."
            )

    return warnings


def validate_any_export_contract(
    metadata: dict,
    strict_completeness: bool = False,
) -> list[str]:
    """Auto-detect pipeline type and validate accordingly.

    Uses the 'contract_version' field to select the correct validator:
    - contract_version starts with "off_exchange" → off-exchange validation
    - Otherwise → MBO pipeline validation

    Args:
        metadata: Metadata dict from any pipeline export.
        strict_completeness: Passed to MBO validator only.

    Returns:
        List of non-fatal warnings.

    Raises:
        ContractError: On any hard contract violation.
    """
    cv = metadata.get("contract_version", "")
    if isinstance(cv, str) and cv.startswith("off_exchange"):
        return validate_off_exchange_export_contract(metadata)
    else:
        return validate_export_contract(metadata, strict_completeness=strict_completeness)


def assert_finite_array(
    arr: np.ndarray,
    *,
    name: str,
    extra_diagnostic: Optional[str] = None,
) -> None:
    """Fail-loud on NaN/Inf in a numpy array per hft-rules §8.

    SSoT extraction 2026-05-07 (#PY-63): consolidates 7 sites that had
    duplicated this idiom across producer-side fail-loud checks. Previously
    each site inlined the pattern with slightly different error message
    templates. This helper standardizes on:
        "{name}: array contains {n_nan} NaN, {n_inf} Inf out of {N} total
        — input invariant violation. {extra_diagnostic}"

    Per hft-rules §8 ("Never silently drop, clamp, or 'fix' data"): silent
    NaN/Inf substitution masks upstream data corruption as legitimate model
    output, biasing IC/DA/correlation metrics downstream toward 0.0. Hard
    invariants at producer boundaries should raise, not silently zero.

    Args:
        arr: Array to validate. None passthrough is NOT supported — caller
            must guard for None before invoking.
        name: Identifier for error messages (e.g., "predicted_returns",
            "SignalExporter._infer_regression"). Required keyword-only.
        extra_diagnostic: Optional extra context appended to the error
            message (e.g., suggested debugging path, model_type echo).

    Raises:
        ValueError: If arr contains any NaN or Inf, with diagnostic counts.

    Example:
        >>> from hft_contracts.validation import assert_finite_array
        >>> import numpy as np
        >>> assert_finite_array(np.array([1.0, 2.0, 3.0]), name="x")
        >>> assert_finite_array(
        ...     np.array([1.0, np.nan]),
        ...     name="x",
        ...     extra_diagnostic="Investigate upstream producer.",
        ... )
        Traceback (most recent call last):
            ...
        ValueError: x: array contains 1 NaN, 0 Inf out of 2 total ...
    """
    if not np.all(np.isfinite(arr)):
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        msg = (
            f"{name}: array contains {n_nan} NaN, {n_inf} Inf "
            f"out of {arr.size} total — input invariant violation."
        )
        if extra_diagnostic:
            msg += f" {extra_diagnostic}"
        raise ValueError(msg)
