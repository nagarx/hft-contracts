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
"""

from __future__ import annotations

from typing import Sequence

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
    encoding = metadata.get("label_encoding", {})
    if isinstance(encoding, dict) and "values" in encoding:
        values_map = encoding["values"]
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
