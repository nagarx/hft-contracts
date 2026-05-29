"""Internal validation primitives for artifact __post_init__ methods.

Module-internal (underscore prefix per project convention at root
CLAUDE.md Multi-Agent Coordination — Shared Surface §C). Do NOT
import from cross-module boundaries.

Consumers within hft-contracts:
  - feature_importance_artifact.py (H-1 parity fix, 2026-05-28)
  - experiment_recorder.py (M-2 content_hash validation)
  - test_metrics_ci_artifact.py (future migration)
  - pairwise_compare_artifact.py (future migration)

Extracts the 11 validation primitives used across the Phase 2 gold-
standard artifacts (TestMetricsCIArtifact + PairwiseCompareArtifact)
into a shared internal SSoT per hft-rules section 0 (reuse-first).

Reserved-for-migration API: `validate_open_unit_interval`,
`validate_sha256_hex`, and `validate_optional_sha256_hex` currently
have no in-package caller. They are the API surface for the pending
Phase 2 consolidation — TestMetricsCIArtifact validates `ci in (0,1)`
and PairwiseCompareArtifact validates `alpha in (0,1)` + Optional
64-hex fingerprints with INLINE checks today; migrating those onto
these primitives is the planned next step that also closes the
bool-rejection divergence (the inline checks accept `True` as a
number; these primitives reject it). Kept rather than deleted to
avoid churn when that migration lands.

Error convention: all validators raise ValueError immediately on
failure (matches existing artifact __post_init__ pattern — fail-loud
per hft-rules section 8). No DeprecationWarning, no logging — callers
decide on warn-vs-raise policy above the validator layer.

Dependency discipline: ZERO intra-package imports. Defines a local
_SHA256_HEX_RE to avoid importing from signal_manifest (which would
invert the dependency direction — utility should not depend on domain).
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional, Tuple

_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")


def validate_finite_float(
    value: float,
    field_name: str,
    *,
    context: str = "",
) -> None:
    """Raise ValueError if value is NaN or +/-Inf."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}{field_name} must be a number, got "
            f"{type(value).__name__}"
        )
    if not math.isfinite(value):
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}{field_name}={value!r} is not finite"
        )


def validate_positive_int(
    value: int,
    field_name: str,
    *,
    context: str = "",
) -> None:
    """Raise ValueError if value <= 0 (strictly positive integer)."""
    if isinstance(value, bool) or not isinstance(value, int):
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}{field_name} must be int, got "
            f"{type(value).__name__}"
        )
    if value <= 0:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}{field_name}={value} must be > 0"
        )


def validate_non_negative_int(
    value: int,
    field_name: str,
    *,
    context: str = "",
) -> None:
    """Raise ValueError if value < 0 (>= 0 required)."""
    if isinstance(value, bool) or not isinstance(value, int):
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}{field_name} must be int, got "
            f"{type(value).__name__}"
        )
    if value < 0:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}{field_name}={value} must be >= 0"
        )


def validate_min_int(
    value: int,
    field_name: str,
    minimum: int,
    *,
    context: str = "",
) -> None:
    """Raise ValueError if value < minimum (inclusive lower bound)."""
    if isinstance(value, bool) or not isinstance(value, int):
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}{field_name} must be int, got "
            f"{type(value).__name__}"
        )
    if value < minimum:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}{field_name}={value} must be >= {minimum}"
        )


def validate_open_unit_interval(
    value: float,
    field_name: str,
    *,
    context: str = "",
) -> None:
    """Raise ValueError if value not in (0.0, 1.0) strictly."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}{field_name} must be a number, got "
            f"{type(value).__name__}"
        )
    if not (0.0 < value < 1.0):
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}{field_name}={value} must be in (0, 1)"
        )


def validate_closed_unit_interval(
    value: float,
    field_name: str,
    *,
    context: str = "",
) -> None:
    """Raise ValueError if value not in [0.0, 1.0] inclusive."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}{field_name} must be a number, got "
            f"{type(value).__name__}"
        )
    if not (0.0 <= value <= 1.0):
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}{field_name}={value} must be in [0, 1]"
        )


def validate_sha256_hex(
    value: str,
    field_name: str,
    *,
    context: str = "",
) -> None:
    """Raise ValueError if value is not 64-char lowercase hex."""
    if not isinstance(value, str) or not _SHA256_HEX_RE.match(value):
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}{field_name}={value!r} is not a valid "
            f"64-char lowercase hex SHA-256"
        )


def validate_optional_sha256_hex(
    value: Optional[str],
    field_name: str,
    *,
    context: str = "",
) -> None:
    """Validate SHA-256 hex ONLY if value is not None."""
    if value is None:
        return
    validate_sha256_hex(value, field_name, context=context)


def validate_non_empty_string(
    value: str,
    field_name: str,
    *,
    context: str = "",
) -> None:
    """Raise ValueError if value is not a non-empty string."""
    if not isinstance(value, str) or not value:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}{field_name} must be a non-empty string, "
            f"got {type(value).__name__}={value!r}"
        )


def validate_ci_ordering(
    low: float,
    high: float,
    *,
    context: str = "",
    low_name: str = "ci_low",
    high_name: str = "ci_high",
) -> None:
    """Raise ValueError if not (low <= high)."""
    if low > high:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}{low_name}={low} > {high_name}={high} "
            f"— CI bounds inverted"
        )


def validate_feature_set_ref(
    ref: Optional[Dict[str, Any]],
    field_name: str,
    *,
    context: str = "",
) -> None:
    """Validate feature_set_ref dict structure when not None.

    Checks: dict with non-empty string 'name' and 64-hex 'content_hash'.
    When ref is None: no-op (caller decides warn vs raise for None).
    """
    if ref is None:
        return
    prefix = f"{context}: " if context else ""
    if not isinstance(ref, dict):
        raise ValueError(
            f"{prefix}{field_name} must be a dict or None, "
            f"got {type(ref).__name__}"
        )
    name = ref.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"{prefix}{field_name}['name'] must be a non-empty string, "
            f"got {type(name).__name__}={name!r}"
        )
    content_hash = ref.get("content_hash")
    if not isinstance(content_hash, str):
        raise ValueError(
            f"{prefix}{field_name}['content_hash'] must be a string, "
            f"got {type(content_hash).__name__}"
        )
    if not _SHA256_HEX_RE.match(content_hash):
        raise ValueError(
            f"{prefix}{field_name}['content_hash']={content_hash!r} "
            f"is not a valid 64-char lowercase hex SHA-256"
        )
