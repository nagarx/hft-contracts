"""Cross-module ``CompatibilityContract`` — SSoT for artifact-vs-consumer compatibility.

This module introduces a first-class, content-hashable handle that every pipeline
artifact (signal directory, checkpoint, feature-importance, future kinds) embeds
to declare the shape-determining facts of its producer. Consumers re-compute the
expected contract from THEIR config + SSoT constants and compare fingerprints
BEFORE loading arrays.

Phase II (plan v2.0, 2026-04-20). Solves validation-report D1/D10/D11:
    D1  signal_metadata.json lacked shape-determining fields
    D10 calibrated_returns.npy precedence was file-existence-only, no freshness gate
    D11 SignalManifest.validate did NOT re-check schema/contract version on load

Contract surface (11 shape-determining keys):
    - contract_version       : hft_contracts.SCHEMA_VERSION at producer time
    - schema_version         : alias of contract_version today (reserved for future split)
    - feature_count          : e.g., 98 | 148 | 34
    - window_size            : sequence length (T)
    - feature_layout         : FeatureSet.content_hash OR "default" (registry tag)
    - data_source            : "mbo_lob" | "off_exchange" | ...
    - label_strategy_hash    : sha256 of full LabelsConfig dict (not flat string —
                                captures threshold, horizons-tuple, smoothing, etc.)
    - calibration_method     : None | "variance_match" | ...
    - primary_horizon_idx    : HMHP-family: which horizon the backtester consumes
    - horizons               : full tuple of horizon values (e.g., (10, 60, 300))
    - normalization_strategy : "none" | "zscore" | "market_structure_zscore"

Why 11 and not 8 (v1.0 plan):
    Adversarial validation flagged 3 missing axes — different ``horizons`` produce
    different signal surfaces even at same feature_count/window_size; different
    ``primary_horizon_idx`` chooses a different horizon for backtesting; different
    ``normalization_strategy`` yields different signal ranges. All are shape-
    determining choices that belong in the fingerprint.

Why ``label_strategy_hash`` and not flat ``label_strategy`` string:
    A label strategy's behavior depends on its PARAMETERS (smoothing_window,
    thresholds, barrier multipliers, etc.). A flat string would collide across
    parameterizations of the same strategy. Hashing the full config dict preserves
    granularity while keeping the contract surface flat.

Why ``contract_version`` and ``schema_version`` both (not deduplicated):
    Today they are aliased (both map to ``hft_contracts.SCHEMA_VERSION``). Preserving
    both in the canonical form reserves the ability to SPLIT them in the future
    without breaking every committed fingerprint. The redundancy is cheap (two
    strings) and the cost of conflating them later is high.

Design principles applied (hft-rules):
    - §1 Single source of truth: fingerprint computation uses hft_contracts.canonical_hash
    - §2 Deterministic: sha256 over sorted-keys canonical JSON
    - §5 Fail-fast: fingerprint mismatch raises ContractError, not silent drift
    - §6 Test Contract type: every field varied → fingerprint changes (locked in tests)
    - §7 Reproducibility: same inputs → same fingerprint across environments
    - §11 Self-documenting: inline contract of which fields are "shape-determining"
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hft_contracts.canonical_hash import canonical_json_blob, sanitize_for_hash, sha256_hex

__all__ = [
    "CompatibilityContract",
    "COMPATIBILITY_CONTRACT_SCHEMA_VERSION",
]


# Bumped when the set of fields OR the canonicalization rule changes.
# Consumers read this to decide whether a stored fingerprint is comparable.
COMPATIBILITY_CONTRACT_SCHEMA_VERSION: str = "1.0.0"


@dataclass(frozen=True)
class CompatibilityContract:
    """Shape-determining inputs pinned per-artifact for cross-module compatibility.

    Instances are frozen and hashable. ``fingerprint()`` returns a 64-char lowercase
    hex SHA-256 that a consumer compares to the stored producer fingerprint. A
    mismatch means the artifact was produced against a different contract than the
    consumer expects — load MUST fail (hft-rules §8 hard error, not silent drift).

    Fields intentionally kept flat (no nested dataclasses) to keep the canonical
    form diff-stable and JSON-transportable without recursion.

    Examples:
        >>> c = CompatibilityContract(
        ...     contract_version="2.2", schema_version="2.2",
        ...     feature_count=98, window_size=100,
        ...     feature_layout="default", data_source="mbo_lob",
        ...     label_strategy_hash="abc...",
        ...     calibration_method=None,
        ...     primary_horizon_idx=0,
        ...     horizons=(10, 60, 300),
        ...     normalization_strategy="none",
        ... )
        >>> fp = c.fingerprint()
        >>> len(fp) == 64
        True
        >>> c.diff(c) == {}
        True
    """

    # Contract schema versions (alias today, future-splittable per docstring rationale)
    contract_version: str
    schema_version: str

    # Array geometry
    feature_count: int
    window_size: int

    # Feature layout identity (FeatureSet.content_hash OR "default")
    feature_layout: str

    # Source pipeline (MBO vs off-exchange vs future asset classes)
    data_source: str

    # Full LabelsConfig hash (captures strategy + horizons + thresholds + smoothing)
    label_strategy_hash: str

    # Calibration state — None means no calibration applied
    calibration_method: Optional[str]

    # Multi-horizon model dispatch: which horizon the backtester selects
    primary_horizon_idx: Optional[int]

    # Full horizons tuple — stored as tuple so hashable & serializable
    horizons: Optional[Tuple[int, ...]]

    # Normalization state (raw / zscore / market_structure_zscore / ...)
    normalization_strategy: str

    # ------------------------------------------------------------------
    # Construction-time validation (Phase II hardening, 2026-04-20)
    # ------------------------------------------------------------------
    # hft-rules §5: "If a config option exists but is not fully supported,
    # it must fail fast with a precise error — never silently degrade."
    # Prior to this pass, CompatibilityContract accepted feature_count=0,
    # empty strings, or primary_horizon_idx out of horizons-range — all of
    # which produced valid-looking fingerprints that poisoned downstream
    # comparisons. Construction-time validation closes the class.
    def __post_init__(self) -> None:
        # Geometry — zero-valued geometry is never real and would poison fingerprints.
        if not isinstance(self.feature_count, int) or self.feature_count <= 0:
            raise ValueError(
                f"feature_count must be a positive int, got {self.feature_count!r} "
                f"(type={type(self.feature_count).__name__})"
            )
        if not isinstance(self.window_size, int) or self.window_size <= 0:
            raise ValueError(
                f"window_size must be a positive int, got {self.window_size!r} "
                f"(type={type(self.window_size).__name__})"
            )
        # String identity fields — empty strings are a silent-drift trap
        # (they serialize + fingerprint but carry zero information).
        _non_empty_strs: Dict[str, Any] = {
            "contract_version": self.contract_version,
            "schema_version": self.schema_version,
            "feature_layout": self.feature_layout,
            "data_source": self.data_source,
            "label_strategy_hash": self.label_strategy_hash,
            "normalization_strategy": self.normalization_strategy,
        }
        for name, val in _non_empty_strs.items():
            if not isinstance(val, str) or not val:
                raise ValueError(
                    f"{name} must be a non-empty string, got {val!r} "
                    f"(type={type(val).__name__})"
                )
        # calibration_method (Optional[str]): when set, must be non-empty.
        if self.calibration_method is not None:
            if not isinstance(self.calibration_method, str) or not self.calibration_method:
                raise ValueError(
                    f"calibration_method must be None or a non-empty string, got "
                    f"{self.calibration_method!r}"
                )
        # horizons: coerce list → tuple for diff-stable fingerprint (frozen dataclass
        # so use object.__setattr__). Without this, a JSON round-trip of the to_dict()
        # form (where tuples serialize as lists) + re-construction would produce a
        # different canonical form AFTER asdict() — a silent fingerprint drift.
        if self.horizons is not None:
            if not isinstance(self.horizons, (list, tuple)):
                raise ValueError(
                    f"horizons must be None or a list/tuple of ints, got "
                    f"{self.horizons!r} (type={type(self.horizons).__name__})"
                )
            if len(self.horizons) == 0:
                raise ValueError(
                    "horizons must be None or a non-empty tuple of ints, got empty sequence"
                )
            for h in self.horizons:
                if not isinstance(h, int) or isinstance(h, bool) or h <= 0:
                    raise ValueError(
                        f"horizons must contain positive ints only, got {h!r} "
                        f"(type={type(h).__name__})"
                    )
            if not isinstance(self.horizons, tuple):
                object.__setattr__(self, "horizons", tuple(self.horizons))
        # primary_horizon_idx must index into horizons when horizons is set.
        if self.primary_horizon_idx is not None:
            if not isinstance(self.primary_horizon_idx, int) or isinstance(
                self.primary_horizon_idx, bool
            ):
                raise ValueError(
                    f"primary_horizon_idx must be None or int, got "
                    f"{self.primary_horizon_idx!r}"
                )
            if self.primary_horizon_idx < 0:
                raise ValueError(
                    f"primary_horizon_idx must be >= 0, got {self.primary_horizon_idx}"
                )
            if self.horizons is not None and self.primary_horizon_idx >= len(self.horizons):
                raise ValueError(
                    f"primary_horizon_idx={self.primary_horizon_idx} out of range for "
                    f"horizons={self.horizons} (len={len(self.horizons)})"
                )

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Return a canonical dict form (tuples → lists, sorted keys implicit).

        This is the form that ``fingerprint()`` hashes. Kept as a public method so
        consumers can inspect / log / compare fields without re-hashing.
        """
        d = asdict(self)
        # asdict already converts frozen dataclass to plain dict, but tuples stay tuples.
        # canonical_json_blob handles tuple→list via its json dumper by default.
        # Apply sanitize_for_hash to normalize tuples → lists explicitly for parity
        # with content_hash() helpers elsewhere in hft_contracts.
        return sanitize_for_hash(d)

    def fingerprint(self) -> str:
        """SHA-256 hex digest over the canonical form. 64-char lowercase.

        Invariants:
            - Same instance → same fingerprint (deterministic).
            - Two instances with identical fields → identical fingerprints (equality ⇒ same fp).
            - ANY field differing → different fingerprints (locked by tests).
            - Computed via ``hft_contracts.canonical_hash`` SSoT, so cross-module
              consumers computing the same canonical form get the same bytes.
        """
        canonical = self.to_canonical_dict()
        return sha256_hex(canonical_json_blob(canonical))

    def diff(self, other: "CompatibilityContract") -> Dict[str, Tuple[Any, Any]]:
        """Field-level diff for actionable error messages.

        Returns:
            Dict mapping field name to ``(self_value, other_value)`` for every
            field whose values differ. Empty dict if the contracts are equal.
        """
        self_d = asdict(self)
        other_d = asdict(other)
        return {
            k: (self_d[k], other_d[k])
            for k in self_d
            if self_d[k] != other_d[k]
        }

    def key_fields(self) -> List[str]:
        """List of field names participating in the fingerprint.

        Exposed so consumers can introspect the contract surface for debug output
        without hardcoding the field list anywhere (hft-rules §1 single source of truth).
        """
        return list(asdict(self).keys())


def compute_label_strategy_hash(labels_config: Any) -> str:
    """Derive ``label_strategy_hash`` for a CompatibilityContract from any labels config.

    The labels config may be a dataclass, dict, or any object convertible via
    ``asdict()`` / ``vars()`` / ``__dict__``. Hashing goes through the same
    ``hft_contracts.canonical_hash`` SSoT that other contract hashes use, so the
    fingerprint is consistent with ``feature_set.content_hash`` semantics.

    Args:
        labels_config: LabelsConfig dataclass instance, dict, or any canonicalizable
            object. Hashed via ``asdict()`` if it's a dataclass, else via ``vars()``.

    Returns:
        64-char lowercase hex SHA-256.
    """
    if hasattr(labels_config, "__dataclass_fields__"):
        payload = asdict(labels_config)
    elif isinstance(labels_config, dict):
        payload = dict(labels_config)
    else:
        # Best-effort introspection; sanitize_for_hash will flatten further
        payload = vars(labels_config) if hasattr(labels_config, "__dict__") else {"value": labels_config}
    return sha256_hex(canonical_json_blob(sanitize_for_hash(payload)))
