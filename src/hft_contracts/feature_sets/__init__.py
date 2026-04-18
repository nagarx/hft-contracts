"""
hft_contracts.feature_sets — contract-plane FeatureSet schema + hashing.

Phase 6 6B.3 (2026-04-17): the SCHEMA + HASHING half of the Phase 4
FeatureSet registry lives here. The PRODUCER (orchestration + atomic
writes) and READ-SIDE REGISTRY (filesystem scan) stay in hft-ops
because they do filesystem I/O (``Path.glob`` / walk-up discovery /
atomic rename) — that kind of I/O would violate hft-contracts' invariant
that *import* has no side effects.

What lives here:
    - ``FeatureSet``            — the frozen dataclass + ``build`` + ``from_dict``
    - ``FeatureSetRef``         — lightweight {name, content_hash} pointer
    - ``FeatureSetAppliesTo``   — applicability metadata (NOT hashed)
    - ``FeatureSetProducedBy``  — producer provenance (NOT hashed)
    - ``FeatureSetValidationError``, ``FeatureSetIntegrityError``
    - ``FEATURE_SET_SCHEMA_VERSION``
    - ``validate_feature_set_dict`` — imperative schema validator
    - ``compute_feature_set_hash``  — SHA-256 over PRODUCT fields only

What stays in hft-ops:
    - ``hft_ops.feature_sets.writer``   — atomic JSON writer
    - ``hft_ops.feature_sets.registry`` — filesystem scan + get by name
    - ``hft_ops.feature_sets.producer`` — evaluator orchestration

See:
- ``PIPELINE_ARCHITECTURE.md`` §14.8 (FeatureSet Registry) + §17.3
  (producer→consumer matrix).
- ``contracts/feature_sets/SCHEMA.md`` — JSON schema reference.
- ``hft_contracts.canonical_hash`` — the SHA-256 SSoT.
"""

from __future__ import annotations

from hft_contracts.feature_sets.hashing import compute_feature_set_hash
from hft_contracts.feature_sets.schema import (
    FEATURE_SET_SCHEMA_VERSION,
    FeatureSet,
    FeatureSetAppliesTo,
    FeatureSetIntegrityError,
    FeatureSetProducedBy,
    FeatureSetRef,
    FeatureSetValidationError,
    validate_feature_set_dict,
)

__all__ = [
    # hashing
    "compute_feature_set_hash",
    # schema
    "FEATURE_SET_SCHEMA_VERSION",
    "FeatureSet",
    "FeatureSetAppliesTo",
    "FeatureSetIntegrityError",
    "FeatureSetProducedBy",
    "FeatureSetRef",
    "FeatureSetValidationError",
    "validate_feature_set_dict",
]
