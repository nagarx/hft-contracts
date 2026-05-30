"""
Canonical JSON serialization + SHA-256 hashing primitives.

**Single source of truth** for the canonical-form hashing convention used by:

- `hft_ops.ledger.dedup.compute_fingerprint` — experiment fingerprint
- `hft_contracts.provenance.hash_config_dict` — arbitrary config hash
  (moved to hft-contracts in Phase 6 6B.4; hft-ops has a re-export shim)
- `hft_contracts.feature_sets.hashing.compute_feature_set_hash` —
  FeatureSet product hash (moved to hft-contracts in Phase 6 6B.3; hft-ops
  has a re-export shim)
- `hft_evaluator.pipeline.compute_profile_hash` — evaluator profile hash
- `lobtrainer.data.feature_set_resolver._compute_content_hash` — trainer
  delegation (Phase 6 6B.2 retired the inline mirror; trainer now imports
  `canonical_json_blob` + `sha256_hex` from here directly). Drift detector:
  `lob-model-trainer/tests/test_feature_set_resolver.py::TestCanonicalHashGolden`
  (which replaced the deleted `test_feature_set_resolver_parity.py`).

Extracted 2026-04-15 as a Phase 4 Batch 4c hardening measure after an
adversarial architectural audit identified 5 independent implementations
of the same canonical form — a convergent-evolution risk that would have
caused silent hash drift if any single site diverged. Phase 6 6B.2
(2026-04-17) closed the last duplication site (trainer inline).

**Frozen contract** (must never change without a new symbol name):

.. code-block:: python

    canonical_json_blob(obj)  = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    sha256_hex(blob)          = hashlib.sha256(blob).hexdigest()

Rationale for `sort_keys=True, default=str` (and the deliberate EXCLUSIONS):

- ``sort_keys=True``: deterministic dict ordering; required because Python 3.7+
  preserves insertion order but callers should not have to rely on it.
- ``default=str``: defensive fallback for non-JSON-native types (e.g., Path,
  Enum); will only fire for malformed inputs since hashed dicts should already
  be JSON-native.
- NO ``separators=(',',':')``: matches the existing monorepo convention.
  Compact-separator variants would produce different bytes and break existing
  fingerprints; the whitespace convention is load-bearing.
- NO ``ensure_ascii=False``: same rationale. ASCII-only output matches existing
  bytes.

**Portability caveat**: the canonical form is stable across CPython versions
and platforms but is NOT byte-portable to other languages' default JSON
serializers (e.g., Rust ``serde_json`` emits ``","`` item separators where
Python's default emits ``", "``). Cross-language reproduction requires matching
Python's whitespace convention exactly.

**NaN/Inf handling**: by default, ``canonical_json_blob`` passes input through
unchanged — non-finite floats serialize to the non-strict-JSON tokens ``NaN``
/ ``Infinity`` (which are deterministic across CPython but reject-on-load in
strict consumers). Pass ``sanitize=True`` to preprocess the input with
``sanitize_for_hash``, which recursively replaces non-finite floats with
``None`` and coerces numpy scalars/arrays to native Python types. Use
sanitization when the hashed payload carries statistical metrics that
legitimately have NaN-as-missing semantics (e.g., evaluator
``FeatureProfile`` p-values for paths without hypothesis tests).

See ``tests/test_canonical_hash.py`` for the byte-level golden-hash fixtures
that lock this behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

__all__ = [
    "canonical_json_blob",
    "sanitize_for_hash",
    "sha256_hex",
]


def sanitize_for_hash(obj: Any) -> Any:
    """Recursively coerce to canonical, strict-JSON-safe native types.

    Walks dicts (recurse into values), lists and tuples (recurse into items,
    tuples canonicalize to lists — JSON-identical), and floats (NaN/Inf/-Inf
    → None). NumPy scalars and arrays are coerced to their native Python
    equivalents (``np.generic`` → ``.item()``, ``np.ndarray`` → ``.tolist()``)
    before any other handling, then re-walked. All remaining types pass
    through unchanged.

    Rationale (non-finite floats): NaN ≠ NaN by IEEE 754. ``json.dumps``
    emits the non-strict tokens ``NaN`` / ``Infinity`` which ARE deterministic
    across Python versions, but round-tripping through a strict-JSON consumer
    (e.g., a different language reading the hash input) would reject them.
    Mapping non-finite floats to ``None`` keeps canonical form
    strictly-JSON-safe AND semantically correct in the common case (NaN
    p-value = "no hypothesis test run", i.e., absent information).

    Rationale (numpy coercion): a numpy scalar reaching ``json.dumps`` would
    fall through to the ``default=str`` fallback and serialize to its ``str``
    repr — e.g. ``np.int64(5)`` → ``'"5"'`` (a JSON *string*), NOT ``5`` — so
    a numpy value would produce a DIFFERENT, wrong hash than the native value
    every producer already passes after ``float()``/``int()`` boundary
    wrapping. Coercing here makes numpy inputs hash IDENTICALLY to the native
    form. The hash-changing types are ``np.int64`` / ``np.bool_`` /
    ``np.ndarray`` — non-``float`` types the ``float`` branch skips, which
    would otherwise reach ``default=str``. The numpy branches run before the
    ``float`` branch so that ``np.float64`` (a ``float`` subclass) is uniformly
    coerced to native too and routed through the NaN/Inf guard below; note
    ``np.float64`` already serializes to a correct number either way, so this
    ordering is for output uniformity, not hash-correctness. NOTE: this hardens
    only the ``sanitize=True`` path; the default ``sanitize=False`` path is
    unchanged and leaves boundary-coercion to the caller (FeatureSet / dedup /
    cache sites already wrap with ``int()``/``str()``).

    Tuples are canonicalized to lists because JSON has no tuple type; a
    tuple of the same contents always serializes identically to a list
    regardless, so the canonicalization ensures equality-based testing
    matches JSON-serialized behavior.

    Args:
        obj: Any value. NumPy scalars/arrays are coerced to native Python
            first; dicts/lists/tuples are recursed; floats are checked with
            ``math.isfinite``; other types pass through.

    Returns:
        Same structure with numpy types coerced to native, non-finite floats
        replaced by None, and tuples replaced by lists. All other values
        preserved.
    """
    if isinstance(obj, np.ndarray):
        return sanitize_for_hash(obj.tolist())
    if isinstance(obj, np.generic):
        return sanitize_for_hash(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_for_hash(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_hash(v) for v in obj]
    return obj


def canonical_json_blob(obj: Any, *, sanitize: bool = False) -> bytes:
    """Serialize ``obj`` to canonical JSON bytes.

    **Contract** (frozen — never change without a new symbol name):

        canonical_json_blob(obj) ≡ json.dumps(
            obj, sort_keys=True, default=str
        ).encode("utf-8")

    When ``sanitize=True``, the input is preprocessed with
    ``sanitize_for_hash`` first to replace non-finite floats with None and
    canonicalize tuples to lists. Use this for hashes over payloads that
    legitimately carry NaN/Inf (e.g., evaluator profiles).

    Args:
        obj: Any JSON-serializable value (after ``default=str`` fallback).
            Dicts, lists, tuples, primitives are natively supported. Enum
            and Path values serialize via their ``str`` representation.
        sanitize: When True, preprocess with ``sanitize_for_hash``.
            Default False keeps the blob purely structural.

    Returns:
        UTF-8 encoded canonical JSON bytes. Suitable for feeding directly
        to ``hashlib.sha256`` or to any downstream consumer that agrees
        on the canonical form.
    """
    source = sanitize_for_hash(obj) if sanitize else obj
    return json.dumps(source, sort_keys=True, default=str).encode("utf-8")


def sha256_hex(blob: bytes) -> str:
    """Return hex-encoded SHA-256 digest of ``blob``.

    Thin wrapper that matches the monorepo convention (64-char lowercase
    hex, no ``sha256:`` prefix — prefix reserved for external identifiers
    like databento manifests).

    Args:
        blob: Raw bytes. Typically the output of ``canonical_json_blob``.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(blob).hexdigest()
