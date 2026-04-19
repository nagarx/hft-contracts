"""Deprecation shim — forwards to ``hft_contracts.atomic_io``.

REV 2 pre-push hygiene (2026-04-20): the canonical home was renamed
from ``hft_contracts._atomic_io`` (underscore-prefix = "module-internal")
to ``hft_contracts.atomic_io`` (public). The underscore was a
mis-classification — this module is cross-module-consumed by
``hft-ops`` (``feature_sets/writer.py``, ``ledger/ledger.py``),
violating the monorepo's "underscore = module-internal, never
cross-module" rule in root CLAUDE.md.

Migrate to the public name:

    from hft_contracts.atomic_io import atomic_write_json, AtomicWriteError

This shim emits ``DeprecationWarning`` on first access of each symbol
and is scheduled for removal on ``_REMOVAL_DATE`` (``2026-10-31``),
matching the 6-month deprecation window used by
``lobbacktest.data.signal_manifest`` (Phase 6 6B.5).
"""

from __future__ import annotations

import importlib
import warnings as _warnings

_CANONICAL_MODULE = "hft_contracts.atomic_io"
_REMOVAL_DATE = "2026-10-31"
_PUBLIC_NAMES = frozenset({"atomic_write_json", "AtomicWriteError"})
_WARNED: set[str] = set()


def __getattr__(name: str):
    """Lazy re-export with one-time DeprecationWarning per symbol."""
    if name in _PUBLIC_NAMES:
        if name not in _WARNED:
            _WARNED.add(name)
            _warnings.warn(
                f"`hft_contracts._atomic_io.{name}` is a REV 2 pre-push "
                f"deprecation shim. Migrate to "
                f"`from {_CANONICAL_MODULE} import {name}` before the "
                f"{_REMOVAL_DATE} removal deadline. "
                f"(This warning fires once per symbol per process.)",
                DeprecationWarning,
                stacklevel=2,
            )
        return getattr(importlib.import_module(_CANONICAL_MODULE), name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = sorted(_PUBLIC_NAMES)
