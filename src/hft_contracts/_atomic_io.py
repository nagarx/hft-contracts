"""Atomic JSON write primitive — single source of truth.

Phase 7 Stage 7.4 Round 5 (2026-04-20): extracted from three previously
divergent call sites to prevent serialization-convention drift:

- ``hft_contracts.experiment_record._atomic_write_json`` (Round 4; NO
  ``sort_keys``, NO trailing newline).
- ``hft_ops.feature_sets.writer.atomic_write_json`` (Phase 4; ``sort_keys=True``
  + trailing newline).
- ``hft_ops.ledger.ledger._save_index`` (Round 4; inline; NO ``sort_keys``,
  NO trailing newline).

All three now delegate here. ``hft_ops.feature_sets.writer.atomic_write_json``
is a thin re-export for back-compat with pre-Round-5 importers.

**Canonical convention** (locked by tests in hft-contracts + hft-ops):

- ``sort_keys=True`` — deterministic key order for diff tooling, content
  addressing, and cross-run byte equality in golden fixtures.
- ``trailing_newline=True`` — POSIX text-file convention; golden-fixture
  generators at ``lob-model-trainer/tests/fixtures/golden/generate_snapshots.py``
  rely on this.
- ``default=str`` — graceful handling of ``Path``, ``datetime``, ``Enum``.
- ``indent=2`` — golden-fixture convention.

**Protocol**:

1. Resolve target; ensure parent dir exists.
2. Write to ``<path>.tmp.<pid>.<ns_time>`` with unique suffix so concurrent
   writers cannot collide on the tmp file.
3. ``f.flush()`` + ``os.fsync(fd)`` — durability barrier before rename.
4. ``os.replace(tmp, target)`` — atomic on POSIX (single-syscall rename
   within a filesystem); atomic on Windows (``MoveFileEx`` with
   ``MOVEFILE_REPLACE_EXISTING``).
5. Cleanup: on ANY exception (OSError, TypeError, MemoryError,
   KeyboardInterrupt, ...), unlink the tmp file best-effort and re-raise.
   We use ``except BaseException`` deliberately — leaking an orphan tmp
   on Ctrl-C or MemoryError is the worse failure mode.

**I/O invariant note**: hft-contracts now owns ONE I/O utility. This is a
deliberate minimal weakening of the "contract plane is I/O-free"
invariant from Phase 6 6B.3 — it's load-bearing for
``ExperimentRecord.save()`` which the contract plane already owns. If
hft-contracts ever targets no-I/O execution contexts (Rust/PyO3, WASM,
serverless), this module's 3 standard-library imports (``os``, ``time``,
``json``) can be lazy-imported inside the function rather than at module
load time.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Union


class AtomicWriteError(OSError):
    """An atomic write failed after tmp creation but before rename.

    Phase 7 Stage 7.4 Round 5 (2026-04-20): moved from
    ``hft_ops.feature_sets.writer`` to the contract plane so all three
    atomic-write callers share one exception type. ``hft_ops.feature_sets.writer``
    re-exports for back-compat.

    ``IS-A`` ``OSError`` — callers using ``except OSError`` continue to
    match. Raised ONLY for ``OSError`` during the tmp-write / rename
    sequence; ``TypeError`` / ``ValueError`` from ``json.dump`` on an
    un-serializable value propagate unchanged (they represent caller
    bugs, not I/O failures).
    """


def atomic_write_json(
    path: Union[Path, str],
    obj: Any,
    *,
    sort_keys: bool = True,
    indent: int = 2,
    trailing_newline: bool = True,
) -> None:
    """Write a JSON-serializable object to ``path`` atomically.

    Crash-safe: on ANY exception between tmp-file open and
    ``os.replace``, the tmp file is unlinked (best-effort) and the
    exception re-raised. The target file is never left partial — the
    filesystem state is always either "pre-existing content" (if any)
    or "fully-written new content", never an intermediate.

    Concurrent writers to the same target path are isolated via the
    ``{pid}.{time_ns()}`` suffix on the tmp file: each writer produces
    its own tmp, and ``os.replace`` is atomic per-call. "Last writer
    wins" on the final rename — acceptable for ledger index writes and
    record writes, since higher-level locks (sweep_run sequential)
    prevent true concurrent writes in our pipeline.

    Args:
        path: Target file path (parent dirs auto-created). ``str`` is
            coerced to ``Path`` for consistency.
        obj: JSON-serializable value. Non-scalar types (``Path``,
            ``datetime``, ``Enum``) are stringified via ``default=str``.
        sort_keys: Emit keys in sorted order. Default ``True`` for
            deterministic output (diff stability, byte-equal golden
            fixtures). Pass ``False`` only when insertion order is
            load-bearing (rare; document why at the call site).
        indent: JSON indent spaces. Default 2.
        trailing_newline: Append final ``\\n`` after the JSON body.
            Default ``True`` (POSIX convention).

    Raises:
        AtomicWriteError: Wraps ``OSError`` occurring during the
            tmp-write / fsync / rename sequence. Includes the target
            path in the error message for triage. ``IS-A`` ``OSError``.
        TypeError: From ``json.dump`` when ``obj`` contains a
            non-JSON-serializable value that ``default=str`` cannot
            stringify. This is a CALLER bug — sanitize upstream.
        ValueError: From ``json.dump`` on invalid input (rare).
        KeyboardInterrupt / SystemExit / MemoryError: Propagated after
            tmp cleanup. Use this function at the outermost layer of
            signal-handling, not inside a custom ``except`` block that
            might swallow these.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, sort_keys=sort_keys, indent=indent, default=str)
            if trailing_newline:
                f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except OSError as exc:
        # I/O failure during write/fsync/rename — wrap as AtomicWriteError
        # for informative error messages while preserving IS-A OSError.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass  # Best-effort; orphan tmp is a lesser evil than
                  # double-failure mid-cleanup.
        raise AtomicWriteError(
            f"Atomic write failed for {path}: {exc}"
        ) from exc
    except BaseException:
        # Non-OSError (TypeError, KeyboardInterrupt, MemoryError, ...) —
        # cleanup but do NOT wrap. The original exception type is more
        # useful to callers than a synthetic AtomicWriteError here.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


__all__ = ["AtomicWriteError", "atomic_write_json"]
