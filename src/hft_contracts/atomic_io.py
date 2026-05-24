"""Atomic write primitives — single source of truth.

Phase 7 Stage 7.4 Round 5 (2026-04-20): extracted ``atomic_write_json``
from three previously divergent call sites to prevent
serialization-convention drift:

- ``hft_contracts.experiment_record._atomic_write_json`` (Round 4; NO
  ``sort_keys``, NO trailing newline).
- ``hft_ops.feature_sets.writer.atomic_write_json`` (Phase 4; ``sort_keys=True``
  + trailing newline).
- ``hft_ops.ledger.ledger._save_index`` (Round 4; inline; NO ``sort_keys``,
  NO trailing newline).

All three now delegate here. ``hft_ops.feature_sets.writer.atomic_write_json``
is a thin re-export for back-compat with pre-Round-5 importers.

REV 2 pre-push (2026-04-20): this module was originally published as
``hft_contracts._atomic_io``. The underscore-prefix was a
mis-classification since hft-ops imports from it across the module
boundary. Renamed to the public ``hft_contracts.atomic_io`` before the
first GitHub public push; ``hft_contracts._atomic_io`` remains as a
deprecation shim (removal 2026-10-31).

**#PY-73 closure cycle (2026-05-11)**: extended with ``atomic_write_binary``
generic primitive + 3 typed wrappers (``atomic_write_torch``,
``atomic_write_npy``, ``atomic_write_pickle``) + ``atomic_copy`` helper.
Closes 20 non-atomic write sites across lob-model-trainer + lob-models +
lob-backtester that SIGKILL mid-write could corrupt (poisoning
content-addressed cache cells per ``hft-ops/extraction_cache``). Mirrors
``atomic_write_json`` semantics: tmp + fsync + ``os.replace`` +
BaseException-safe cleanup. **Lazy-imports** ``torch`` inside
``atomic_write_torch`` body so hft-contracts retains its torch-free
invariant at module load time (verified by ``test_atomic_io_imports.py``
AST regression test). ``numpy`` IS already a hft-contracts dependency
(via ``label_factory`` + ``signal_manifest``) so ``atomic_write_npy``
imports it top-level.

**Canonical convention** (locked by tests in hft-contracts + hft-ops):

- ``sort_keys=True`` — deterministic key order for diff tooling, content
  addressing, and cross-run byte equality in golden fixtures (JSON only).
- ``trailing_newline=True`` — POSIX text-file convention; golden-fixture
  generators at ``lob-model-trainer/tests/fixtures/golden/generate_snapshots.py``
  rely on this (JSON only).
- ``default=str`` — graceful handling of ``Path``, ``datetime``, ``Enum``
  (JSON only).

  **Caller-responsibility note** (#PY-371 cycle 2026-05-24; L101 lesson):
  ``default=str`` SILENTLY COERCES any non-JSON-native type (``set``,
  custom objects, ``bytes``, ...) into its ``repr`` string. This is
  graceful for the common Path/datetime/Enum case but VIOLATES hft-rules
  §8 ("never silently drop, clamp, or fix data") when callers may pass
  user-supplied dicts containing unexpected types. When fail-loud-on-
  bad-types semantics are required, the caller MUST pre-validate via a
  bare ``json.dumps(obj)`` call BEFORE invoking ``atomic_write_json``.
  ``json.dumps`` without ``default=`` raises ``TypeError`` on unsupported
  types — preserving §8 fail-loud at the caller boundary while still
  gaining tmp+fsync+os.replace atomicity from this SSoT for the actual
  write. **Canonical exemplar**:
  ``databento-ingest/src/databento_ingest/manifest.py:80-81`` —
  pre-validates a caller-supplied ``metadata: dict`` before atomic write.
- ``indent=2`` — golden-fixture convention (JSON only).
- ``min_bytes=1`` — empty-write guard for binary primitives, per
  hft-rules §8 ("never silently drop, clamp, or fix data without
  recording diagnostics"). 0-byte tmp file is treated as caller bug.
- ``pickle.DEFAULT_PROTOCOL`` not ``HIGHEST_PROTOCOL`` — DEFAULT is
  stable across point releases; HIGHEST may rotate forward and break
  older readers. Caller may pass ``HIGHEST_PROTOCOL`` explicitly when
  forward-compat is not needed.

**Protocol** (all primitives):

1. Resolve target; ensure parent dir exists.
2. Write to ``<path>.tmp.<pid>.<ns_time>.<rand4>`` with unique 3-tuple
   suffix so concurrent writers cannot collide on the tmp file. The
   ``secrets.token_hex(4)`` component hardens against PID-recycle +
   coarse ``time_ns()`` granularity on macOS (~1µs).
3. ``f.flush()`` + ``os.fsync(fd)`` — durability barrier before rename.
4. Empty-write guard (binary primitives): ``tmp_path.stat().st_size >=
   min_bytes`` BEFORE rename. Raise ``AtomicWriteError`` otherwise.
5. ``os.replace(tmp, target)`` — atomic on POSIX (single-syscall rename
   within a filesystem); atomic on Windows (``MoveFileEx`` with
   ``MOVEFILE_REPLACE_EXISTING``). Cross-filesystem rename raises
   ``OSError(EXDEV)`` — caller must ensure tmp + target are on the
   same mount.
6. Cleanup: on ANY exception (OSError, TypeError, MemoryError,
   KeyboardInterrupt, ...), unlink the tmp file best-effort and re-raise.
   We use ``except BaseException`` deliberately — leaking an orphan tmp
   on Ctrl-C or MemoryError is the worse failure mode.

**NFS / SMB / FUSE caveat**: ``os.replace`` atomicity is
filesystem-dependent. POSIX local + NFSv3+ are guaranteed; SMB/CIFS not
guaranteed; cloud-FUSE varies. Use local SSD for checkpoint storage on
non-POSIX networks.

**I/O invariant note**: hft-contracts now owns ONE I/O utility module.
This is a deliberate minimal weakening of the "contract plane is I/O-free"
invariant from Phase 6 6B.3 — load-bearing for ``ExperimentRecord.save()``
which the contract plane already owns. The #PY-73 wrappers preserve the
torch-free invariant via lazy import; ``numpy`` is already a declared
dep so its top-level import is safe.
"""

from __future__ import annotations

import json
import os
import pickle
import secrets
import time
from pathlib import Path
from typing import Any, BinaryIO, Callable, Union

import numpy as np


class AtomicWriteError(OSError):
    """An atomic write failed after tmp creation but before rename.

    Phase 7 Stage 7.4 Round 5 (2026-04-20): moved from
    ``hft_ops.feature_sets.writer`` to the contract plane so all three
    atomic-write callers share one exception type. ``hft_ops.feature_sets.writer``
    re-exports for back-compat.

    ``IS-A`` ``OSError`` — callers using ``except OSError`` continue to
    match. Raised for ``OSError`` during the tmp-write / rename
    sequence, AND for the empty-write guard (``tmp_path.stat().st_size
    < min_bytes``) in binary primitives. ``TypeError`` / ``ValueError``
    from ``json.dump`` / ``torch.save`` / ``pickle.dump`` on an
    un-serializable value propagate unchanged (they represent caller
    bugs, not I/O failures).
    """


def _make_tmp_path(path: Path) -> Path:
    """Generate a unique tmp path: ``<name>.tmp.<pid>.<time_ns>.<rand4>``.

    The 3-tuple suffix (PID + monotonic-ish ns + 8-hex random)
    hardens against PID-recycle on long-running processes and coarse
    ``time_ns()`` granularity (~1µs on macOS per
    ``time.get_clock_info('monotonic')``).
    """
    return path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}.{secrets.token_hex(4)}"
    )


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
    3-tuple ``{pid}.{time_ns()}.{rand4}`` suffix on the tmp file: each
    writer produces its own tmp, and ``os.replace`` is atomic per-call.
    "Last writer wins" on the final rename — acceptable for ledger
    index writes and record writes, since higher-level locks
    (sweep_run sequential) prevent true concurrent writes in our
    pipeline.

    See module docstring for the full atomic-write Protocol and NFS
    caveat.

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
    tmp_path = _make_tmp_path(path)
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


def atomic_write_binary(
    path: Union[Path, str],
    write_fn: Callable[[BinaryIO], None],
    *,
    min_bytes: int = 1,
) -> None:
    """Atomically write binary content via tmp + fsync + ``os.replace``.

    Generic primitive underlying the typed wrappers
    (``atomic_write_torch`` / ``atomic_write_npy`` /
    ``atomic_write_pickle``). Caller-supplied ``write_fn`` receives a
    binary file handle opened for write; it must call
    ``f.write(...)``, ``torch.save(obj, f)``, ``np.save(f, arr)``,
    ``pickle.dump(obj, f)``, etc.

    Crash-safe per BaseException-safe cleanup: ANY exception between
    tmp-file open and ``os.replace`` leaves the target untouched and
    the tmp file unlinked (best-effort). See module docstring for full
    Protocol.

    Args:
        path: Target file path (parent dirs auto-created).
        write_fn: Callable receiving an opened-for-write binary file
            handle. Must perform the actual serialization. Should NOT
            close the handle (the ``with`` block here owns it).
        min_bytes: Minimum size of the tmp file (in bytes) after
            ``write_fn`` returns. Empty/silent writes raise
            ``AtomicWriteError`` per hft-rules §8 (default 1; pass 0
            to disable the guard, e.g., for legitimately-empty
            sentinel writes).

    Raises:
        AtomicWriteError: Wraps ``OSError`` during the
            tmp-write/fsync/rename sequence, OR fires if the tmp
            file's size is below ``min_bytes`` after ``write_fn``
            returns (empty-write guard).
        TypeError / ValueError / pickle.PickleError: Propagated from
            ``write_fn`` unchanged (caller bug — un-serializable
            obj, wrong shape, etc.).
        BaseException: Propagated after tmp cleanup.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _make_tmp_path(path)
    try:
        with open(tmp_path, "wb") as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        # Empty-write guard per hft-rules §8: write_fn that produced
        # no payload is a caller bug (silent-corruption hazard).
        size = tmp_path.stat().st_size
        if size < min_bytes:
            # Clean up tmp + raise inside the try (caught by outer
            # OSError handler which surfaces AtomicWriteError).
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise AtomicWriteError(
                f"Atomic binary write produced {size}-byte file "
                f"(min_bytes={min_bytes}) at {path}. write_fn likely no-op."
            )
        os.replace(tmp_path, path)
    except AtomicWriteError:
        # Already wrapped (e.g., empty-write guard) — re-raise as-is.
        raise
    except OSError as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise AtomicWriteError(
            f"Atomic binary write failed for {path}: {exc}"
        ) from exc
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_torch(
    path: Union[Path, str],
    obj: Any,
    *,
    _use_new_zipfile_serialization: bool = True,
) -> None:
    """Atomic ``torch.save`` via ``atomic_write_binary``.

    Lazy-imports ``torch`` to preserve hft-contracts' torch-free
    invariant at module load time. The AST regression test at
    ``tests/test_atomic_io_imports.py::test_atomic_io_torch_free``
    locks this contract.

    Closes #PY-73 producer-side corruption hazard for the 2 ``torch.save``
    sites in lob-model-trainer (``trainer.py:1384`` Trainer.save_checkpoint
    and ``callbacks.py:680`` ModelCheckpoint._save_checkpoint).

    Args:
        path: Target ``.pt`` file path.
        obj: torch-serializable object (state_dict, dict containing
            tensors, or pickled module). ``torch.save`` itself
            validates serializability.
        _use_new_zipfile_serialization: Pass-through to ``torch.save``.
            Default ``True`` (PyTorch ≥1.6 ZIP-archive format). Leading
            underscore matches ``torch.save`` signature convention.

    Raises:
        AtomicWriteError: See ``atomic_write_binary``.
        Various torch errors (RuntimeError, TypeError, ...): Propagated
            from ``torch.save`` unchanged.
    """
    import torch  # LAZY — preserves hft-ops torch-free invariant

    def _write(f: BinaryIO) -> None:
        torch.save(
            obj,
            f,
            _use_new_zipfile_serialization=_use_new_zipfile_serialization,
        )

    atomic_write_binary(path, _write)


def atomic_write_npy(
    path: Union[Path, str],
    arr: np.ndarray,
    *,
    allow_pickle: bool = False,
) -> None:
    """Atomic ``np.save`` via ``atomic_write_binary``.

    Rejects non-ndarray inputs explicitly per hft-rules §8 (no silent
    conversion). Caller must pass ``np.asarray(...)`` at the call site
    if conversion is intended.

    Closes #PY-73 producer-side corruption hazard for the ~15
    ``np.save`` sites across lob-model-trainer (4 in
    ``simple_trainer.py:761-764`` + 9 in ``exporter.py:497-529``) and
    lob-backtester (``registry.py:120``).

    Args:
        path: Target ``.npy`` file path.
        arr: numpy ndarray. **TypeError** raised on non-ndarray
            (no implicit ``np.asarray`` conversion).
        allow_pickle: Pass-through to ``np.save``. Default ``False``
            (numpy ≥1.16.3 convention — reject object dtypes that
            require pickle).

    Raises:
        TypeError: ``arr`` is not a ``numpy.ndarray``.
        AtomicWriteError: See ``atomic_write_binary``.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"atomic_write_npy requires numpy.ndarray, got "
            f"{type(arr).__name__}. Pass np.asarray(...) at the call "
            f"site if conversion is intended."
        )

    def _write(f: BinaryIO) -> None:
        np.save(f, arr, allow_pickle=allow_pickle)

    atomic_write_binary(path, _write)


def atomic_write_pickle(
    path: Union[Path, str],
    obj: Any,
    *,
    protocol: int = pickle.DEFAULT_PROTOCOL,
) -> None:
    """Atomic ``pickle.dump`` via ``atomic_write_binary``.

    Default protocol is ``pickle.DEFAULT_PROTOCOL`` (stable across
    point releases — currently 5 on Python ≥3.8), NOT
    ``pickle.HIGHEST_PROTOCOL`` (the newest supported, which may
    rotate forward and break older readers). Caller may explicitly
    pass ``HIGHEST_PROTOCOL`` when reader Python version is
    controlled.

    Closes #PY-73 producer-side corruption hazard for the 2
    ``pickle.dump`` sites (``lob-models/.../base_simple.py:80``
    BaseSimpleModel.save and
    ``MBO-LOB-analyzer/.../orchestrator.py:224`` — the latter is
    foreign-agent state and DEFERRED to a separate cycle).

    Args:
        path: Target ``.pkl`` file path.
        obj: Picklable object.
        protocol: pickle protocol version. Default
            ``pickle.DEFAULT_PROTOCOL`` (forward-compat preserved).
            Use ``pickle.HIGHEST_PROTOCOL`` for max performance when
            reader version is controlled.

    Raises:
        pickle.PickleError / TypeError: Propagated from ``pickle.dump``
            unchanged (non-picklable obj).
        AtomicWriteError: See ``atomic_write_binary``.
    """
    def _write(f: BinaryIO) -> None:
        pickle.dump(obj, f, protocol=protocol)

    atomic_write_binary(path, _write)


def atomic_copy(
    src: Union[Path, str],
    dst: Union[Path, str],
    *,
    min_bytes: int = 1,
) -> None:
    """Atomic file copy via tmp + ``os.replace``.

    Reads ``src`` bytes into a tmp file alongside ``dst``, then
    atomic-renames. Mirrors ``shutil.copy`` semantics but eliminates
    the partial-write window that SIGKILL-mid-copy would leave on a
    bare ``shutil.copy``.

    Does NOT preserve file metadata (st_mode / atime / mtime). If
    metadata preservation is needed, use the existing
    ``hft_ops/ledger/ledger.py:611`` ``shutil.copy2 → tmp → os.replace``
    pattern (which is already atomic).

    Closes #PY-73 producer-side corruption hazard for the 1
    ``shutil.copy`` site at ``lob-model-trainer/callbacks.py:689``
    (ModelCheckpoint copies ``<epoch>.pt`` to ``best.pt``).

    Args:
        src: Source file path.
        dst: Destination file path (parent auto-created).
        min_bytes: Empty-write guard (per ``atomic_write_binary``).
            Default 1.

    Raises:
        FileNotFoundError: ``src`` does not exist (raised by ``open``).
        AtomicWriteError: See ``atomic_write_binary``.
    """
    import shutil

    src = Path(src)
    with open(src, "rb") as src_f:
        atomic_write_binary(
            dst,
            lambda dst_f: shutil.copyfileobj(src_f, dst_f),
            min_bytes=min_bytes,
        )


__all__ = [
    "AtomicWriteError",
    "atomic_write_json",
    "atomic_write_binary",
    "atomic_write_torch",
    "atomic_write_npy",
    "atomic_write_pickle",
    "atomic_copy",
]
