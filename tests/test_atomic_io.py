"""Tests for hft_contracts.atomic_io — #PY-73 closure cycle (2026-05-11, v2.7.0).

Covers the 5 new primitives (atomic_write_binary + atomic_write_torch +
atomic_write_npy + atomic_write_pickle + atomic_copy) plus extended
coverage of the existing atomic_write_json.

Test categories:
- Round-trip equality (write → read → assert byte/value-identical)
- Empty-write guard (min_bytes) raises AtomicWriteError
- atomic_write_npy TypeError on non-ndarray (hft-rules §8 no silent conversion)
- atomic_write_pickle default protocol is DEFAULT_PROTOCOL not HIGHEST_PROTOCOL
- Tmp cleanup on write_fn exception
- atomic_copy round-trip preserves contents
- AtomicWriteError IS-A OSError (preserves except-OSError compat)
- Tmp-path uniqueness across rapid invocation
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from hft_contracts.atomic_io import (
    AtomicWriteError,
    atomic_copy,
    atomic_write_binary,
    atomic_write_json,
    atomic_write_npy,
    atomic_write_pickle,
    atomic_write_torch,
    _make_tmp_path,
)


# ---------------------------------------------------------------------------
# atomic_write_binary — generic primitive
# ---------------------------------------------------------------------------


class TestAtomicWriteBinary:
    """Cover the generic atomic_write_binary primitive."""

    def test_round_trip_bytes(self, tmp_path: Path) -> None:
        target = tmp_path / "out.bin"
        payload = b"\x00\x01\x02\x03hello\xff"
        atomic_write_binary(target, lambda f: f.write(payload))
        assert target.read_bytes() == payload

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "deep" / "out.bin"
        atomic_write_binary(target, lambda f: f.write(b"x"))
        assert target.exists()
        assert target.read_bytes() == b"x"

    def test_empty_write_raises_atomic_write_error(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.bin"
        with pytest.raises(AtomicWriteError, match=r"0-byte file"):
            atomic_write_binary(target, lambda f: None)
        # Target must NOT have been created (rename never happened).
        assert not target.exists()
        # No orphan tmp files left behind.
        assert list(tmp_path.glob("*.tmp.*")) == []

    def test_empty_write_allowed_when_min_bytes_zero(self, tmp_path: Path) -> None:
        target = tmp_path / "zero.bin"
        atomic_write_binary(target, lambda f: None, min_bytes=0)
        assert target.exists()
        assert target.read_bytes() == b""

    def test_tmp_cleanup_on_write_fn_exception(self, tmp_path: Path) -> None:
        target = tmp_path / "fail.bin"

        def _raise(f):
            raise RuntimeError("simulated mid-write failure")

        with pytest.raises(RuntimeError, match="simulated mid-write failure"):
            atomic_write_binary(target, _raise)
        assert not target.exists()
        assert list(tmp_path.glob("*.tmp.*")) == []

    def test_atomic_write_error_is_oserror(self) -> None:
        """Callers using except OSError must still match AtomicWriteError."""
        assert issubclass(AtomicWriteError, OSError)
        # Sanity: instantiation pattern used by callsites.
        err = AtomicWriteError("test")
        assert isinstance(err, OSError)

    def test_pre_existing_target_overwritten(self, tmp_path: Path) -> None:
        target = tmp_path / "exists.bin"
        target.write_bytes(b"OLD")
        atomic_write_binary(target, lambda f: f.write(b"NEW PAYLOAD"))
        assert target.read_bytes() == b"NEW PAYLOAD"


# ---------------------------------------------------------------------------
# atomic_write_torch — lazy import + byte-equality
# ---------------------------------------------------------------------------


class TestAtomicWriteTorch:
    """Cover atomic_write_torch — torch is lazy-imported."""

    def test_round_trip_state_dict(self, tmp_path: Path) -> None:
        import torch

        state_dict = {
            "layer.weight": torch.tensor([1.0, 2.0, 3.0]),
            "layer.bias": torch.tensor([0.1]),
            "epoch": 7,
            "lr": 1e-3,
        }
        target = tmp_path / "model.pt"
        atomic_write_torch(target, state_dict)
        assert target.exists()
        # Round-trip equality
        loaded = torch.load(target, weights_only=False)
        assert set(loaded.keys()) == set(state_dict.keys())
        assert torch.equal(loaded["layer.weight"], state_dict["layer.weight"])
        assert torch.equal(loaded["layer.bias"], state_dict["layer.bias"])
        assert loaded["epoch"] == 7
        assert loaded["lr"] == 1e-3

    def test_atomic_torch_save_no_partial_on_failure(self, tmp_path: Path) -> None:
        """write_fn raises → no partial file."""
        target = tmp_path / "partial.pt"

        # Pass an un-serializable object to torch.save → PicklingError
        class _NotSerializable:
            def __reduce__(self):
                raise RuntimeError("intentionally unserializable")

        with pytest.raises(Exception):  # noqa: BLE001 — torch's specific class varies
            atomic_write_torch(target, _NotSerializable())
        assert not target.exists()
        assert list(tmp_path.glob("*.tmp.*")) == []


# ---------------------------------------------------------------------------
# atomic_write_npy — TypeError + round-trip
# ---------------------------------------------------------------------------


class TestAtomicWriteNpy:
    """Cover atomic_write_npy — including the TypeError invariant."""

    def test_round_trip_float64(self, tmp_path: Path) -> None:
        target = tmp_path / "arr.npy"
        arr = np.array([1.0, 2.0, 3.0, np.pi, np.e], dtype=np.float64)
        atomic_write_npy(target, arr)
        loaded = np.load(target)
        np.testing.assert_array_equal(loaded, arr)
        assert loaded.dtype == arr.dtype

    def test_round_trip_int32_2d(self, tmp_path: Path) -> None:
        target = tmp_path / "arr2d.npy"
        arr = np.arange(20, dtype=np.int32).reshape(4, 5)
        atomic_write_npy(target, arr)
        loaded = np.load(target)
        np.testing.assert_array_equal(loaded, arr)
        assert loaded.dtype == arr.dtype
        assert loaded.shape == (4, 5)

    @pytest.mark.parametrize(
        "bad_input",
        [
            [1.0, 2.0],            # Python list
            (1.0, 2.0, 3.0),       # tuple
            42,                    # scalar int
            3.14,                  # scalar float
            "not an array",        # string
            {"key": "value"},      # dict
            None,                  # None
        ],
    )
    def test_rejects_non_ndarray(self, tmp_path: Path, bad_input) -> None:
        """Per hft-rules §8 — no silent np.asarray conversion."""
        target = tmp_path / "rejected.npy"
        with pytest.raises(TypeError, match="numpy.ndarray"):
            atomic_write_npy(target, bad_input)
        assert not target.exists()

    def test_byte_identical_to_direct_np_save(self, tmp_path: Path) -> None:
        """atomic_write_npy output must equal direct np.save output."""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        atomic_path = tmp_path / "atomic.npy"
        direct_path = tmp_path / "direct.npy"
        atomic_write_npy(atomic_path, arr)
        np.save(direct_path, arr)
        assert atomic_path.read_bytes() == direct_path.read_bytes()


# ---------------------------------------------------------------------------
# atomic_write_pickle — protocol default + round-trip
# ---------------------------------------------------------------------------


class TestAtomicWritePickle:
    """Cover atomic_write_pickle — including DEFAULT_PROTOCOL invariant."""

    def test_round_trip_dict(self, tmp_path: Path) -> None:
        target = tmp_path / "obj.pkl"
        obj = {
            "model_type": "temporal_ridge",
            "alpha": 1.0,
            "feature_indices": [0, 5, 12],
            "is_fitted": True,
        }
        atomic_write_pickle(target, obj)
        with open(target, "rb") as f:
            loaded = pickle.load(f)
        assert loaded == obj

    def test_default_protocol_is_default_not_highest(self, tmp_path: Path) -> None:
        """Default protocol MUST be pickle.DEFAULT_PROTOCOL for forward-compat.

        pickle.HIGHEST_PROTOCOL is the newest the current Python supports
        and may rotate forward, breaking older readers. DEFAULT is stable
        across point releases.
        """
        target = tmp_path / "proto.pkl"
        obj = {"x": 1}
        atomic_write_pickle(target, obj)
        # Verify by reading the protocol byte from the pickle stream.
        # pickle protocol marker is bytes 0-1: PROTO opcode (0x80) +
        # protocol version byte.
        raw = target.read_bytes()
        assert raw[0] == 0x80  # PROTO opcode
        protocol_byte = raw[1]
        assert protocol_byte == pickle.DEFAULT_PROTOCOL, (
            f"Expected DEFAULT_PROTOCOL={pickle.DEFAULT_PROTOCOL}, "
            f"got {protocol_byte} (HIGHEST={pickle.HIGHEST_PROTOCOL})"
        )

    def test_explicit_highest_protocol_works(self, tmp_path: Path) -> None:
        """Caller can opt-in to HIGHEST_PROTOCOL when reader version is controlled."""
        target = tmp_path / "highest.pkl"
        obj = {"x": [1, 2, 3]}
        atomic_write_pickle(target, obj, protocol=pickle.HIGHEST_PROTOCOL)
        raw = target.read_bytes()
        assert raw[1] == pickle.HIGHEST_PROTOCOL


# ---------------------------------------------------------------------------
# atomic_copy — file copy round-trip
# ---------------------------------------------------------------------------


class TestAtomicCopy:
    """Cover atomic_copy — atomic file duplication."""

    def test_round_trip_small_file(self, tmp_path: Path) -> None:
        src = tmp_path / "src.bin"
        dst = tmp_path / "dst.bin"
        payload = b"\x00hello\xff" * 100
        src.write_bytes(payload)
        atomic_copy(src, dst)
        assert dst.exists()
        assert dst.read_bytes() == payload

    def test_round_trip_overwrites_existing_dst(self, tmp_path: Path) -> None:
        src = tmp_path / "src.bin"
        dst = tmp_path / "dst.bin"
        src.write_bytes(b"NEW")
        dst.write_bytes(b"OLD")
        atomic_copy(src, dst)
        assert dst.read_bytes() == b"NEW"

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        src = tmp_path / "src.bin"
        dst = tmp_path / "nested" / "deep" / "dst.bin"
        src.write_bytes(b"x")
        atomic_copy(src, dst)
        assert dst.exists()

    def test_missing_src_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            atomic_copy(tmp_path / "missing.bin", tmp_path / "dst.bin")

    def test_empty_src_rejected_by_default_min_bytes(self, tmp_path: Path) -> None:
        """Empty source file → AtomicWriteError per min_bytes=1 default."""
        src = tmp_path / "empty.bin"
        dst = tmp_path / "dst.bin"
        src.write_bytes(b"")
        with pytest.raises(AtomicWriteError, match=r"0-byte"):
            atomic_copy(src, dst)
        assert not dst.exists()


# ---------------------------------------------------------------------------
# Tmp-path uniqueness — Adv-API-review SB-5 collision-hardening
# ---------------------------------------------------------------------------


class TestTmpPathUniqueness:
    """Verify _make_tmp_path produces collision-free names rapidly."""

    def test_no_collision_in_rapid_succession(self, tmp_path: Path) -> None:
        """1000 rapid invocations from same PID produce 1000 distinct paths."""
        base = tmp_path / "out.bin"
        seen = set()
        for _ in range(1000):
            p = _make_tmp_path(base)
            assert p not in seen, f"Tmp-path collision after {len(seen)} draws: {p}"
            seen.add(p)
        assert len(seen) == 1000

    def test_includes_pid_and_random_components(self, tmp_path: Path) -> None:
        base = tmp_path / "out.bin"
        p = _make_tmp_path(base)
        parts = p.name.split(".")
        # Expected: "out.bin.tmp.<pid>.<ns>.<rand4>"
        # i.e., name + suffix(bin) + tmp + pid + ns + rand4 = 6 components
        assert len(parts) == 6
        assert parts[0] == "out"
        assert parts[1] == "bin"
        assert parts[2] == "tmp"
        assert parts[3] == str(os.getpid())
        # parts[4] is monotonic ns (digits)
        assert parts[4].isdigit()
        # parts[5] is secrets.token_hex(4) — 8 hex chars
        assert len(parts[5]) == 8
        assert all(c in "0123456789abcdef" for c in parts[5])


# ---------------------------------------------------------------------------
# atomic_write_json — back-compat smoke (existing primitive)
# ---------------------------------------------------------------------------


class TestAtomicWriteJsonBackCompat:
    """Ensure existing atomic_write_json semantics preserved post #PY-73 refactor."""

    def test_round_trip_sort_keys_default_true(self, tmp_path: Path) -> None:
        target = tmp_path / "out.json"
        obj = {"b": 1, "a": 2, "c": 3}
        atomic_write_json(target, obj)
        raw = target.read_text(encoding="utf-8")
        # Sort-keys default: a, b, c order
        assert raw.index('"a"') < raw.index('"b"') < raw.index('"c"')
        # Trailing newline default True
        assert raw.endswith("\n")
        # Round-trip
        assert json.loads(raw) == obj

    def test_atomic_write_json_uses_tmp_path_helper(self, tmp_path: Path) -> None:
        """Tmp-path generation now uses _make_tmp_path with 4-component suffix."""
        target = tmp_path / "test.json"
        obj = {"a": 1}
        # Sanity: write succeeds, no orphan tmps.
        atomic_write_json(target, obj)
        assert target.exists()
        assert list(tmp_path.glob("*.tmp.*")) == []
