"""Phase 0 benchmark fixture generator.

Deterministic (seed=42) synthetic data matching MBO + BASIC export contracts.
Regenerate via: python -m pytest hft-contracts/tests/fixtures/phase0_benchmark/generate.py::main
Or directly:   python hft-contracts/tests/fixtures/phase0_benchmark/generate.py

These fixtures are the regression baseline for the Architectural Hardening Cycle (plan v2.0).
Every Phase I/II/III/IV refactor regresses forward-pass / fingerprint / serialization
outputs against the golden values stored here. Any unintended drift fails CI.

Acceptable-drift policy:
  - Any phase that INTENTIONALLY changes a golden value MUST update golden_values.json
    in the same commit and document the delta in the phase CHANGELOG entry.
  - Silent value changes (no CHANGELOG delta + no golden-values update) = CI failure.

Fixtures produced:
  - synthetic_mbo.npz       (N=10, T=20, F=98)        — regression labels, forward_prices
  - synthetic_basic.npz     (N=10, T=20, F=34)        — point-return regression labels
  - fixture_metadata_mbo.json         — matches real dataset_manifest contract
  - fixture_metadata_basic.json       — matches real BASIC metadata contract
  - golden_values.json                — populated by downstream phase tests

Seed convention: seed=42 everywhere. numpy.random.default_rng(42) is the canonical RNG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

FIXTURE_DIR = Path(__file__).parent
SEED = 42

# Dimensions
N_SAMPLES = 10
WINDOW_SIZE = 20
MBO_FEATURE_COUNT = 98
BASIC_FEATURE_COUNT = 34
MBO_HORIZONS = [10, 60, 300]
BASIC_HORIZONS = [1, 2, 3, 5, 10, 20, 30, 60]
SMOOTHING_WINDOW_OFFSET = 5


def _rng() -> np.random.Generator:
    """Canonical deterministic RNG. Call afresh to avoid state leakage between fixtures."""
    return np.random.default_rng(SEED)


def _sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _compute_array_hash(arr: np.ndarray) -> str:
    """Hash of raw bytes for cross-reference in golden_values.json."""
    return _sha256_hex_bytes(arr.tobytes())


def generate_mbo_fixture() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Synthetic MBO export fixture. Returns (arrays_dict, metadata_dict)."""
    rng = _rng()

    # Sequences: [N, T, F] float32. Small magnitudes to be realistic.
    sequences = rng.standard_normal(
        (N_SAMPLES, WINDOW_SIZE, MBO_FEATURE_COUNT), dtype=np.float32
    ) * 0.5

    # Positive values in size features (indices 10-19, 30-39) — size must be positive.
    size_indices = list(range(10, 20)) + list(range(30, 40))
    sequences[:, :, size_indices] = np.abs(sequences[:, :, size_indices]) + 1.0

    # Mid prices (feature index 40) positive around $100 (matches NVDA scale roughly).
    sequences[:, :, 40] = np.abs(sequences[:, :, 40]) + 100.0

    # Spread_bps (index 42): small positive. Real values typically 0.5-10 bps.
    sequences[:, :, 42] = np.abs(sequences[:, :, 42]) + 0.8

    # Regression labels: [N, H] float64 bps. Heavy-tail via Student-t-ish shape.
    t_samples = rng.standard_t(df=3.0, size=(N_SAMPLES, len(MBO_HORIZONS)))
    regression_labels = (t_samples * 5.0).astype(np.float64)

    # Forward prices: [N, smoothing_offset + max_H + 1] float64 USD.
    max_h = max(MBO_HORIZONS)
    n_cols = SMOOTHING_WINDOW_OFFSET + max_h + 1
    base_prices = 100.0 + rng.standard_normal(N_SAMPLES) * 0.5
    # Price trajectory is base + cumulative tiny increments
    increments = rng.standard_normal((N_SAMPLES, n_cols - 1)) * 0.01
    forward_prices = np.zeros((N_SAMPLES, n_cols), dtype=np.float64)
    forward_prices[:, 0] = base_prices
    forward_prices[:, 1:] = base_prices[:, None] + np.cumsum(increments, axis=1)

    arrays = {
        "sequences": sequences,
        "regression_labels": regression_labels,
        "forward_prices": forward_prices,
    }

    metadata: Dict[str, Any] = {
        "contract_version": "2.2",
        "day": "20260420_phase0_mbo",
        "export_timestamp": datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc).isoformat(),
        "forward_prices": {
            "column_layout": f"col_0=t-{SMOOTHING_WINDOW_OFFSET}, col_{SMOOTHING_WINDOW_OFFSET}=t, col_{SMOOTHING_WINDOW_OFFSET + max_h}=t+max_H",
            "exported": True,
            "max_horizon": max_h,
            "n_columns": n_cols,
            "smoothing_window_offset": SMOOTHING_WINDOW_OFFSET,
            "units": "USD",
        },
        "label_dtype": "float64",
        "label_strategy": "regression",
        "labeling": {
            "horizons": list(MBO_HORIZONS),
            "label_encoding": {
                "description": "PointReturn forward return in bps at each horizon (synthetic)",
                "dtype": "float64",
                "format": "continuous_bps",
                "unit": "basis_points",
            },
            "label_mode": "regression",
            "num_horizons": len(MBO_HORIZONS),
            "return_type": "PointReturn",
        },
        "n_features": MBO_FEATURE_COUNT,
        "n_sequences": N_SAMPLES,
        "normalization": {
            "applied": False,
            "feature_layout": "raw_ask_prices_10_ask_sizes_10_bid_prices_10_bid_sizes_10",
            "levels": 10,
            "params_file": None,
            "sample_count": N_SAMPLES,
            "strategy": "none",
        },
        "provenance": {
            "config_hash": "phase0_synthetic_mbo_v1",
            "contract_version": "2.2",
            "export_timestamp_utc": datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc).isoformat(),
            "extractor_version": "phase0_synthetic",
            "git_commit": "phase0_deterministic",
            "git_dirty": False,
        },
        "schema_version": "2.2",
        "tensor_format": None,
        "validation": {
            "label_range_valid": True,
            "no_nan_inf": True,
            "sequences_labels_match": True,
            "values_scanned": N_SAMPLES * WINDOW_SIZE * MBO_FEATURE_COUNT,
        },
        "window_size": WINDOW_SIZE,
        "data_source": "mbo_lob_phase0_synthetic",
        "seed": SEED,
    }

    return arrays, metadata


def generate_basic_fixture() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Synthetic BASIC (off-exchange) export fixture. Returns (arrays_dict, metadata_dict).

    BASIC has point-return regression labels at 8 horizons; different index layout from MBO.
    """
    rng = _rng()

    # Sequences: [N, T, F=34] float32
    sequences = rng.standard_normal(
        (N_SAMPLES, WINDOW_SIZE, BASIC_FEATURE_COUNT), dtype=np.float32
    ) * 0.3

    # BASIC has its own feature layout — see hft_contracts.OffExchangeFeatureIndex.
    # For synthetic purposes, ensure a positive "mid_price" at position 0 and
    # positive "spread_bps" somewhere. Real layout defined in off-exchange contract.
    # Keeping this generic — trainers will validate against real indices at load.
    sequences[:, :, 0] = np.abs(sequences[:, :, 0]) + 100.0  # pseudo mid_price
    sequences[:, :, 1] = np.abs(sequences[:, :, 1]) + 0.8    # pseudo spread_bps

    # Point-return labels: [N, H=8] float64 bps
    t_samples = rng.standard_t(df=4.0, size=(N_SAMPLES, len(BASIC_HORIZONS)))
    labels = (t_samples * 3.0).astype(np.float64)

    # Forward prices for BASIC: [N, max_H + 1]
    max_h = max(BASIC_HORIZONS)
    n_cols = max_h + 1
    base_prices = 100.0 + rng.standard_normal(N_SAMPLES) * 0.5
    increments = rng.standard_normal((N_SAMPLES, n_cols - 1)) * 0.01
    forward_prices = np.zeros((N_SAMPLES, n_cols), dtype=np.float64)
    forward_prices[:, 0] = base_prices
    forward_prices[:, 1:] = base_prices[:, None] + np.cumsum(increments, axis=1)

    arrays = {
        "sequences": sequences,
        "labels": labels,
        "forward_prices": forward_prices,
    }

    metadata: Dict[str, Any] = {
        "contract_version": "2.2",
        "day": "20260420_phase0_basic",
        "export_timestamp": datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc).isoformat(),
        "data_source": "off_exchange_phase0_synthetic",
        "n_features": BASIC_FEATURE_COUNT,
        "n_sequences": N_SAMPLES,
        "window_size": WINDOW_SIZE,
        "bin_size_seconds": 60,
        "horizons": list(BASIC_HORIZONS),
        "label_strategy": "point_return",
        "label_dtype": "float64",
        "normalization": {
            "strategy": "none",
            "applied": False,
        },
        "provenance": {
            "config_hash": "phase0_synthetic_basic_v1",
            "contract_version": "2.2",
            "export_timestamp_utc": datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc).isoformat(),
            "extractor_version": "phase0_synthetic",
            "git_commit": "phase0_deterministic",
            "git_dirty": False,
        },
        "schema_version": "2.2",
        "seed": SEED,
    }

    return arrays, metadata


def write_fixture(
    name: str,
    arrays: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Persist fixture. Returns golden-value entry (array hashes + metadata hash)."""
    npz_path = output_dir / f"{name}.npz"
    meta_path = output_dir / f"fixture_metadata_{name.replace('synthetic_', '')}.json"

    np.savez_compressed(npz_path, **arrays)

    # Canonical metadata JSON (sort_keys for reproducibility)
    meta_json = json.dumps(metadata, indent=2, sort_keys=True)
    meta_path.write_text(meta_json)

    # Golden values: hash per array + metadata hash
    golden: Dict[str, Any] = {
        "npz_file": npz_path.name,
        "metadata_file": meta_path.name,
        "array_hashes": {k: _compute_array_hash(v) for k, v in arrays.items()},
        "array_shapes": {k: list(v.shape) for k, v in arrays.items()},
        "array_dtypes": {k: str(v.dtype) for k, v in arrays.items()},
        "metadata_sha256": _sha256_hex_bytes(meta_json.encode("utf-8")),
    }
    return golden


def main(output_dir: Path | None = None) -> Dict[str, Any]:
    """Generate all Phase 0 fixtures. Returns golden_values dict."""
    if output_dir is None:
        output_dir = FIXTURE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    mbo_arrays, mbo_meta = generate_mbo_fixture()
    basic_arrays, basic_meta = generate_basic_fixture()

    golden: Dict[str, Any] = {
        "schema_version": "1",
        "generator": "phase0_benchmark/generate.py",
        "generated_at_utc": datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc).isoformat(),
        "seed": SEED,
        "description": (
            "Deterministic synthetic fixtures for the Architectural Hardening Cycle "
            "(plan v2.0). Arrays hashes + metadata hash are pinned here. "
            "Downstream phases populate 'forward_pass' and 'compute_loss' sub-blocks "
            "as their models land. See README.md for acceptable-drift policy."
        ),
        "mbo": write_fixture("synthetic_mbo", mbo_arrays, mbo_meta, output_dir),
        "basic": write_fixture("synthetic_basic", basic_arrays, basic_meta, output_dir),
        # Populated by each phase as models/hashes become stable.
        "forward_pass": {
            # Populated by I.A: "hmhp_agreement_score" (fixed by FRESH-2),
            # "hmhp_nonzero_fraction", "xgboost_best_iteration" (fixed by P0-1), etc.
            # Populated by I.B: per-model "compute_loss_value" (on fixture input).
            # Each entry documents the phase that pinned it and the phase that last changed it.
        },
        "signal_boundary": {
            # Populated by Phase II: "compatibility_fingerprint" on a reference
            # signal export constructed from this fixture.
        },
    }

    golden_path = output_dir / "golden_values.json"
    golden_path.write_text(json.dumps(golden, indent=2, sort_keys=True))

    return golden


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Phase 0 benchmark fixtures")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: this script's directory)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Regenerate and assert no drift vs existing golden_values.json",
    )
    args = parser.parse_args()

    if args.verify:
        existing_path = (args.output_dir or FIXTURE_DIR) / "golden_values.json"
        if not existing_path.exists():
            raise SystemExit("golden_values.json missing — run without --verify first")
        existing = json.loads(existing_path.read_text())
        # Generate into temp location
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            regenerated = main(output_dir=tmp_dir)

        # Compare hashes
        for pipeline in ("mbo", "basic"):
            for key in ("array_hashes", "metadata_sha256"):
                if existing[pipeline][key] != regenerated[pipeline][key]:
                    raise SystemExit(
                        f"DRIFT DETECTED in {pipeline}.{key}:\n"
                        f"  committed: {existing[pipeline][key]}\n"
                        f"  regenerated: {regenerated[pipeline][key]}\n"
                        f"Fix: either the environment is non-deterministic OR the "
                        f"generator code changed. Regenerate fixtures and commit if "
                        f"the change is intentional."
                    )
        print("OK: committed golden_values.json matches regeneration (no drift)")
    else:
        golden = main(args.output_dir)
        print(f"Generated fixtures in {args.output_dir or FIXTURE_DIR}")
        print(f"  MBO: shape={golden['mbo']['array_shapes']['sequences']}")
        print(f"  BASIC: shape={golden['basic']['array_shapes']['sequences']}")
        print(f"  Golden values: {golden.get('generated_at_utc')}")
