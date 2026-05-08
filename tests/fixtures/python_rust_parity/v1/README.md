# Python ↔ Rust LabelFactory Parity Fixture v1

Closes **CR6 / NEW-CRITICAL-2** (root `CLAUDE.md` cited "max diff 7.56e-12 observed" between Python `LabelFactory.smoothed_return` and Rust `MultiHorizonLabelGenerator` without an executable test backing the claim).

## What this fixture does

Provides a checked-in JSON ground-truth for cross-language formula parity testing across:
- `LabelFactory.smoothed_return` ↔ `MultiHorizonLabelGenerator.generate_labels()` (pct_change extracted from third tuple element)
- `LabelFactory.point_return` ↔ `MagnitudeGenerator.compute_returns().point_return`
- `LabelFactory.mean_return` ↔ `MagnitudeGenerator.compute_returns().mean_return`
- `LabelFactory.peak_return` ↔ `MagnitudeGenerator.compute_returns().dominant_return()` (tie-break + sign convention verified equivalent in Wave4 audit Q5)

## Fixture format (schema_version=1)

```json
{
  "schema_version": 1,
  "rust_commit": "<git SHA at generation time, or empty>",
  "rust_crate_version": "0.1.0",
  "generated_at_utc": "<ISO-8601 UTC>",
  "description": "...",
  "scenarios": [
    {
      "name": "monotone_k2_h2_n5",
      "description": "...",
      "k": 2,
      "horizons": [2],
      "n_samples": 5,
      "prices_total": 11,
      "prices_flat": [...],            // [n_samples + 2k + max_h] f64
      "forward_prices_2d": [...],      // [n_samples, k+max_h+1] f64
      "rust_labels_decimal": {
        "smoothed_return": {"h2": [...]},  // RAW DECIMAL fractions (NOT bps)
        "point_return":    {"h2": [...]},
        "mean_return":     {"h2": [...]},
        "dominant_return": {"h2": [...]}
      }
    }
  ]
}
```

## Unit-conversion contract (CRITICAL)

- Rust outputs are **RAW DECIMAL FRACTIONS** (e.g., `0.02` = 2%) per `magnitude.rs:56` doc-comment.
- Python `LabelFactory.{smoothed,point,mean,peak}_return` multiply by `10000.0` to produce **BASIS POINTS**.
- The Python parity test applies `× 10000.0` to Rust outputs before `np.allclose` comparison.

## Index alignment

Given `prices_flat[0..total]` and `forward_prices_2d[i] = prices_flat[i..i+k+max_h+1]`:
- Python row `i` corresponds to entry-time `t = i + k`
- Rust `MultiHorizonLabelGenerator` outputs at `t in [k, total-h-k)` — first `n_samples` entries align
- Rust `MagnitudeGenerator` (smoothing_window=None) outputs at `t in [0, total-h)` — sliced to indices `[k, k+n_samples)`

For `n_samples` Python rows: `total = n_samples + 2k + max_h`.

## Regeneration

When the Rust labeler implementation changes, regenerate via:

```bash
cd feature-extractor-MBO-LOB
cargo test -p hft-labeling --test parity_fixture_gen -- --ignored --nocapture
```

This `#[ignore]`-gated integration test writes the fixture to its canonical location.
The Python parity test (`hft-contracts/tests/test_label_factory_parity.py`) runs in regular CI:

```bash
cd hft-contracts && python -m pytest tests/test_label_factory_parity.py -v
```

## Tolerance

`rtol=1e-12, atol=1e-15` — matches existing convention at
`hft-contracts/tests/test_label_factory.py:118` for hand-calculated tests.

The CLAUDE.md historical "7.56e-12" max-diff observation is ~30 ulps near 1.0 — expected
for sum-of-(k+1) differences in the smoothed-return formula.

## Scenarios

| Name | Description | Status |
|---|---|---|
| `monotone_k2_h2_n5` | Strictly increasing prices [100..110]; deterministic smoke test. Hand-verifiable: row 0 yields `(103-101)/101 = 0.0198…` ✓ | SHIPPED |

## Future scenarios (mechanical extensions per Wave3-B + Wave4 design)

- `random_walk_seed42_k5_h20_n50` — deterministic seeded random walk reflecting `nvda_xnas_128feat_regression_fwd_prices_v3p0` config (k=5, max_h=20, multi-horizon `[1, 5, 10, 20]`). Validates parity across realistic horizon lattice.
- `near_zero_base_k1_h1_n3` — exercises `DIVISION_GUARD_EPS` parity (Python guards via `safe_base = where(|base| > eps, base, 1.0)`; Rust via `safe_ratio` returning `Option<f64>` and skipping degenerate entries — different shapes across languages, may need split fixture or expanded contract).

Each new scenario is mechanical: pick `prices_flat + (k, horizons, n_samples)` satisfying `total = n_samples + 2k + max_h`, then call `build_scenario` in `parity_fixture_gen.rs`.

## Why `#[ignore]`-gated regeneration?

- Python test runs in regular CI; the fixture is checked-in.
- Rust regenerator runs ONLY on operator command. This decouples Rust toolchain availability from Python test environments.
- A SHA-256 hash on the fixture file would be brittle (Rust `serde_json::to_string` and Python `json.dumps` differ in float string representations) — the parity test instead PARSES the JSON and compares numerical arrays via `np.allclose` (Wave4 finding Q4).

## History

- 2026-05-08 — Initial fixture (schema_version=1) shipped per CR6 / NEW-CRITICAL-2 closure. 1 scenario; 4 ReturnType variants × 1 horizon = 4 parity assertions. 11 Python tests in `test_label_factory_parity.py`. All PASS.
