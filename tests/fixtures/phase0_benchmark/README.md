# Phase 0 Benchmark Fixtures

**Purpose**: regression baseline for the Architectural Hardening Cycle (plan v2.0 at
`/Users/knight/.claude/plans/fuzzy-discovering-flask.md`). Every Phase I/II/III/IV refactor
regresses forward-pass outputs, signal-boundary fingerprints, and metadata hashes against
these committed golden values. Unintended numerical drift fails CI.

## What's in here

| File | Purpose |
|---|---|
| `generate.py` | Deterministic fixture generator (seed=42). |
| `synthetic_mbo.npz` | MBO-pipeline synthetic: [N=10, T=20, F=98] sequences + [N=10, H=3] regression labels + forward_prices. |
| `synthetic_basic.npz` | BASIC-pipeline synthetic: [N=10, T=20, F=34] sequences + [N=10, H=8] point-return labels + forward_prices. |
| `fixture_metadata_mbo.json` | Metadata matching the real MBO export contract (schema_version, horizons, label_strategy, provenance, etc.). |
| `fixture_metadata_basic.json` | Metadata matching the BASIC export contract. |
| `golden_values.json` | Pinned array hashes + metadata hashes + per-phase forward-pass values (populated progressively). |
| `README.md` | This file. |

## Regeneration

```sh
# Full regenerate (overwrites local fixtures):
python hft-contracts/tests/fixtures/phase0_benchmark/generate.py

# Verify no drift vs committed golden_values.json (use in CI):
python hft-contracts/tests/fixtures/phase0_benchmark/generate.py --verify
```

The generator is bit-deterministic on the reference environment (numpy≥1.26, CPython 3.10+).
Running `generate.py` twice must produce identical file bytes. The `--verify` flag refuses
to run if a regeneration does not match the committed golden values, pointing at either an
environment-non-determinism issue or an intended generator change (commit the new fixtures).

## Acceptable-drift policy

Golden values are PINNED. Any change must be explicit:

1. **Intentional phase change**: a Phase (I.A/I.B/II/III/IV) deliberately changes a model's
   math (e.g., Phase I.A fixes FRESH-2 → agreement_score for the all-zero fixture changes
   from 1.0 to 0.5). Update `golden_values.json` in the SAME COMMIT as the code change,
   with a CHANGELOG entry in the affected module documenting the delta.

2. **Accidental drift**: pure refactor that was not expected to change values, but did.
   Treat as a bug. Investigate before updating. Fingerprint / forward-pass changes outside
   the documented phase deltas are exactly what this fixture set exists to catch.

3. **Environment drift**: `--verify` fails but neither code nor fixtures changed. This
   means the numerical environment shifted (e.g., numpy version bump, BLAS implementation).
   Pin the environment or, if unavoidable, regenerate + explicitly version the acceptance
   threshold.

### Forward-pass tensor tolerance (V.A.1 REDESIGN, 2026-04-21)

The `forward_pass/*.npz` reference tensors (HMHP classifier + HMHP regressor) are
consumed by `lob-models/tests/integration/test_phase0_forward_pass.py` via
`torch.testing.assert_close(observed, expected, rtol=1e-4, atol=1e-6)`. The tolerance
is calibrated to be PLATFORM-DRIFT-TOLERANT (Mac ARM64 Accelerate BLAS vs Linux x86
OpenBLAS) while still catching any real Phase I.B.2-class math change.

**ULP-count rationale** (documenting the calibration):

For the pinned HMHP config (`hidden_dim=16`, `n_encoder_layers=1`, `T=20`, `N=10`,
horizons `[10, 60, 300]`), the forward pass is a chain of ~20 FMA operations per
output element (1 transformer layer × attention + MLP + softmax). Empirical
float32 ULP drift between BLAS implementations is 1-4 ULP per op = ~5e-7 relative
per op. Non-associative reductions (softmax, LayerNorm) add 3-5 ULP. Realistic
maximum BLAS-induced drift on this chain: **~5e-6 relative**.

`rtol=1e-4` is **20× above** the BLAS ceiling — a safe margin that tolerates
cross-platform execution. It is simultaneously **~1000× below** typical Phase
I.B.2 math changes:
- Pooling swap (last-timestep → mean-pool): logits shift ~10-30% relative on
  affected elements.
- FRESH-2 sign-fix: agreement tensor changes ~100% on affected samples.

Both are >> 1e-4, so real math changes still fail loudly even with the safety
margin. `atol=1e-6` is an absolute floor for near-zero values (cross-platform
ULP-scale).

**Tightening runway**: consider `rtol=1e-5` (still 2× above BLAS ceiling) after
1 week of green CI on the current tolerance. Tighter rtol catches smaller drifts
earlier without false-positives.

**Scalar compute_loss tolerance**: separate from tensor tolerance. Uses
`pytest.approx(rel=1e-6)` which was shown empirically (V.A.1 initial CI, 2026-04-21)
to be cross-platform stable on these scalar reductions. Keep at `rel=1e-6`.

**Storage layout** (V.A.1 REDESIGN 2026-04-21):

```
phase0_benchmark/
├── forward_pass/                # NEW directory
│   ├── hmhp_classifier.npz      # logits, horizon_logits_{10,60,300}, agreement, confidence
│   └── hmhp_regressor.npz       # horizon_predictions_{10,60,300}, agreement
└── golden_values.json           # metadata: tensors_file pointer + tensor_keys + tolerance + compute_loss scalars
```

`golden_values.json::forward_pass.<model>.forward` no longer stores SHA-256
hashes. Instead, it stores a **file pointer** (`tensors_file`), a **key list**
(`tensor_keys`), **shapes** (`tensor_shapes`), **dtypes** (`tensor_dtypes`),
and **tolerance** (`tolerance: {rtol, atol}`). The test loads tensors via
`np.load(fixture_dir / tensors_file)` and delegates comparison to
`torch.testing.assert_close`.

## Golden-values schema

```jsonc
{
  "schema_version": "1",
  "generator": "phase0_benchmark/generate.py",
  "generated_at_utc": "2026-04-20T00:00:00+00:00",
  "seed": 42,
  "mbo": {
    "npz_file": "synthetic_mbo.npz",
    "metadata_file": "fixture_metadata_mbo.json",
    "array_hashes": { "sequences": "<sha256>", "regression_labels": "<sha256>", ... },
    "array_shapes": { "sequences": [10, 20, 98], ... },
    "array_dtypes": { "sequences": "float32", ... },
    "metadata_sha256": "<sha256>"
  },
  "basic": { /* same shape */ },
  "forward_pass": {
    // Populated by Phase I.A tests (FRESH-2 agreement_score fixture, XGBoost best_iteration, etc.)
    // Populated by Phase I.B tests (per-model compute_loss scalar on fixture input)
    // Each entry documents: { "value": X, "pinned_by_phase": "I.A|I.B|...", "last_changed_by_phase": "..." }
  },
  "signal_boundary": {
    // Populated by Phase II (CompatibilityContract fingerprint on reference signal export)
  }
}
```

## Why this fixture set is small

N=10 samples × T=20 timesteps × F=98 features is deliberately tiny:
- Fast regeneration (<1s total)
- Small git footprint (~100KB total)
- Zero training required — Phase 0 asserts on FORWARD-PASS deterministic outputs only,
  not on model convergence or learned behavior
- Any refactor that changes math → golden value changes → caught immediately

For larger-scale empirical validation (e.g., pooling A/B in Phase I.B.0), use real or
scaled-synthetic data produced separately. Phase 0 is the integrity gate, not the empirical
study.

## Why fixtures live in `hft-contracts`

`hft-contracts` is the pipeline's SSoT — every downstream module imports it. Placing Phase 0
fixtures here means:
- One canonical home shared by all consumers (trainer, backtester, evaluator, hft-ops).
- Versioned with the contract: a contract schema bump can co-commit fixture regen.
- Consumer integration tests reference `importlib.resources.files("hft_contracts.tests.fixtures.phase0_benchmark")`
  (or equivalent path) without hardcoded absolute paths.
