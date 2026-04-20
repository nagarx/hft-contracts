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
