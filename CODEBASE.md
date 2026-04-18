# hft-contracts — Codebase Reference

> **Version**: 2.2.0 | **Schema Version**: 2.2 | **Tests**: 264 | **Last Updated**: 2026-04-18 (Phase 6 Post-Audit Hardening closeout)

## Role in the Pipeline

`hft-contracts` is the **single source of truth (SSoT) for cross-module data contracts** in the HFT pipeline. Every Python consumer (lobtrainer, lobbacktest, hft-ops, hft-feature-evaluator, basic-quote-processor) imports from here instead of maintaining independent copies.

Zero runtime state, pure types + validation functions. I/O is lazy — importing any hft-contracts module has **no side effects**; subprocess (git) and filesystem (SignalManifest.validate, hash_file, hash_directory_manifest) occur only when explicit capture/validate/load functions are invoked.

## Package Layout (post-Phase-6, 2026-04-17)

```
hft-contracts/
├── src/hft_contracts/
│   ├── __init__.py            # Public-API re-exports (every symbol below is importable as `from hft_contracts import <X>`)
│   ├── _generated.py          # AUTO-GENERATED from contracts/pipeline_contract.toml — DO NOT HAND-EDIT
│   ├── canonical_hash.py      # SHA-256 SSoT (canonical_json_blob, sanitize_for_hash, sha256_hex)
│   ├── label_factory.py       # Python label-generator (LabelFactory); parallel Rust impl in hft-labeling
│   ├── labels.py              # LabelContract, RegressionLabelContract, 4 pre-built contracts (TLOB, TB, Opportunity, Regression)
│   ├── validation.py          # validate_export_contract + 8 gate-level validators (export shape, metadata, schema version, etc.)
│   ├── provenance.py          # Provenance + GitInfo + build_provenance (Phase 6 6B.4, co-moved from hft-ops)
│   ├── signal_manifest.py     # SignalManifest + ContractError + _CONTENT_HASH_RE (Phase 6 6B.5, co-moved from backtester)
│   ├── experiment_record.py   # ExperimentRecord + RecordType (Phase 6 6B.1a, co-moved from hft-ops — NARROW MOVE; Phase 7 6B.1b will retire `lobtrainer.experiments.ExperimentRegistry`)
│   └── feature_sets/          # Phase 6 6B.3 co-move (2-of-5: schema + hashing only; writer/registry/producer stay in hft-ops)
│       ├── __init__.py        # Public-API re-exports
│       ├── schema.py          # FeatureSet + FeatureSetRef + FeatureSetAppliesTo + FeatureSetProducedBy + validate_feature_set_dict
│       └── hashing.py         # compute_feature_set_hash (PRODUCT-only SHA-256) + _sanitize_for_hash re-export
├── tests/                     # 264 tests
│   ├── test_canonical_hash.py            # 44 tests — canonical-form byte-stability + SSoT invariants
│   ├── test_contract_self_consistency.py # Contract invariants (feature counts sum correctly, etc.)
│   ├── test_experiment_record.py         # 12 tests — Phase 6 6B.1a mirror tests
│   ├── test_feature_sets.py              # 20 tests — Phase 6 6B.3 mirror tests
│   ├── test_label_factory.py             # Multi-horizon label-factory parity vs Rust
│   ├── test_provenance.py                # 21 tests — Phase 6 6B.4 mirror tests
│   ├── test_signal_manifest.py           # 15 tests — Phase 6 6B.5 mirror tests
│   └── test_validation_gates.py          # Export-contract validator tests
├── pyproject.toml             # Declares numpy>=1.26 runtime dep (required by label_factory.py + signal_manifest.py)
└── README.md
```

## Key Types (Schema Surface)

| Type / Constant | Module | Purpose |
|---|---|---|
| `FeatureIndex` | `_generated` | Enum for stable feature indices 0-97 (LOB + derived + MBO) |
| `ExperimentalFeatureIndex` | `_generated` | Enum for experimental indices 98-147 |
| `SignalIndex` | `_generated` | Enum for trading-signal indices 84-91 |
| `OffExchangeFeatureIndex` | `_generated` | Off-exchange feature indices 0-33 (basic-quote-processor) |
| `SCHEMA_VERSION` | `_generated` | Current schema (2.2); emitted at feature index 97 |
| `FEATURE_COUNT` / `STANDARD_FEATURE_COUNT` / ... | `_generated` | Feature-count formulas per configuration |
| `LabelContract` / `RegressionLabelContract` | `labels` | Label-encoding contracts for 4 strategies (TLOB, TripleBarrier, Opportunity, Regression) |
| `TLOB_CONTRACT` / `TB_CONTRACT` / `OPPORTUNITY_CONTRACT` / `REGRESSION_CONTRACT` | `labels` | Pre-built instances |
| `ForwardPriceContract` | `label_factory` | T9 forward-prices schema — smoothing_window_offset + horizons invariant |
| `LabelFactory` | `label_factory` | Python SSoT for label computation (smoothed_return, point_return, peak_return, mean_return, multi_horizon). Rust parity locked by golden-value tests (max diff 7.56e-12 observed). |
| `canonical_json_blob` / `sha256_hex` / `sanitize_for_hash` | `canonical_hash` | Canonical JSON + SHA-256 SSoT (eliminated 5-site duplication in Phase 4 Batch 4c) |
| `Provenance` / `GitInfo` | `provenance` | Experiment lineage (Phase 6 6B.4) |
| `build_provenance` / `capture_git_info` / `hash_file` / `hash_directory_manifest` | `provenance` | Lazy-I/O capture functions |
| `SignalManifest` / `ContractError` / `_CONTENT_HASH_RE` | `signal_manifest` | Trainer/backtester/orchestrator signal-metadata schema (Phase 6 6B.5) |
| `ExperimentRecord` / `RecordType` | `experiment_record` | Ledger record dataclass + 6-variant type enum (Phase 6 6B.1a) |
| `FeatureSet` / `FeatureSetRef` / `FeatureSetAppliesTo` / `FeatureSetProducedBy` | `feature_sets.schema` | Content-addressed feature-selection artifact (Phase 6 6B.3) |
| `compute_feature_set_hash` | `feature_sets.hashing` | PRODUCT-only SHA-256 over (indices, source_feature_count, contract_version) |
| `validate_export_contract` | `validation` | Master validator dispatching to 8 gate-level checks |
| `ContractError` (in validation) | `validation` | Exception for contract violations |

## Import Patterns

**Ergonomic (preferred — post-Phase-6 package-level re-exports)**:
```python
from hft_contracts import (
    FeatureIndex, SCHEMA_VERSION, LabelFactory, TLOB_CONTRACT,
    Provenance, GitInfo, build_provenance,
    SignalManifest, ExperimentRecord, RecordType,
    FeatureSet, FeatureSetRef, compute_feature_set_hash,
    canonical_json_blob, sha256_hex,
)
```

**Explicit submodule**:
```python
from hft_contracts.provenance import Provenance, build_provenance
from hft_contracts.signal_manifest import SignalManifest, ContractError
from hft_contracts.experiment_record import ExperimentRecord, RecordType
from hft_contracts.feature_sets import FeatureSet, FeatureSetRef
from hft_contracts.feature_sets.hashing import compute_feature_set_hash
from hft_contracts.canonical_hash import canonical_json_blob, sha256_hex
```

**Legacy (deprecated, Phase 6 6B.{1a/3/4/5} shims emit DeprecationWarning — migrate before 0.4.0 removal)**:
```python
from hft_ops.provenance.lineage import Provenance               # DeprecationWarning
from hft_ops.ledger.experiment_record import ExperimentRecord   # DeprecationWarning
from hft_ops.feature_sets.schema import FeatureSet              # DeprecationWarning
from hft_ops.feature_sets.hashing import compute_feature_set_hash  # DeprecationWarning
from lobbacktest.data.signal_manifest import SignalManifest     # DeprecationWarning
```

## Design Invariants

1. **No side effects at import** — importing any hft-contracts module performs only dataclass + function definition; no subprocess, no filesystem I/O, no network.
2. **Lazy I/O** — subprocess (git capture) and filesystem (hash_file, hash_directory_manifest, SignalManifest.validate, np.load) occur ONLY when explicit capture/validate/load functions are invoked.
3. **Single source of truth** — every contract-plane primitive that migrates here is removed from its previous home and replaced with a re-export shim; no duplication sanctioned.
4. **Non-breaking contract evolution** — additive changes preferred; breaking changes require `[[changelog]]` entry + `SCHEMA_VERSION` bump + coordinated consumer updates.
5. **Byte-portable canonical form** — canonical JSON uses `sort_keys=True, default=str` and lowercase-hex-64 SHA-256 output; any consumer needing cross-language parity must mirror Python's whitespace convention.

## Recent Phase History

- **Phase 4 Batch 4c hardening (2026-04-15)**: `canonical_hash.py` extracted as SSoT. Eliminated 5-site duplication (`hft_ops.ledger.dedup`, `hft_ops.provenance.lineage`, `hft_ops.feature_sets.hashing`, `hft_evaluator.pipeline`, trainer inline).
- **Phase 6 Post-Audit Hardening (2026-04-17)**: 5 primitives co-moved to hft-contracts (6B.{1a/2/3/4/5}); numpy declared as explicit runtime dep.
- **Phase 6 post-validation (2026-04-18)**: `hash_directory_manifest` now delegates to `canonical_json_blob` SSoT; `ExperimentRecord.from_dict` is now non-mutating; `_sanitize_for_hash` added to `__all__` for shim back-compat; 20 Phase-6 primitives re-exported at package level.

## Cross-References

- Authoritative contract: `contracts/pipeline_contract.toml` → regen `_generated.py` via `python contracts/generate_python_contract.py`
- Rust constant parity: `feature-extractor-MBO-LOB/crates/hft-feature-contract/src/generated.rs` (verified by CI via `contracts/verify_rust_constants.py`)
- Pipeline architecture: `PIPELINE_ARCHITECTURE.md` §17.3 producer→consumer matrix
- Shared coordination surface: root `CLAUDE.md` §"Multi-Agent Coordination — Shared Surface"
