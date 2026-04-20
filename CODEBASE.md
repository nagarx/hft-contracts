# hft-contracts — Codebase Reference

> **Version**: 2.2.0 | **Schema Version**: 2.2 | **Tests**: 300 authoring / 295 + 5 skip fresh-clone | **Last Updated**: 2026-04-20 (REV 2 public-push + follow-up hardening — `atomic_io` rename, `CONTENT_HASH_RE` public, unified `ContractError`, `__version__`, `py.typed` marker, `__getattr__`-based deprecation shims)

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
│   ├── signal_manifest.py     # SignalManifest + CONTENT_HASH_RE (REV 2 public) + re-exported ContractError from validation (REV 2 F1 consolidation — was two independent classes). Module-level `__getattr__` gates `_CONTENT_HASH_RE` legacy access with one-time DeprecationWarning (removal 2026-10-31). (Phase 6 6B.5 co-move + REV 2 public-API hygiene)
│   ├── experiment_record.py   # ExperimentRecord + RecordType (Phase 6 6B.1a, co-moved from hft-ops — NARROW MOVE; Phase 7 6B.1b will retire `lobtrainer.experiments.ExperimentRegistry`)
│   ├── feature_sets/          # Phase 6 6B.3 co-move (2-of-5: schema + hashing only; writer/registry/producer stay in hft-ops)
│   │   ├── __init__.py        # Public-API re-exports
│   │   ├── schema.py          # FeatureSet + FeatureSetRef + FeatureSetAppliesTo + FeatureSetProducedBy + validate_feature_set_dict
│   │   └── hashing.py         # compute_feature_set_hash (PRODUCT-only SHA-256) + _sanitize_for_hash re-export
│   ├── atomic_io.py           # REV 2 public home (renamed from `_atomic_io.py`, 2026-04-20). Canonical atomic_write_json + AtomicWriteError — SSoT unified across ExperimentRecord.save, hft-ops ledger _save_index, and hft-ops feature_sets writer (thin re-export shim). Convention: sort_keys=True + trailing newline + BaseException-safe cleanup.
│   ├── _atomic_io.py          # REV 2 deprecation shim (52 LOC, 2026-04-20). Module-level `__getattr__` forwards `atomic_write_json` / `AtomicWriteError` to `atomic_io.py` canonical module with one-time DeprecationWarning per symbol. Non-public attribute access raises AttributeError. Removal deadline: 2026-10-31.
│   ├── gate_report.py         # Phase 7 Stage 7.4 Round 5 (2026-04-20): GateReportDict TypedDict + GATE_STATUS_VALUES frozenset — documents the cross-stage convention for StageResult.captured_metrics["gate_report"] dicts. Consumed by cli.py::_record_experiment generic harvest + ExperimentRecord.gate_reports + index_entry projection.
│   └── py.typed               # PEP 561 marker (REV 2 follow-up, 2026-04-20). Signals to mypy/pyright/pyre that inline annotations are authoritative. Required since pyproject.toml advertises `Typing :: Typed` classifier.
├── tests/                     # 300 tests authoring env / 295 + 5 skip fresh-clone (REV 2: +11 regression tests — F1 ContractError identity + F2 _atomic_io shim + F8 CONTENT_HASH_RE public/legacy/warning + __version__ presence/format/pyproject-agreement + shim DeprecationWarning telemetry)
│   ├── test_canonical_hash.py            # 44 tests — canonical-form byte-stability + SSoT invariants
│   ├── test_contract_self_consistency.py # Contract invariants (feature counts sum correctly, etc.)
│   ├── test_experiment_record.py         # 37 tests — Phase 6 6B.1a mirror tests + Phase 7 Round 4/5 (gate_reports + atomic save + classification whitelist + non-mutating shim + atomic-io crash-safety)
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
| `SignalManifest` / `ContractError` (re-exported from `validation`) / `CONTENT_HASH_RE` (public; `_CONTENT_HASH_RE` is a DeprecationWarning-gated shim via module-level `__getattr__`, removal 2026-10-31) | `signal_manifest` | Trainer/backtester/orchestrator signal-metadata schema (Phase 6 6B.5 co-move + REV 2 public-API hygiene) |
| `ExperimentRecord` / `RecordType` | `experiment_record` | Ledger record dataclass + 6-variant type enum (Phase 6 6B.1a) |
| `FeatureSet` / `FeatureSetRef` / `FeatureSetAppliesTo` / `FeatureSetProducedBy` | `feature_sets.schema` | Content-addressed feature-selection artifact (Phase 6 6B.3) |
| `compute_feature_set_hash` | `feature_sets.hashing` | PRODUCT-only SHA-256 over (indices, source_feature_count, contract_version) |
| `validate_export_contract` | `validation` | Master validator dispatching to 8 gate-level checks |
| `ContractError` (in validation) | `validation` | Exception for contract violations |
| `atomic_write_json` / `AtomicWriteError` | `atomic_io` (REV 2 public home) | Canonical crash-safe JSON write SSoT (Phase 7 Stage 7.4 Round 5, 2026-04-20, renamed from `_atomic_io` in REV 2). tmp + fsync + os.replace + BaseException cleanup. sort_keys=True + trailing newline defaults for diff-stable output. Used by `ExperimentRecord.save`, `hft_ops.ledger.ledger._save_index`, and `hft_ops.feature_sets.writer.atomic_write_json` (thin re-export). Legacy `_atomic_io` shim retained through 2026-10-31 with DeprecationWarning. |
| `GateReportDict` / `GATE_STATUS_VALUES` | `gate_report` | Cross-stage gate-report convention (Phase 7 Stage 7.4 Round 5, 2026-04-20). TypedDict + frozenset({"pass","warn","fail","abort"}). Emitted by every stage runner under `captured_metrics["gate_report"]`; consumed by `cli.py::_record_experiment` and projected into `ExperimentRecord.index_entry()["gate_reports"]`. TypedDict (not Protocol) deliberately — upgrade to full Protocol when a 3rd gate ships. |

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

**Legacy (deprecated, Phase 6 6B.{1a/3/4/5} shims emit DeprecationWarning — migrate before the 2026-10-31 `_REMOVAL_DATE` calendar deadline)**:
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

## Public API — Package-level surface

In addition to the contracts listed above, `hft_contracts.__init__.py` exposes the following as ergonomic package-level imports (`from hft_contracts import X`):

- **Core**: `FeatureIndex`, `ExperimentalFeatureIndex`, `SignalIndex`, `OffExchangeFeatureIndex`, `SCHEMA_VERSION`, `FEATURE_COUNT`, 60+ feature-count/slice/name constants.
- **Label contracts**: `LabelContract`, `RegressionLabelContract`, `TLOB_CONTRACT`, `TB_CONTRACT`, `OPPORTUNITY_CONTRACT`, `REGRESSION_CONTRACT`, `get_contract`, `LabelingStrategy`, encoding constants.
- **Label computation**: `LabelFactory`, `ForwardPriceContract`, `DIVISION_GUARD_EPS`.
- **Validation**: `ContractError`, `validate_export_contract`, 8 gate-level validators.
- **Canonical hashing**: `canonical_json_blob`, `sanitize_for_hash`, `sha256_hex`.
- **Phase 6 co-moves**: `Provenance`, `GitInfo`, `build_provenance`, `capture_git_info`, `hash_file`, `hash_directory_manifest`, `hash_config_dict`, `NOT_GIT_TRACKED_SENTINEL`, `PROVENANCE_SCHEMA_VERSION`, `SignalManifest`, `ExperimentRecord`, `RecordType`, `FeatureSet`, `FeatureSetRef`, `FeatureSetAppliesTo`, `FeatureSetProducedBy`, `FeatureSetValidationError`, `FeatureSetIntegrityError`, `FEATURE_SET_SCHEMA_VERSION`, `compute_feature_set_hash`, `validate_feature_set_dict`.
- **Phase 7 Stage 7.4 Round 5**: `GATE_STATUS_VALUES`, `GateReportDict`.
- **REV 2 public-API hygiene (2026-04-20)**: `atomic_write_json`, `AtomicWriteError` (renamed home: `hft_contracts.atomic_io`; legacy `hft_contracts._atomic_io` is a DeprecationWarning shim through 2026-10-31); `CONTENT_HASH_RE` (renamed from `_CONTENT_HASH_RE`; legacy name still resolves via module-level `__getattr__` with DeprecationWarning, removal 2026-10-31); `__version__ = "2.2.0"`.

## Recent Phase History

- **Phase 4 Batch 4c hardening (2026-04-15)**: `canonical_hash.py` extracted as SSoT. Eliminated 5-site duplication (`hft_ops.ledger.dedup`, `hft_ops.provenance.lineage`, `hft_ops.feature_sets.hashing`, `hft_evaluator.pipeline`, trainer inline).
- **Phase 6 Post-Audit Hardening (2026-04-17)**: 5 primitives co-moved to hft-contracts (6B.{1a/2/3/4/5}); numpy declared as explicit runtime dep.
- **Phase 6 post-validation (2026-04-18)**: `hash_directory_manifest` now delegates to `canonical_json_blob` SSoT; `ExperimentRecord.from_dict` is now non-mutating; `_sanitize_for_hash` added to `__all__` for shim back-compat; 20 Phase-6 primitives re-exported at package level.

- **REV 2 public-push + follow-up hardening (2026-04-20)** — first public release on https://github.com/nagarx/hft-contracts.git:
  - **F1 — `ContractError` consolidated**: two previously-independent classes (`validation.ContractError` + `signal_manifest.ContractError`) unified. `signal_manifest.py` now imports + re-exports `validation.ContractError`; `__all__` still lists `"ContractError"` for pre-REV-2 back-compat. Regression test (`test_contract_error_is_single_class_across_modules`) locks identity invariant via `is` comparison.
  - **F2 — `_atomic_io` → `atomic_io` rename**: underscore-prefix was a mis-classification (module is cross-module-consumed by hft-ops). Canonical public home is `hft_contracts.atomic_io`. `_atomic_io.py` remains as a 52-LOC deprecation shim with `__getattr__` forwarding to canonical + one-time `DeprecationWarning` per symbol; non-public attribute access raises `AttributeError`. Removal deadline: 2026-10-31.
  - **F8 — `_CONTENT_HASH_RE` → `CONTENT_HASH_RE` rename**: same underscore-prefix mis-classification. Public compiled regex at module scope; legacy name gated through module-level `__getattr__` (PEP 562) with one-time DeprecationWarning citing migration path + removal deadline. Uniform lifecycle with `_atomic_io` shim.
  - **`__version__` attribute**: `hft_contracts.__version__ = "2.2.0"` wired in `__init__.py`. Regression tests (`TestPackageVersion`) lock presence + SemVer format + pyproject.toml agreement.
  - **`py.typed` marker** (REV 2 follow-up): PEP 561 marker file at `src/hft_contracts/py.typed` + explicit `include` in `[tool.hatch.build.targets.wheel]`. Enables downstream mypy / pyright / pyre to use the package's inline type annotations.
  - **Build system**: `requires = ["hatchling>=1.26"]` (REV 2 follow-up bump from 1.22). PEP 639 `license = "LicenseRef-Proprietary"` + `license-files = ["LICENSE"]` require hatchling ≥1.26 in constrained environments.
  - **Publishing-hygiene additions**: `LICENSE` (LicenseRef-Proprietary), `CHANGELOG.md` (Keep-a-Changelog v1.1.0), `.github/workflows/test.yml` (py 3.10/3.11/3.12 matrix; ruff non-blocking via `continue-on-error: true`).
  - **Test decoupling**: 5 shim-parity tests (`TestHftOpsShimCompat` classes + `test_hft_ops_reexport_is_identical`) now `pytest.importorskip("hft_ops")` — fresh-clone installs run 295 + 5 skip instead of 5 ERRORs.
  - **hft-ops co-update** (commit `4696e4e`): `feature_sets/writer.py`, `ledger/ledger.py`, `stages/signal_export.py` migrated to new canonical import paths.
  - **Test counts**: 289 → **300** authoring env (+11 REV 2 regression tests) / **295 + 5 skip** fresh-clone env.

- **Phase 7 Stage 7.4 Round 4 (2026-04-20)** — `ExperimentRecord` additions:
  - **`gate_reports: Dict[str, Dict[str, Any]]`** — generic cross-stage gate-report surface, keyed by runner stage name (`"validation"`, `"post_training_gate"`, future `"post_backtest_gate"`, ...). Replaces the Round 1 pattern of nesting post-training gate output under `training_metrics["post_training_gate"]` (which silently failed the flat-scalar-dict convention of `training_metrics` and was filtered out of `index_entry()`). Fingerprint-stable: `gate_reports` content explicitly NOT hashed by `hft_ops.ledger.dedup.compute_fingerprint` — gate outcomes are observations, not treatments. Default `dict()` so older records load without migration; records written 2026-04-19 between Round 1 and Round 4 are lifted via `from_dict` migration shim (removal deadline 2026-08-01).
  - **`_atomic_write_json` module helper** — `ExperimentRecord.save()` now crash-safe (tmp + fsync + `os.replace`). Prior non-atomic write was vulnerable to silent data loss because the ledger's `_rebuild_index` skips records that fail JSON parse — a half-written record would be visible on disk but invisible to every query.
  - **`index_entry()` whitelist expansion**: Round 1 added 7 regression `test_*` keys (`test_ic`, `test_directional_accuracy`, `test_r2`, `test_mae`, `test_rmse`, `test_pearson`, `test_profitable_accuracy`) for `PostTrainingGateRunner` prior-best queries. Round 4 added 8 `best_val_*` variants + 1 classification extra (`best_val_ic`, `best_val_directional_accuracy`, `best_val_r2`, `best_val_pearson`, `best_val_profitable_accuracy`, `best_val_loss`, `best_val_mae`, `best_val_rmse`, `best_val_signal_rate`) + surfaces `gate_reports[stage].status` for `ledger list --gate-status` filtering. **Phase 8B (2026-04-20) RESOLVES the silent-omission class**: exported `INDEX_SCHEMA_VERSION: str = "1.0.0"` (canonical at `hft_contracts.experiment_record`, re-exported at package level); `hft-ops/ledger/index.json` is now envelope-formatted `{"schema": {...}, "entries": [...]}` with the version embedded, and `ExperimentLedger._load_index` compares on-disk MAJOR.MINOR against the code-side constant — mismatch triggers automatic rebuild (WARN-logged) in dev mode OR `StaleLedgerIndexError` (fail-fast) under `--strict-index` / `CI=true`. Extending the whitelist now requires bumping `INDEX_SCHEMA_VERSION` MINOR (enforced mechanically by `TestIndexEntryCompleteness` + `test_index_entry_top_level_key_set_frozen` golden test). See root `CLAUDE.md` Change-Coordination Checklist row for the workflow.

## Cross-References

- Authoritative contract: `contracts/pipeline_contract.toml` → regen `_generated.py` via `python contracts/generate_python_contract.py`
- Rust constant parity: `feature-extractor-MBO-LOB/crates/hft-feature-contract/src/generated.rs` (verified by CI via `contracts/verify_rust_constants.py`)
- Pipeline architecture: `PIPELINE_ARCHITECTURE.md` §17.3 producer→consumer matrix
- Shared coordination surface: root `CLAUDE.md` §"Multi-Agent Coordination — Shared Surface"
