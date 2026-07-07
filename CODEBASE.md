# hft-contracts — Codebase Reference

> **Pipeline scope (2026-06-02).** This module is part of an **intraday trading research pipeline** — an experiment-first platform for discovering and validating *any* profitable **intraday** trading edge (no overnight positions), across approach classes (microstructure/HFT, scalping, intraday momentum, intraday statistical arbitrage, …) and instruments (equities, futures, same-day options). The pipeline *originated* as a high-frequency NVDA MBO/LOB microstructure system — that origin explains the "HFT" / "LOB" / "MBO" naming here — and that microstructure-direction program is now one (largely-closed) track among many. **Names are historical; the mission is general.** This module's role: the contract-plane SSoT — auto-generated cross-module constants from `pipeline_contract.toml` + LabelFactory + ForwardPriceContract + `canonical_hash` + provenance / experiment-record / signal-manifest / feature-set contracts + atomic I/O; the cross-module contract authority, multi-source by design (the off-exchange `OffExchangeFeatureIndex` schema is the precedent for registering a new data source / approach). For the full mission + approach taxonomy + capability-readiness boundary, see root `CLAUDE.md` §Research Scope & Charter (+ `CROSS_ASSET_OFI_FINDINGS_AND_ISSUES_2026_06_01.md` §9).

> **Version**: 2.10.0 (RecordType +`DISCOVERY` for discovery-harness verdicts → 8 variants) | **Schema Version**: 3.0 (Phase G G.6.A bump 2.2 → 3.0 MAJOR per CLAUDE.md root rule: any modification to stable features 0-97 = BREAKING) | **Tests**: 754+ passing (run `pytest --collect-only -q` for the live count; post Phase 8D `experiment_recorder` SSoT v2.8.0 commit `d773ac4` 2026-05-14 + Sub-cycle 2 v2.6.0 composer signature + Phase V.A.4 trust-column fingerprint + γ-1 LITE close-out) | **Last Updated**: 2026-06-27 (doc-sync: version → 2.10.0; RecordType.DISCOVERY)
>
> **Phase V.A.4 SHIPPED (2026-04-21, commit `a0fa3d2`)** — New `ExperimentRecord.compatibility_fingerprint: Optional[str]` field (64-hex SHA-256 validated via `CONTENT_HASH_RE`). Surfaces cross-experiment comparability: every record produced against the same CompatibilityContract version has the same fingerprint. Projected into `index_entry()` for `hft-ops ledger list --compatibility-fp <hex>` filter. `INDEX_SCHEMA_VERSION` bumped 1.3.0 → **1.4.0** (MINOR additive — triggers envelope auto-rebuild on existing ledgers per Phase 8B mechanism).
>
> **Phase V.1 L1.2 SHIPPED (2026-04-21, commit `0efabe0`)** — New `ExperimentRecord.signal_export_output_dir: Optional[str]` field (run-time-captured absolute path from `SignalExportRunner.run`). Closes Agent 2 H1 manifest-move-resilience gap — `hft-ops sweep compare` adapter prefers this stored path over manifest re-parse. Record-level only (NOT projected into `index_entry()` — access pattern is `ledger.get(exp_id)` full load). 4 new regression tests: default-None, round-trip, index-exclusion lock, pre-V.1 backward-compat via `from_dict` absent-key filter.
>
> **Previously** (Phase 8C-α Integration Close-Out 2026-04-20) — `FeatureImportanceArtifact.__post_init__` WARN on missing `feature_set_ref` for `method="permutation"` (architect-Q9.1; exploratory runs still allowed); preserves Stage C.2 + 2-round post-audit: v2 schema + `ExperimentRecord.artifacts[]` + `block_length_samples` rename + `from_dict` migration + v1/v2 content-hash divergence lock + `compute_stability` in `__all__` + (0,0)→0.0.

## Role in the Pipeline

`hft-contracts` is the **single source of truth (SSoT) for cross-module data contracts** in the pipeline. Every Python consumer (lobtrainer, lobbacktest, hft-ops, hft-feature-evaluator, basic-quote-processor) imports from here instead of maintaining independent copies.

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
│   ├── validation.py          # validate_export_contract + 8 gate-level validators (export shape, metadata, schema version, etc.) + validate_export_dir (directory-level integrity SSoT — Foundation Integrity C1, 2026-05-30) + validate_day_metadata / validate_off_exchange_export_contract / validate_any_export_contract
│   ├── provenance.py          # Provenance (+ producer_commits: Dict[str,str] record-level build-lineage OBSERVATION — Foundation Integrity P1a, 2026-05-30; excluded from the dedup fingerprint) + GitInfo + build_provenance (Phase 6 6B.4, co-moved from hft-ops)
│   ├── signal_manifest.py     # SignalManifest + CONTENT_HASH_RE (REV 2 public) + re-exported ContractError from validation (REV 2 F1 consolidation — was two independent classes). Module-level `__getattr__` gates `_CONTENT_HASH_RE` legacy access with one-time DeprecationWarning (removal 2026-10-31). (Phase 6 6B.5 co-move + REV 2 public-API hygiene)
│   ├── experiment_record.py   # ExperimentRecord + RecordType (Phase 6 6B.1a, co-moved from hft-ops — NARROW MOVE; Phase 7 6B.1b will retire `lobtrainer.experiments.ExperimentRegistry`)
│   ├── experiment_recorder.py # Phase 8D SSoT — composes an ExperimentRecord from on-disk artifacts + signal metadata (harvest_trust_columns{,_from_signal_metadata} + record_from_artifacts + HarvestedTrustColumns). ONE construction site shared by BOTH the hft-ops orchestrator AND the lob-model-trainer direct-trainer path (keeps the two consumers bit-identical). Reuses ExperimentRecord + build_provenance + compute_experiment_provenance_hash + atomic_write_json SSoTs — no new primitive.
│   ├── feature_sets/          # Phase 6 6B.3 co-move (2-of-5: schema + hashing only; writer/registry/producer stay in hft-ops)
│   │   ├── __init__.py        # Public-API re-exports
│   │   ├── schema.py          # FeatureSet + FeatureSetRef + FeatureSetAppliesTo + FeatureSetProducedBy + validate_feature_set_dict
│   │   └── hashing.py         # compute_feature_set_hash (PRODUCT-only SHA-256) + _sanitize_for_hash re-export
│   ├── atomic_io.py           # REV 2 public home (renamed from `_atomic_io.py`, 2026-04-20). Crash-safe write SSoT — a FAMILY, not just JSON: atomic_write_json / atomic_write_binary / atomic_write_torch (torch LAZY-imported so hft-contracts stays a numpy-only leaf) / atomic_write_npy / atomic_write_pickle / atomic_copy, all sharing tmp+fsync+os.replace + BaseException-safe cleanup + AtomicWriteError (see `__all__`). JSON conventions: sort_keys=True + trailing newline. Unifies ExperimentRecord.save, hft-ops ledger _save_index + feature_sets writer, and (#PY-73) ~20 non-atomic torch/npy/pickle write sites across the trainer/models/backtester.
│   ├── _atomic_io.py          # REV 2 deprecation shim (52 LOC, 2026-04-20). Module-level `__getattr__` forwards `atomic_write_json` / `AtomicWriteError` to `atomic_io.py` canonical module with one-time DeprecationWarning per symbol. Non-public attribute access raises AttributeError. Removal deadline: 2026-10-31.
│   ├── gate_report.py         # Phase 7 Stage 7.4 Round 5 (2026-04-20): GateReportDict TypedDict + GATE_STATUS_VALUES frozenset — documents the cross-stage convention for StageResult.captured_metrics["gate_report"] dicts. Consumed by cli.py::_record_experiment generic harvest + ExperimentRecord.gate_reports + index_entry projection.
│   ├── compatibility.py       # Phase II (2026-04-20): CompatibilityContract frozen dataclass (11 shape-determining keys) + fingerprint() via canonical_hash SSoT + diff() + compute_label_strategy_hash() helper. `__post_init__` defensive validators (P-2, 2026-04-20): reject feature_count<=0, window_size<=0, empty strings on required-string fields, non-positive horizons, primary_horizon_idx out-of-range. Horizons list→tuple coerced via `object.__setattr__` for JSON round-trip fingerprint stability. COMPATIBILITY_CONTRACT_SCHEMA_VERSION="1.0.0".
│   ├── feature_importance_artifact.py # Post-stage artifact contract (Phase 8C-α) — FeatureImportanceArtifact + FeatureImportance + compute_stability. See "## Post-stage artifact contracts" below.
│   ├── test_metrics_ci_artifact.py     # Post-stage artifact contract (Phase 2 P2.A) — TestMetricsCIArtifact + MetricCIBound (bootstrap-CI test-split metrics). ⚠️ src/ module, NOT a pytest test file despite the `test_` prefix.
│   ├── pairwise_compare_artifact.py    # Post-stage artifact contract (Phase 2 P2.C) — PairwiseCompareArtifact + PairwiseResultRecord (K-way paired-bootstrap + BH-FDR comparison). See "## Post-stage artifact contracts" below.
│   ├── timestamp_utils.py     # ISO-8601 UTC-aware parse + cutoff-comparison SSoT (parse_iso8601_utc, is_after_cutoff). Replaces fragile lexicographic ISO string compares (silently wrong on non-UTC offsets crossing a day boundary). Always returns tz-aware UTC; fail-loud ValueError on malformed input.
│   ├── _validators.py         # INTERNAL (underscore — do NOT import across module boundaries). Shared field-validator primitive library consumed by the artifact/dataclass `__post_init__` methods: finite-float / positive-int / sha256-hex / CI-ordering / feature-set-ref, etc. Fail-loud ValueError; rejects bool-as-number. ZERO intra-package imports.
│   ├── _testing.py            # INTERNAL test-support: monorepo-root discovery (require_monorepo_root) + phase0 fixture-dir helper (phase0_fixture_dir). Editable-install only; wheel/sdist raise FileNotFoundError. Consumed by sibling-repo integration tests, not production code.
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

## Post-stage artifact contracts

`feature_importance_artifact.py`, `test_metrics_ci_artifact.py`, and `pairwise_compare_artifact.py` are **one coherent subsystem** — the post-stage artifact-contract family — and share a single, stable pattern. Learn it once and reuse it; do NOT hand-roll an ad-hoc artifact.

Every member is:

- a **frozen dataclass** carrying its own `<KIND>_SCHEMA_VERSION` string (on the artifact, not on the data inside);
- `content_hash()` delegating to the `canonical_hash` SSoT;
- `save()` / `load()` delegating to the `atomic_io` SSoT (crash-safe write);
- `to_dict()` / `from_dict()` where `from_dict` is a **version-migration shim** — MINOR bump = additive fields with `None` defaults (legacy artifacts still load); MAJOR bump = rename/remove (requires an explicit migration path);
- **produced** by the trainer-side libraries (`lobtrainer.training.importance.*` for feature-permutation importance; `lobtrainer.analysis.stat_rigor.{ci,pairwise}` for the CI + pairwise artifacts) and **content-addressed** into the hft-ops ledger (`hft-ops/ledger/<kind>/{yyyy_mm}/<sha256>.json`) via `_POST_STAGE_ARTIFACT_PATTERNS` (in `hft-ops/src/hft_ops/ledger/ledger.py`). The `record.artifacts[].sha256` projection integrates them into the Phase-Y `experiment_provenance_hash` graph.

The three members differ only in payload: `FeatureImportanceArtifact` = per-feature permutation importance; `TestMetricsCIArtifact` = bootstrap-CI test-split metrics (Politis–Romano / Künsch moving-block); `PairwiseCompareArtifact` = K-way paired-bootstrap + BH-FDR comparison.

⚠️ **Naming trap**: `test_metrics_ci_artifact.py` is a `src/` production module, NOT a pytest test file — pytest does not collect it.

**To add a fourth artifact kind**, follow the root `CLAUDE.md` Change-Coordination Checklist row "Add a new post-stage artifact kind (Phase 8C-α Stage C.3 convention)" rather than improvising — it enumerates the schema-in-TOML → producer-contract → ledger-routing → fingerprint-invariant steps.

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
| `Provenance` / `GitInfo` | `provenance` | Experiment lineage (Phase 6 6B.4). `Provenance.producer_commits: Dict[str, str]` (Foundation Integrity P1a, 2026-05-30) — record-level OBSERVATION of producer-code git lineage (`extractor_git_sha` / `reconstructor_git_sha` / `hft_statistics_git_sha` + `reconstructor_source` + `completeness`); populated fail-open at extraction time; EXCLUDED from the dedup fingerprint; default `{}` (back-compat). |
| `build_provenance` / `capture_git_info` / `hash_file` / `hash_directory_manifest` | `provenance` | Lazy-I/O capture functions |
| `SignalManifest` / `ContractError` (re-exported from `validation`) / `CONTENT_HASH_RE` (public; `_CONTENT_HASH_RE` is a DeprecationWarning-gated shim via module-level `__getattr__`, removal 2026-10-31) | `signal_manifest` | Trainer/backtester/orchestrator signal-metadata schema (Phase 6 6B.5 co-move + REV 2 public-API hygiene). Phase II (2026-04-20): `validate(expected_contract=..., expected_fields=..., strict=...)` — 3-way fingerprint check + partial-assertion Dict API. Phase II hardening (2026-04-21): empty `expected_fields` dict raises `ValueError` fail-loud (SB-D). |
| `CompatibilityContract` / `COMPATIBILITY_CONTRACT_SCHEMA_VERSION` / `compute_label_strategy_hash` | `compatibility` | Phase II (2026-04-20) signal-boundary fingerprint contract (11 shape-determining keys: contract_version, schema_version, feature_count, window_size, feature_layout, data_source, label_strategy_hash, calibration_method, primary_horizon_idx, horizons, normalization_strategy). Producer (trainer exporter) emits; consumer (backtester) validates via `SignalManifest.validate(expected_fields={...})` partial assertion or `expected_contract=...` full fingerprint match. `fingerprint()` via `canonical_hash` SSoT. Registered in `contracts/pipeline_contract.toml [artifacts.compatibility_contract_schema]` (v2.22). |
| `ExperimentRecord` / `RecordType` | `experiment_record` | Ledger record dataclass + 8-variant type enum (Phase 6 6B.1a; TRAINING / ANALYSIS / CALIBRATION / BACKTEST / EVALUATION / SWEEP_AGGREGATE / SWEEP_FAILURE [Phase 8A.1] / DISCOVERY [discovery-harness verdicts]) |
| `FeatureSet` / `FeatureSetRef` / `FeatureSetAppliesTo` / `FeatureSetProducedBy` | `feature_sets.schema` | Content-addressed feature-selection artifact (Phase 6 6B.3) |
| `compute_feature_set_hash` | `feature_sets.hashing` | PRODUCT-only SHA-256 over (indices, source_feature_count, contract_version) |
| `validate_export_contract` | `validation` | Master validator dispatching to 8 gate-level checks |
| `validate_export_dir` / `validate_day_metadata` / `validate_off_exchange_export_contract` / `validate_any_export_contract` | `validation` | Directory-level + per-day export-integrity validators (Foundation Integrity C1, 2026-05-30). `validate_export_dir(export_dir, *, strict=True)` composes the per-day SSoT across a directory + adds cross-day `schema_version`/`git_commit` uniformity + manifest↔disk count reconciliation (CF-1: pre-align `total_sequences` NOT compared) + split disjointness; off-exchange dirs fail CLEAR with a single pointer. In `__all__`; consumed by hft-ops `validate_manifest` preflight. |
| `ContractError` (in validation) | `validation` | Exception for contract violations |
| `atomic_write_json` / `AtomicWriteError` | `atomic_io` (REV 2 public home) | Canonical crash-safe JSON write SSoT (Phase 7 Stage 7.4 Round 5, 2026-04-20, renamed from `_atomic_io` in REV 2). tmp + fsync + os.replace + BaseException cleanup. sort_keys=True + trailing newline defaults for diff-stable output. Used by `ExperimentRecord.save`, `hft_ops.ledger.ledger._save_index`, and `hft_ops.feature_sets.writer.atomic_write_json` (thin re-export). Legacy `_atomic_io` shim retained through 2026-10-31 with DeprecationWarning. |
| `GateReportDict` / `GATE_STATUS_VALUES` | `gate_report` | Cross-stage gate-report convention (Phase 7 Stage 7.4 Round 5, 2026-04-20). TypedDict + frozenset({"pass","warn","fail","abort"}). Emitted by every stage runner under `captured_metrics["gate_report"]`; consumed by `cli.py::_record_experiment` and projected into `ExperimentRecord.index_entry()["gate_reports"]`. TypedDict (not Protocol) deliberately — upgrade to full Protocol when a 3rd gate ships. |
| `record_from_artifacts` / `harvest_trust_columns` / `harvest_trust_columns_from_signal_metadata` / `HarvestedTrustColumns` | `experiment_recorder` | Phase 8D ExperimentRecord-composition SSoT — assembles an `ExperimentRecord` from on-disk artifacts + `signal_metadata.json` at the ONE construction site shared by the hft-ops orchestrator AND the trainer direct path (dual-consumer boundary; keeps records bit-identical). Trust-column source is EITHER `signal_metadata_path` (disk) XOR `captured_metrics_for_trust` (in-memory) — mutually exclusive. `require_complete_provenance=True` opts into fail-loud on missing Phase-Y components; default gracefully degrades to `None`. |
| `FeatureImportanceArtifact` / `FeatureImportance` / `compute_stability` / `FEATURE_IMPORTANCE_SCHEMA_VERSION` | `feature_importance_artifact` | Post-stage artifact (see "## Post-stage artifact contracts") — per-feature permutation-importance payload. `__post_init__` WARNs when `method="permutation"` and `feature_set_ref is None` (exploratory runs still emit). |
| `TestMetricsCIArtifact` / `MetricCIBound` / `TEST_METRICS_CI_SCHEMA_VERSION` | `test_metrics_ci_artifact` | Post-stage artifact (see "## Post-stage artifact contracts") — bootstrap-CI test-split metrics (Politis–Romano / Künsch moving-block). ⚠️ a `src/` module, NOT a pytest test file. |
| `PairwiseCompareArtifact` / `PairwiseResultRecord` / `PAIRWISE_COMPARE_SCHEMA_VERSION` | `pairwise_compare_artifact` | Post-stage artifact (see "## Post-stage artifact contracts") — K-way paired-bootstrap + BH-FDR comparison (BH meaningful only at K≥3); per-pair effect sizes + a strict shared `paired_compat_fingerprint`; `from_hft_metrics_result` adapter over `hft_metrics.pairwise`. |
| `parse_iso8601_utc` / `is_after_cutoff` | `timestamp_utils` | ISO-8601 UTC-aware parse + cutoff-comparison SSoT. Always returns a tz-aware UTC `datetime`; fail-loud `ValueError` on malformed input. Route every timestamp comparison through here — a raw lexicographic ISO string compare is silently wrong for non-UTC offsets crossing a day boundary. |

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
- **Validation**: `ContractError`, `validate_export_contract`, 8 gate-level validators, `validate_export_dir` (directory-level integrity SSoT — Foundation Integrity C1, 2026-05-30), `validate_day_metadata`, `validate_off_exchange_export_contract`, `validate_any_export_contract`.
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

- **Phase II — CompatibilityContract + SignalManifest 3-way validation (2026-04-20)** — signal-boundary version-skew detection for validation-report D1/D10/D11 silent-drift class:
  - **New `compatibility.py` module**: `CompatibilityContract` frozen dataclass (11 shape-determining keys) + `fingerprint()` via `canonical_hash` SSoT + `diff()` + `compute_label_strategy_hash()` helper. `COMPATIBILITY_CONTRACT_SCHEMA_VERSION="1.0.0"`. Registered in `contracts/pipeline_contract.toml [artifacts.compatibility_contract_schema]`.
  - **`SignalManifest.validate()`** extended with `expected_contract` (full fingerprint match) + `expected_fields: Optional[Dict[str, Any]]` (partial-assertion API for consumers whose config only covers a subset) + `strict` (reject legacy manifests). 3-way check: tamper-detection (recomputed fingerprint) + version-skew-detection (expected_fingerprint) + calibration-precedence (D10 orphan-file rule). Unknown keys in `expected_fields` raise `ValueError` (typo detection, fail-loud).
  - **`SignalManifest` dataclass fields** extended: `compatibility`, `compatibility_fingerprint`, `calibration_method`, `data_source`. Legacy pre-Phase-II manifests load with `DeprecationWarning` in lenient mode, raise in strict.
  - **`ALIGNED_FILES`** now includes `calibrated_returns.npy` — shape cross-check catches stale-file mismatches.

- **Phase II hardening post-audit (2026-04-21)** — 5-agent adversarial validation closed 2 ship-blockers + 1 revert:
  - **P-2**: `CompatibilityContract.__post_init__` defensive validators (feature_count>0, window_size>0, non-empty required-string fields, empty-or-positive-int horizons, primary_horizon_idx in range, horizons list→tuple coerce for fingerprint stability). `object.__setattr__` is the frozen-dataclass idiom.
  - **SB-D empty-dict reject**: `validate(expected_fields={})` now raises `ValueError` per hft-rules §5 — empty dict is caller-side logic error, not a silent no-op. Closes the degenerate path where a config-parsing mistake silently dropped the assertion.
  - **SB-C retracted**: initial scope-restriction of the orphan-file rule to Phase-II-aware manifests broke the existing `test_orphan_calibrated_file_raises_via_validate` test, which locks the coherent `validate=True=strict / validate=False=legacy` boundary. Reverted; extended ContractError message with the opt-out hint ("pass validate=False to load legacy directories").
  - **P-4**: `[artifacts.compatibility_contract_schema]` block registered in `contracts/pipeline_contract.toml` with 11 required_fields enumerated + fingerprint_algorithm + fingerprint_module citing `hft_contracts.canonical_hash` SSoT. `[[changelog]]` entries v2.21 (Phase II ship) + v2.22 (hardening) + v2.22.1 (post-audit closure).
  - **Test counts**: 300 → **465** (+165 across Phase II cycle: 29 compatibility + 24 signal_manifest_compat + ~15 Phase 0 fixtures + other).

## Cross-References

- Authoritative contract: `contracts/pipeline_contract.toml` → regen `_generated.py` via `python contracts/generate_python_contract.py`
- Rust constant parity: `feature-extractor-MBO-LOB/crates/hft-feature-contract/src/generated.rs` (verified by CI via `contracts/verify_rust_constants.py`)
- Pipeline architecture: `PIPELINE_ARCHITECTURE.md` §17.3 producer→consumer matrix
- Shared coordination surface: root `CLAUDE.md` §"Multi-Agent Coordination — Shared Surface"
