# hft-contracts

Single source of truth for every cross-module data contract in the **intraday trading research pipeline** (origin: HFT microstructure) — feature indices, label contracts, provenance, canonical hashing, FeatureSet registry schema, and atomic I/O.

> **Pipeline scope (2026-06-02).** This module is part of an **intraday trading research pipeline** — an experiment-first platform for discovering and validating *any* profitable **intraday** trading edge (no overnight positions), across approach classes (microstructure/HFT, scalping, intraday momentum, intraday statistical arbitrage, …) and instruments (equities, futures, same-day options). The pipeline *originated* as a high-frequency NVDA MBO/LOB microstructure system — that origin explains the "HFT" / "LOB" / "MBO" naming here — and that microstructure-direction program is now one (largely-closed) track among many. **Names are historical; the mission is general.** This module's role: the contract-plane SSoT — auto-generated cross-module constants from `pipeline_contract.toml` + LabelFactory + ForwardPriceContract + `canonical_hash` + provenance / experiment-record / signal-manifest / feature-set contracts + atomic I/O; the cross-module contract authority, multi-source by design (the off-exchange `OffExchangeFeatureIndex` schema is the precedent for registering a new data source / approach). For the full mission + approach taxonomy + capability-readiness boundary, see root `CLAUDE.md` §Research Scope & Charter (+ `CROSS_ASSET_OFI_FINDINGS_AND_ISSUES_2026_06_01.md` §9).

**Version**: 2.10.0 | **Schema**: 3.0 | **Tests**: run `pytest --collect-only -q` for the live count (hft-rules §11) | **Last Updated**: 2026-07-07 (Phase-2 TRUTH doc-drift fixes: version/schema header, LabelFactory method list, package tree, test-count pointers)

---

## Quick Start

```bash
# Editable install from a local monorepo checkout:
pip install -e /path/to/HFT-pipeline-v2/hft-contracts

# Or directly from GitHub (once the remote is published):
pip install git+https://github.com/nagarx/hft-contracts.git
```

```python
import hft_contracts
print(hft_contracts.__version__)        # "2.10.0"
print(hft_contracts.SCHEMA_VERSION)     # "3.0"

# Feature indices (148 total: stable 0-97 + experimental 98-147)
from hft_contracts import FeatureIndex, ExperimentalFeatureIndex, FEATURE_COUNT
assert FeatureIndex.TRUE_OFI == 84
assert FEATURE_COUNT == 98

# Canonical hashing (SSoT for SHA-256 of JSON-serialized dicts)
from hft_contracts import canonical_json_blob, sha256_hex
h = sha256_hex(canonical_json_blob({"a": 1, "b": 2}))

# Atomic I/O for ledger + registry writes (REV 2 public home)
from hft_contracts import atomic_write_json
atomic_write_json("/tmp/record.json", {"foo": 1})

# Contract-plane dataclasses (Phase 6 + Phase 7 consolidations)
from hft_contracts import (
    LabelFactory, ForwardPriceContract,          # Phase 4
    Provenance, GitInfo,                          # Phase 6 6B.4
    SignalManifest, CONTENT_HASH_RE,              # Phase 6 6B.5
    ExperimentRecord, RecordType,                 # Phase 6 6B.1a
    FeatureSet, FeatureSetRef,                    # Phase 6 6B.3
    GateReportDict, GATE_STATUS_VALUES,           # Phase 7 7.4 R6
)
```

---

## What this package is

`hft-contracts` is the **contract plane** for the pipeline. Every
cross-module invariant — the shape of a feature vector, the encoding of
a label, the schema of an experiment record, the format of a content
hash, the atomic-write discipline of a ledger — is defined exactly once,
here. Five consumer repos depend on it:

- `hft-ops` — experiment orchestrator
- `lob-model-trainer` — PyTorch training + signal export
- `lob-backtester` — vectorized backtester
- `hft-feature-evaluator` — 5-path feature evaluation framework
- `lob-dataset-analyzer` — statistical analyzers

Changing a contract here is a cross-repo event. The `[[changelog]]`
stream inside `contracts/pipeline_contract.toml` (at the monorepo root)
is the audit trail; see the top-level `PIPELINE_ARCHITECTURE.md` §17.3
for the Producer→Consumer matrix.

---

## Module map

All paths are relative to `src/hft_contracts/`. This is an entry-point index — `CODEBASE.md` is the authoritative deep map. Per-module line counts are intentionally omitted (they drift every commit); run `wc -l src/hft_contracts/**/*.py` for a live count.

| Module | Purpose |
|---|---|
| `__init__.py` | Package-level re-exports; `__version__`; ergonomic `from hft_contracts import X` surface |
| `_generated.py` | **Auto-generated** from `contracts/pipeline_contract.toml`; never hand-edit. Enums (`FeatureIndex`, `ExperimentalFeatureIndex`, `SignalIndex`, `OffExchangeFeatureIndex`), counts, slices, name dictionaries, schema_version. |
| `canonical_hash.py` | SHA-256 SSoT — `canonical_json_blob`, `sanitize_for_hash`, `sha256_hex`. Five-site duplication eliminator from Phase 4 Batch 4c hardening. |
| `validation.py` | `ContractError` (the ONE class, post REV 2 consolidation) + boundary validators: `validate_export_contract`, `validate_schema_version`, `validate_metadata_completeness`, `validate_export_dir` (directory-level integrity SSoT), etc. |
| `labels.py` | `LabelContract`, `LabelingStrategy`, TLOB / TripleBarrier / Opportunity / Regression contracts, label encoding tables. |
| `label_factory.py` | `LabelFactory` — Python reference for `smoothed_return / point_return / peak_return / mean_return / forward_realized_variance / multi_horizon / classify`. There is NO Python `dominant_return` — that is a Rust-only generator whose parity test maps it to Python `peak_return` (`tests/test_label_factory_parity.py`). `forward_realized_variance` (v2.9.0) is the Python-only second-moment label (bps², no Rust generator yet; equality-locked to `hft_metrics.realized_measures.realized_variance`). `ForwardPriceContract` dataclass. Bit-for-bit parity with Rust `hft-labeling` for the return-type family (locked by golden-value tests). |
| `provenance.py` | `Provenance`, `GitInfo`, `build_provenance`, `capture_git_info`, `hash_file`, `hash_directory_manifest`, `hash_config_dict`. Moved from `hft-ops/provenance/lineage.py` in Phase 6 6B.4. |
| `signal_manifest.py` | `SignalManifest` + `CONTENT_HASH_RE` (public post REV 2). `ContractError` now re-exported from `validation.py` — single class identity. Moved from `lob-backtester/data/signal_manifest.py` in Phase 6 6B.5. |
| `experiment_record.py` | `ExperimentRecord`, `RecordType` enum, `index_entry()` projection. Atomic `.save()` via `atomic_io.atomic_write_json`. Moved from `hft-ops/ledger/experiment_record.py` in Phase 6 6B.1a. |
| `experiment_recorder.py` | Phase 8D ExperimentRecord-**composition** SSoT — `record_from_artifacts` + `harvest_trust_columns{,_from_signal_metadata}` + `HarvestedTrustColumns`. The ONE record-assembly site shared by the hft-ops orchestrator AND the trainer direct-trainer path. |
| `compatibility.py` | `CompatibilityContract` (shape-determining keys) + `fingerprint()` via `canonical_hash` SSoT + `diff()` + `compute_label_strategy_hash`. Phase II signal-boundary version-skew detection. `COMPATIBILITY_CONTRACT_SCHEMA_VERSION`. |
| `atomic_io.py` | Crash-safe write **family** SSoT — `atomic_write_json` / `atomic_write_binary` / `atomic_write_torch` (torch lazy-imported) / `atomic_write_npy` / `atomic_write_pickle` / `atomic_copy` + `AtomicWriteError`. tmp + fsync + `os.replace` + BaseException-safe cleanup. Renamed from `_atomic_io` in REV 2. |
| `_atomic_io.py` | **Deprecation shim** for pre-REV-2 importers. Emits `DeprecationWarning` on access. Removed 2026-10-31. |
| `gate_report.py` | `GateReportDict` TypedDict + `GATE_STATUS_VALUES`. Documents the `StageResult.captured_metrics["gate_report"]` convention. Phase 7 Stage 7.4 Round 6. |
| `timestamp_utils.py` | `parse_iso8601_utc` / `is_after_cutoff` — ISO-8601 UTC-aware parse + cutoff-comparison SSoT. Replaces fragile lexicographic ISO string compares (silently wrong on non-UTC offsets). |
| `feature_importance_artifact.py` | Post-stage artifact contract — `FeatureImportanceArtifact` (per-feature permutation importance). See `CODEBASE.md` "Post-stage artifact contracts". |
| `test_metrics_ci_artifact.py` | Post-stage artifact contract — `TestMetricsCIArtifact` (bootstrap-CI test-split metrics). ⚠️ a `src/` module, NOT a pytest test file. |
| `pairwise_compare_artifact.py` | Post-stage artifact contract — `PairwiseCompareArtifact` (K-way paired-bootstrap + BH-FDR comparison). |
| `_validators.py` | **Internal** (underscore — do not import across modules) shared field-validator primitives consumed by the artifact/dataclass `__post_init__` methods. |
| `_testing.py` | **Internal** test-support — monorepo-root discovery + phase0 fixture-dir helper for sibling-repo integration tests. |
| `feature_sets/__init__.py` | Package-level re-exports for `FeatureSet`, `FeatureSetRef`, `FeatureSetAppliesTo`, `FeatureSetProducedBy`, etc. |
| `feature_sets/schema.py` | `FeatureSet` frozen dataclass; `FeatureSetRef` / `FeatureSetAppliesTo` / `FeatureSetProducedBy`; `FeatureSetValidationError` / `FeatureSetIntegrityError`. Moved from `hft-ops/feature_sets/schema.py` in Phase 6 6B.3. |
| `feature_sets/hashing.py` | `compute_feature_set_hash` (PRODUCT-only SHA-256 over `{sorted(set(indices)), source_feature_count, contract_version}`). |

The three post-stage artifact modules (`feature_importance_artifact` / `test_metrics_ci_artifact` / `pairwise_compare_artifact`) are one subsystem sharing a uniform frozen-dataclass + SSoT-hash + atomic-save + ledger-routing pattern — see `CODEBASE.md` §"Post-stage artifact contracts". The `feature_sets/` map above lists the 2-of-5 primitives co-moved here (schema + hashing); the writer/registry/producer stay in `hft-ops`.

---

## Install

```bash
# Base install (runtime only):
pip install -e .

# With dev extras (pytest + ruff + coverage):
pip install -e '.[dev]'

# With codegen support (tomli, for regenerating _generated.py from the TOML):
pip install -e '.[dev,generate]'
```

Runtime dependency: `numpy >= 1.26, < 3.0`.

---

## Tests

```bash
pytest --collect-only -q                 # live test count (hft-rules §11 — counts are not hand-maintained in docs)
pytest -q                                # run the suite
```

Most tests are **self-contained** (pure `hft-contracts`; run anywhere).
Two conditional-skip classes exist — a fresh `pip install hft-contracts`
→ `pytest -q` reports passes + skips, never ERROR:

- **hft-ops-conditional** — shim-parity regression guards (locked by
  REV 2 Stage 0) verifying that Phase 6 re-export shims in `hft-ops`
  (for `ExperimentRecord`, `FeatureSet`, `atomic_write_json`) continue
  to return the SAME object as the canonical `hft-contracts` symbol.
  They **SKIP automatically** when `hft-ops` is not installed
  (fresh-clone mode) via `pytest.importorskip`.
- **Monorepo/real-corpus-conditional** — `tests/test_export_contract_real_corpus.py`
  (via the `_testing.py` monorepo-root discovery helpers) requires the
  full HFT-pipeline-v2 monorepo checkout + the `data/exports` volume;
  it skips when either is absent.

Install hft-ops alongside hft-contracts (inside the monorepo checkout)
to run the full suite.

---

## Codegen workflow — MONOREPO ONLY

> ⚠️ **`_generated.py` regeneration requires the full
> HFT-pipeline-v2 monorepo checkout.** The codegen script
> (`contracts/generate_python_contract.py`) and the TOML source-of-truth
> (`contracts/pipeline_contract.toml`) live at the **monorepo root**,
> NOT inside `hft-contracts`.

- **Standalone clones of hft-contracts** (from this GitHub remote):
  `_generated.py` is checked into git as a release artifact — consume
  as-is. You cannot regenerate without the monorepo.

- **Monorepo contributors** editing the TOML:
  ```bash
  # From the monorepo root:
  python contracts/generate_python_contract.py          # regenerate
  python contracts/generate_python_contract.py --check  # CI: exit 1 if stale
  python contracts/verify_rust_constants.py             # cross-check Rust
  ```
  Commit the regenerated `_generated.py` alongside the TOML change.

---

## Package structure

```
hft-contracts/
├── LICENSE                    # LicenseRef-Proprietary
├── README.md                  # this file
├── CHANGELOG.md               # Keep-a-Changelog (2.2.0 = first public release)
├── CODEBASE.md                # deep technical reference
├── pyproject.toml             # hatchling build, PEP 639 license
├── .github/workflows/test.yml # CI: py 3.10/3.11/3.12 matrix
├── src/hft_contracts/
│   ├── __init__.py            # package surface (defines __version__, locked to pyproject.toml by TestPackageVersion)
│   ├── _generated.py          # AUTO-GEN — never hand-edit
│   ├── _atomic_io.py          # DEPRECATION SHIM — removed 2026-10-31
│   ├── _testing.py            # INTERNAL test-support (monorepo-root discovery)
│   ├── _validators.py         # INTERNAL shared field-validator primitives
│   ├── atomic_io.py           # canonical atomic-write family
│   ├── canonical_hash.py      # SHA-256 SSoT
│   ├── compatibility.py       # CompatibilityContract signal-boundary fingerprint
│   ├── experiment_record.py   # ledger record dataclass + atomic save
│   ├── experiment_recorder.py # Phase 8D record-composition SSoT
│   ├── feature_importance_artifact.py  # post-stage artifact contract
│   ├── gate_report.py         # gate-report TypedDict
│   ├── label_factory.py       # pure-function label computation
│   ├── labels.py              # label encoding + contracts
│   ├── pairwise_compare_artifact.py    # post-stage artifact contract
│   ├── provenance.py          # git + config hash tracking
│   ├── signal_manifest.py     # signal-export manifest + CONTENT_HASH_RE
│   ├── test_metrics_ci_artifact.py     # post-stage artifact contract (src module, NOT a pytest file)
│   ├── timestamp_utils.py     # ISO-8601 UTC parse + cutoff-comparison SSoT
│   ├── validation.py          # boundary validators + ContractError
│   ├── py.typed               # PEP 561 marker
│   └── feature_sets/
│       ├── __init__.py
│       ├── schema.py          # FeatureSet frozen dataclass
│       └── hashing.py         # PRODUCT-only content hash
└── tests/                     # one test_*.py file per contract surface — run `ls tests/`
                               # for the file list, `pytest --collect-only -q` for the live count
```

---

## Phase 6 + Phase 7 consolidation history

- **Phase 4 Batch 4c (2026-04-15)** — `canonical_hash.py` extracted as
  SSoT; eliminated 5-site SHA-256-of-JSON duplication across hft-ops
  ledger/provenance, evaluator pipeline, hft-ops feature-sets, and
  trainer inline.
- **Phase 6 6B.1a (2026-04-17)** — `ExperimentRecord` + `RecordType`
  co-moved from `hft-ops/ledger/experiment_record.py` to
  `hft-contracts/experiment_record.py`. hft-ops path becomes a
  deprecation shim (removed 2026-10-31).
- **Phase 6 6B.3 (2026-04-17)** — `FeatureSet`, `FeatureSetRef`, and
  `compute_feature_set_hash` co-moved from `hft-ops/feature_sets/schema.py`
  + `hashing.py` to `hft-contracts/feature_sets/`.
- **Phase 6 6B.4 (2026-04-17)** — `Provenance`, `GitInfo`, and
  `build_provenance` co-moved from `hft-ops/provenance/lineage.py` to
  `hft-contracts/provenance.py`.
- **Phase 6 6B.5 (2026-04-17)** — `SignalManifest` +
  `CONTENT_HASH_RE` (then `_CONTENT_HASH_RE`) co-moved from
  `lob-backtester/data/signal_manifest.py` to
  `hft-contracts/signal_manifest.py`.
- **Phase 7 Stage 7.4 Round 5.5 (2026-04-20)** — `atomic_write_json` +
  `AtomicWriteError` extracted as an I/O SSoT (then `_atomic_io.py`).
  `ExperimentRecord.save()`, `hft-ops/feature_sets/writer.py`, and
  `hft-ops/ledger/ledger.py` now all delegate through one canonical
  crash-safe primitive.
- **Phase 7 Stage 7.4 Round 6 (2026-04-20)** — `GateReportDict` +
  `GATE_STATUS_VALUES` added to document the
  `StageResult.captured_metrics["gate_report"]` convention.
- **REV 2 pre-push (2026-04-20)** — final architectural hygiene:
  `ContractError` consolidated (was defined twice → ONE class);
  `_atomic_io` → `atomic_io` (underscore was a mis-classification);
  `_CONTENT_HASH_RE` → `CONTENT_HASH_RE`; `__version__` added.
  Both old names retained as deprecation shims until 2026-10-31.

---

## License

Proprietary. See [`LICENSE`](LICENSE) for full terms. Consumers of this
package within the HFT pipeline monorepo consume it under the same terms.

---

*See also*: [`CODEBASE.md`](CODEBASE.md) for the deep technical reference
(every module, every validator, every data-flow invariant);
[`CHANGELOG.md`](CHANGELOG.md) for release history.
