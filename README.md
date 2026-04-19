# hft-contracts

Single source of truth for every cross-module data contract in the HFT
pipeline — feature indices, label contracts, provenance, canonical
hashing, FeatureSet registry schema, and atomic I/O.

**Version**: 2.2.0 | **Schema**: 2.2 | **Tests**: 298 (293 + 5 hft-ops-conditional) | **Last Updated**: 2026-04-20

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
print(hft_contracts.__version__)        # "2.2.0"
print(hft_contracts.SCHEMA_VERSION)     # "2.2"

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

`hft-contracts` is the **contract plane** for the HFT pipeline. Every
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

All paths are relative to `src/hft_contracts/`. LOCs reflect 2026-04-20 state.

| Module | LOC | Purpose |
|---|---:|---|
| `__init__.py` | 393 | Package-level re-exports; `__version__`; ergonomic `from hft_contracts import X` surface |
| `_generated.py` | 614 | **Auto-generated** from `contracts/pipeline_contract.toml`; never hand-edit. Enums (`FeatureIndex`, `ExperimentalFeatureIndex`, `SignalIndex`, `OffExchangeFeatureIndex`), counts, slices, name dictionaries, schema_version. |
| `canonical_hash.py` | 158 | SHA-256 SSoT — `canonical_json_blob`, `sanitize_for_hash`, `sha256_hex`. Five-site duplication eliminator from Phase 4 Batch 4c hardening. |
| `validation.py` | 441 | `ContractError` (the ONE class, post REV 2 consolidation) + boundary validators: `validate_export_contract`, `validate_schema_version`, `validate_metadata_completeness`, etc. |
| `labels.py` | 237 | `LabelContract`, `LabelingStrategy`, TLOB / TripleBarrier / Opportunity / Regression contracts, label encoding tables. |
| `label_factory.py` | 452 | `LabelFactory` — Python reference for `smoothed_return / point_return / peak_return / mean_return / dominant_return / multi_horizon / classify`. `ForwardPriceContract` dataclass. Bit-for-bit parity with Rust `hft-labeling/src/multi_horizon.rs` + `magnitude.rs` (max diff 7.56e-12). |
| `provenance.py` | 358 | `Provenance`, `GitInfo`, `build_provenance`, `capture_git_info`, `hash_file`, `hash_directory_manifest`, `hash_config_dict`. Moved from `hft-ops/provenance/lineage.py` in Phase 6 6B.4. |
| `signal_manifest.py` | 383 | `SignalManifest` + `CONTENT_HASH_RE` (public post REV 2). `ContractError` now re-exported from `validation.py` — single class identity. Moved from `lob-backtester/data/signal_manifest.py` in Phase 6 6B.5. |
| `experiment_record.py` | 391 | `ExperimentRecord` (20+ fields), `RecordType` enum, `index_entry()` projection. Atomic `.save()` via `atomic_io.atomic_write_json`. Moved from `hft-ops/ledger/experiment_record.py` in Phase 6 6B.1a. |
| `atomic_io.py` | 166 | `atomic_write_json` + `AtomicWriteError`. POSIX atomic-write discipline (tmp + fsync + `os.replace`). Renamed from `_atomic_io` in REV 2 (see CHANGELOG). |
| `_atomic_io.py` | 52 | **Deprecation shim** for pre-REV-2 importers. Emits `DeprecationWarning` on access. Removed 2026-10-31. |
| `gate_report.py` | 82 | `GateReportDict` TypedDict + `GATE_STATUS_VALUES`. Documents the ``StageResult.captured_metrics["gate_report"]`` convention. Phase 7 Stage 7.4 Round 6. |
| `feature_sets/__init__.py` | 59 | Package-level re-exports for `FeatureSet`, `FeatureSetRef`, `FeatureSetAppliesTo`, `FeatureSetProducedBy`, etc. |
| `feature_sets/schema.py` | 490 | `FeatureSet` frozen dataclass (15 fields); `FeatureSetRef` / `FeatureSetAppliesTo` / `FeatureSetProducedBy`; `FeatureSetValidationError` / `FeatureSetIntegrityError`. Moved from `hft-ops/feature_sets/schema.py` in Phase 6 6B.3. |
| `feature_sets/hashing.py` | 137 | `compute_feature_set_hash` (PRODUCT-only SHA-256 over `{sorted(set(indices)), source_feature_count, contract_version}`). |

**Totals**: 15 modules, **4,413 LOC src** (including 614 auto-gen + 52 shim; 3,747 handwritten). 8 test files, **2,990 LOC tests**.

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
pytest -q                                # 298 tests in authoring env
pytest -q --ignore=tests/<slow>.py       # subsets
```

**Test-count breakdown** (2026-04-20):

- **293 self-contained** — pure `hft-contracts` tests; run anywhere.
- **5 hft-ops-conditional** — shim-parity regression guards (locked by
  REV 2 Stage 0) verifying that Phase 6 re-export shims in `hft-ops`
  (for `ExperimentRecord`, `FeatureSet`, `atomic_write_json`) continue
  to return the SAME object as the canonical `hft-contracts` symbol.
  These tests **SKIP automatically** when `hft-ops` is not installed
  (fresh-clone mode) via `pytest.importorskip` — so
  `pip install hft-contracts` → `pytest -q` reports 293 pass + 5 skip,
  never ERROR.

Install hft-ops alongside hft-contracts to run all 298.

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
│   ├── __init__.py            # package surface
│   ├── __version__            # attribute, == "2.2.0"
│   ├── _generated.py          # AUTO-GEN — never hand-edit
│   ├── _atomic_io.py          # DEPRECATION SHIM — removed 2026-10-31
│   ├── atomic_io.py           # canonical atomic-write primitive
│   ├── canonical_hash.py      # SHA-256 SSoT
│   ├── validation.py          # boundary validators + ContractError
│   ├── labels.py              # label encoding + contracts
│   ├── label_factory.py       # pure-function label computation
│   ├── provenance.py          # git + config hash tracking
│   ├── signal_manifest.py     # signal-export manifest + CONTENT_HASH_RE
│   ├── experiment_record.py   # ledger record dataclass + atomic save
│   ├── gate_report.py         # gate-report TypedDict
│   └── feature_sets/
│       ├── __init__.py
│       ├── schema.py          # FeatureSet frozen dataclass
│       └── hashing.py         # PRODUCT-only content hash
└── tests/
    ├── test_canonical_hash.py          (canonical_json_blob + sha256_hex)
    ├── test_contract_self_consistency.py  (enum counts, slices, __version__)
    ├── test_experiment_record.py        (ExperimentRecord + atomic save + shim-parity)
    ├── test_feature_sets.py             (FeatureSet hashing + shim-parity)
    ├── test_label_factory.py            (LabelFactory + bit-exact Rust parity)
    ├── test_provenance.py               (build_provenance + hash_file)
    ├── test_signal_manifest.py          (manifest + ContractError + CONTENT_HASH_RE)
    └── test_validation_gates.py         (validate_export_contract matrix)
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
