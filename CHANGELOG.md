# Changelog

All notable changes to this project are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/) for the Python package; the
cross-module **contract schema version** is tracked independently via
`SCHEMA_VERSION` in `_generated.py` (locked at `2.2` for this release).

---

## [2.2.0] — 2026-04-20

First public release on the GitHub remote
`https://github.com/nagarx/hft-contracts.git`. Prior development happened
in the private monorepo; the 9 pre-2.2.0 commits are preserved in
`git log` on `main`.

### Added

- **Package version** — `hft_contracts.__version__ = "2.2.0"` attribute
  added to `__init__.py` for fresh-clone smoke-test compatibility.
- **Canonical I/O primitive** — `atomic_write_json` + `AtomicWriteError`
  at `hft_contracts.atomic_io` (Phase 7 Stage 7.4 Round 5.5).
  POSIX-atomic (tmp + fsync + `os.replace`), crash-safe on all
  exceptions including `BaseException`. Three previously-divergent
  callers (`ExperimentRecord.save`, `hft-ops/feature_sets/writer.py`,
  `hft-ops/ledger/ledger.py`) now all delegate here.
- **Gate-report contract** — `GateReportDict` TypedDict +
  `GATE_STATUS_VALUES` literal at `hft_contracts.gate_report`
  (Phase 7 Stage 7.4 Round 6).
- **Contract-plane consolidation from hft-ops and lob-backtester**
  (Phase 6, 2026-04-17):
  - `ExperimentRecord` + `RecordType` — co-moved from
    `hft-ops/ledger/experiment_record.py` (6B.1a).
  - `FeatureSet` + `FeatureSetRef` + `compute_feature_set_hash` — co-moved
    from `hft-ops/feature_sets/schema.py` + `hashing.py` (6B.3).
  - `Provenance` + `GitInfo` + `build_provenance` + `hash_file` +
    `hash_directory_manifest` — co-moved from
    `hft-ops/provenance/lineage.py` (6B.4).
  - `SignalManifest` + (then `_CONTENT_HASH_RE`) — co-moved from
    `lob-backtester/src/lobbacktest/data/signal_manifest.py` (6B.5).
- **9 regression tests** locking REV 2 architectural invariants:
  `ContractError` cross-module identity, shim identity for `_atomic_io`
  and `_CONTENT_HASH_RE`, shim DeprecationWarning emission, shim
  non-public-attribute rejection, `__version__` presence + SemVer
  format + pyproject.toml agreement.

### Changed (REV 2 pre-push hygiene, 2026-04-20)

- **`_atomic_io` module → public `atomic_io`**. The underscore-prefix
  was a mis-classification — the module is cross-module-consumed by
  `hft-ops` (`feature_sets/writer.py`, `ledger/ledger.py`), violating
  the monorepo's own "underscore = module-internal, never cross-module"
  rule (root `CLAUDE.md`). `hft_contracts._atomic_io` remains as a
  deprecation shim emitting `DeprecationWarning` on access, scheduled
  for removal 2026-10-31.
- **`_CONTENT_HASH_RE` → public `CONTENT_HASH_RE`**. Same
  cross-module-consumption rationale
  (`hft-ops/stages/signal_export.py`). `_CONTENT_HASH_RE` retained as a
  module-level alias pointing at the same compiled pattern;
  alias removed 2026-10-31 alongside `_atomic_io.py`.
- **`signal_manifest.ContractError` → `validation.ContractError`** —
  previously two independent classes with the same name in different
  modules. Consumers catching `from hft_contracts import ContractError`
  (which exports the validation one) silently missed errors raised by
  `SignalManifest.validate()` (which raised the signal_manifest one).
  Now both names resolve to the same class object; regression test locks
  the identity invariant.
- **Rust path references in `label_factory.py`** — docstring refs
  updated from legacy `feature-extractor-MBO-LOB/src/labeling/*.rs` to
  post-multi-crate-decomposition `crates/hft-labeling/src/*.rs`.

### Contract Schema Version

- `2.2` (148 features: stable `0-97` + experimental `98-147`; unchanged
  from pre-2.2.0). Any breaking change to stable indices or label
  encodings bumps this; additive changes to experimental features do
  not.

### Test counts

- **Authoring env** (hft-ops installed): 298 passing.
- **Fresh-clone env** (hft-ops absent): 293 passing + 5 skipped
  (shim-parity guards skip gracefully via `pytest.importorskip`).

### Historical

Pre-2.2.0 commit history preserved on `main`:

```
6e633c4 feat(hft-contracts): Phase 7 Stage 7.4 Rounds 4+5 — gate_reports field + _atomic_io SSoT + gate_report TypedDict
a2140ed feat(experiment_record): Phase 7 Stage 7.4 — expand index_entry whitelist for regression metrics
f58ffed docs(hft-contracts): Phase 6 final hygiene — CODEBASE.md creation + canonical_hash docstring fresh
629e754 fix(hft-contracts): Phase 6 post-validation — 3 correctness hardening + package re-exports
229e5ba feat(experiment_record): Phase 6 6B.1a — co-move ExperimentRecord to hft_contracts (narrow)
1c1e133 feat(feature_sets): Phase 6 6B.3 — co-move schema + hashing to hft_contracts
a86f381 feat(signal_manifest): Phase 6 6B.5 — co-move SignalManifest to hft_contracts contract plane
9841067 feat(provenance): Phase 6 6B.4 — co-move Provenance to hft_contracts contract plane
eaff69b feat: Initial hft-contracts baseline — contract plane SSoT
```

### Dependencies

- Runtime: `numpy >= 1.26, < 3.0` (upper pin protects against NumPy 3
  breaking changes; matches Phase 6 declaration + REV 2 upper-bound
  hardening).
- Development (`[dev]` extra): `pytest >= 8`, `pytest-cov >= 4`,
  `ruff >= 0.5`.
- Codegen (`[generate]` extra): `tomli >= 2.0`.
- Build: `hatchling >= 1.22` (PEP 639 `license-files` support).

### Consumers (monorepo-internal)

All 5 consumer repos declare `hft-contracts` as a bare-name dependency
in their `pyproject.toml`:

- `hft-ops`
- `lob-model-trainer`
- `lob-backtester`
- `hft-feature-evaluator`
- `lob-dataset-analyzer`

Resolution is via editable local install (`pip install -e
/path/to/hft-contracts`). Git-URL pinning (`hft-contracts @
git+https://github.com/nagarx/hft-contracts.git@v2.2.0`) is a deferred
follow-up for remote-consumer installs.
