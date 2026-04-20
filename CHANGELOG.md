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

### Added (Phase 8C-α, 2026-04-20)

- **`FeatureImportanceArtifact` contract + `FeatureImportance` per-feature
  record** at `hft_contracts.feature_importance_artifact` (Stage C.2).
  Frozen dataclasses with `content_hash()` (via `canonical_hash` SSoT)
  + `save()` (via `atomic_io` SSoT). Produced by the trainer's
  `PermutationImportanceCallback` (Stage C.1), routed into
  `hft-ops/ledger/feature_importance/{yyyy_mm}/<sha256>.json` by the
  ledger hook (Stage C.3), consumed by the evaluator feedback-merge
  step (Stage C.5, planned).
- **`ExperimentRecord.artifacts: List[Dict]` field** — carries references
  to content-addressed post-training artifacts. Schema per entry:
  `{kind, path, sha256, bytes, method?}`. Not part of `compute_fingerprint`
  (post-training artifacts are observations, not treatments).
- **`INDEX_SCHEMA_VERSION` → 1.3.0** (MINOR additive): adds
  `artifact_kinds: sorted(List[str])` projection to `index_entry()`
  for fast `ledger list --has-artifact feature_importance` filtering.
- **`compute_stability(mean, std)`** helper in the same module —
  CV-variant stability metric clipped to [0, 1].
- **`FEATURE_IMPORTANCE_SCHEMA_VERSION = "2"`** after post-audit rename
  (see below). `from_dict` transparently migrates legacy v1 artifacts
  that used `block_size_days` into the v2 `block_length_samples` field.

### Changed (Phase 8C-α Integration Close-Out, 2026-04-20 round-2 follow-up)

- **`FeatureImportanceArtifact.__post_init__` feature_set_ref WARN
  policy** (architect-Q9.1) — when `method="permutation"` AND
  `feature_set_ref is None`, emits a `logging.warning` at construction
  time. Exploratory workflows (ad-hoc `feature_indices` without a
  registered FeatureSet) remain first-class: artifact still emitted +
  ledger-routed, but operators know Stage C.5 feedback-merge cannot
  consume it (no feature-set to reconcile against evaluator profiles).
  Warn-and-allow per hft-rules §8 (never silently drop data; always
  record diagnostics). +2 regression tests
  (`test_missing_feature_set_ref_for_permutation_warns`,
  `test_present_feature_set_ref_does_not_warn`). Test count 333 → 335.

### Changed (Phase 8C-α post-audit hardening, 2026-04-20)

- **`block_size_days` → `block_length_samples`** on
  `FeatureImportanceArtifact` (schema v1 → v2) and on the
  contract-plane `[artifacts.feature_importance_schema]` block. The
  old name silently implied day-semantics that the code never delivered
  (Politis-Romano 1994 autocorrelation preservation requires
  block_length > autocorrelation lag; default=1 is element-wise
  permutation). No v1 artifacts exist in the wild (C.2 shipped hours
  before the rename) so the migration is zero-cost; `from_dict` accepts
  both keys for future-proofing.
- **`from_dict` forward-compat filter** — per-feature dict entries are
  now filtered to known `FeatureImportance` fields, so a v3+ additive
  field does not crash a v2 consumer reading a newer artifact.
- **`index_entry()::artifact_kinds` rejects non-string kinds** —
  previously coerced via `str(...)` leaking `"123"` / `"None"` into
  the ledger index. Now explicitly `isinstance(kind, str)` gated.
- **`compute_stability(0, 0) → 0.0`** (consistent with near-zero-mean
  clamp) + exported in `__all__`.
- **6 new regression tests** locking the post-audit fixes in
  `TestPostAuditFixes`: forward-compat unknown kwargs, non-string
  artifact kind rejection, public-API membership, degenerate stability,
  v1→v2 migration, missing-key fail-loud.

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
- **11 regression tests** locking REV 2 architectural invariants:
  `ContractError` cross-module identity, shim identity for `_atomic_io`
  and `_CONTENT_HASH_RE`, shim DeprecationWarning emission (both
  `_atomic_io` module-level and `_CONTENT_HASH_RE` name-level telemetry),
  shim non-public-attribute rejection, `__version__` presence + SemVer
  format + pyproject.toml agreement.
- **`py.typed` marker** (REV 2 follow-up) — PEP 561 compliance for the
  `Typing :: Typed` classifier claim. Enables downstream mypy / pyright
  / pyre to pick up the package's inline type annotations instead of
  defaulting to `Any`. Explicit `include` in
  `[tool.hatch.build.targets.wheel]` ensures the marker ships in the
  wheel.

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
  (`hft-ops/stages/signal_export.py`). `_CONTENT_HASH_RE` retained
  through a module-level `__getattr__` shim (REV 2 follow-up —
  originally a silent alias) that emits one-time `DeprecationWarning`
  per process citing the migration path + 2026-10-31 removal deadline.
  Uniform deprecation lifecycle with `_atomic_io.py`.
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

- **Authoring env** (hft-ops installed): 300 passing.
- **Fresh-clone env** (hft-ops absent): 295 passing + 5 skipped
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
- Build: `hatchling >= 1.26` (REV 2 follow-up bump — PEP 639 `license` +
  `license-files` full support in hatchling 1.26+; earlier 1.22 floor
  was too permissive for constrained CI environments).

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
