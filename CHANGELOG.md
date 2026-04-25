# Changelog

All notable changes to this project are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/) for the Python package; the
cross-module **contract schema version** is tracked independently via
`SCHEMA_VERSION` in `_generated.py` (locked at `2.2` for this release).

---

## [Unreleased]

### Added (Phase A.5.6 + A.5.7b — bug #6 round-trip regression locks)

- **`tests/test_compatibility_contract.py::TestPostInitInvariants::test_horizons_list_tuple_round_trip_fingerprint_stable`**
  — Phase A.5.6 (2026-04-24) ship-blocker test for plan v4 bug #6.
  Existing tests proved list-vs-tuple equivalence at construction time;
  this test proves the invariant survives the FULL producer → JSON wire
  → consumer reconstruction path (`CompatibilityContract(horizons=[10,60,300])`
  → `to_canonical_dict()` → `json.dumps` → `json.loads` → re-construct →
  `fingerprint()` byte-identical to producer's). Closes the gap for the
  realistic trainer → signal_metadata.json → backtester hop.

No code changes — additive test coverage only. SCHEMA_VERSION unchanged
at 2.2; package version unchanged at 2.3.0 (no need to bump for tests).

---

## [2.3.0] — 2026-04-24

**Phase A.5.1** — two additive primitives shipped together as a single MINOR
bump. Both are defensive prep for the Phase A.5 Scope D v4 dacite→Pydantic
migration of `lob-model-trainer` (trainer migration lands in A.5.3a–i). No
breaking changes; contract schema version unchanged at `2.2`.

### Added (Phase A.5.1, 2026-04-24)

- **`hft_contracts.timestamp_utils`** — ISO-8601 UTC-aware parsing + cutoff
  comparison SSoT. Two helpers:
    - `parse_iso8601_utc(ts: str) -> datetime` — returns a timezone-aware UTC
      datetime. Handles naive timestamps (interpreted as UTC per hft-rules
      §3), the `Z` suffix (normalized to `+00:00` for Python < 3.11
      compat), and explicit offsets (converted via `astimezone`). Raises
      `TypeError` on non-string input (prevents silent coercion bugs) and
      `ValueError` on malformed strings (fail-loud per §5).
    - `is_after_cutoff(timestamp: str, cutoff_iso: str) -> bool` — strict
      UTC `>=` comparison. Replaces the lexicographic `ts >= cutoff` pattern
      that fails silently for non-UTC offsets (e.g., `"2026-04-22T23:59-05:00"`
      is strictly after UTC cutoff `"2026-04-23"` but lex-compares as `<`).
  Rationale: Phase V.A.4 added
  `FINGERPRINT_REQUIRED_AFTER_ISO = "2026-04-23"` + a lexicographic cutoff
  check in `hft_ops.stages.signal_export`. Operators shipping from a
  non-UTC producer would silently fall into the pre-cutoff branch — a
  silent-wrong-result class that violates hft-rules §8. A.5.2 migrates
  `hft-ops` to use `is_after_cutoff`.
  Re-exported at package level: `from hft_contracts import parse_iso8601_utc, is_after_cutoff`.
  15 new tests in `tests/test_timestamp_utils.py` (naive-as-UTC, non-UTC
  offset crossing midnight, Z suffix, malformed raises, TypeError on
  None/bytes/int, empty-string raises, fractional seconds preserved,
  epoch edge, pre/post/boundary cutoff, malformed-timestamp/cutoff
  propagation).
- **`tests/fixtures/pre_pydantic_label_strategy_hash.json`** — byte-identity
  snapshot of `compute_label_strategy_hash` output for the default
  `LabelsConfig()` dataclass field values at 2026-04-24. Locks the
  invariant that the A.5.3a Pydantic migration does NOT rotate any
  stored ledger `compatibility_fingerprint`.
- **`tests/fixtures/pre_pydantic_compatibility_fingerprint.json`** —
  byte-identity snapshot of `CompatibilityContract.fingerprint()` for a
  realistic 98-feat / window=100 / mbo_lob contract. Locks the outer
  11-key fingerprint across any future canonicalization change.
- **3 new byte-identity lock tests** in `tests/test_compatibility_contract.py::TestPydanticParity`:
    - `test_compute_label_strategy_hash_strips_private_prefix_in_vars_fallback`
      — non-tautological: constructs two instances with different
      `_internal_cache` values and asserts identical hashes (proving the
      strip actually runs, not just that output is 64-hex).
    - `test_compute_label_strategy_hash_accepts_pydantic_basemodel` — the
      ship-blocker. Mock Pydantic v2 BaseModel with fields matching the
      real `LabelsConfig` defaults; asserts hash byte-equals the frozen
      dataclass fixture. If this test fails at A.5.3a time, every
      post-migration record's `compatibility_fingerprint` would rotate.
    - `test_compatibility_fingerprint_byte_stability_against_frozen_fixture`
      — outer 11-key fingerprint locked against the frozen payload.

### Changed (Phase A.5.1, 2026-04-24)

- **`hft_contracts.compatibility.compute_label_strategy_hash`** — new
  dispatch order: (1) Pydantic v2 BaseModel (`hasattr model_dump`) →
  `.model_dump(exclude_none=False)`; (2) @dataclass → `asdict()`; (3)
  plain dict → shallow copy; (4) object with `__dict__` → `vars()` with
  `_`-prefix strip; (5) else → `TypeError` (fail-loud per §5).
  **Why Pydantic MUST come first**: a Pydantic BaseModel has BOTH
  `.model_dump()` AND `.__dict__`. Without this ordering, the `__dict__`
  fallback would leak Pydantic internals
  (`__pydantic_fields_set__`, `__pydantic_private__`, `__pydantic_extra__`)
  into the canonical payload → the migration at A.5.3a would silently
  rotate every ledger record's `compatibility_fingerprint`. The new
  dispatch is locked by the A.5.1 byte-identity tests.
  **Backward compatibility**: the dataclass / dict / private-stripped
  `vars()` branches remain; behavior on those inputs is unchanged. The
  previously-silent fallback `{"value": obj}` for unknown types now
  raises `TypeError` — no known caller relied on it (verified via
  monorepo grep).
- **`__version__`** bumped `"2.2.0"` → `"2.3.0"` (additive MINOR).
- **`pyproject.toml`** version bumped to `2.3.0` accordingly.

### Test counts (authoring env post-A.5.1)

- `test_timestamp_utils.py`: **+15** tests (all new — defensive coverage
  added 3 beyond the plan's "12 tests" target for TypeError on None /
  bytes / int, which are each distinct silent-coercion bug classes).
- `test_compatibility_contract.py`: **+3** tests (TestPydanticParity).
- Total hft-contracts: authoring env was 499 pre-A.5.1 → **517** after.

### Operator-facing migration notes (Phase A.5.1 → A.5.2 → A.5.3a)

- hft-ops downstream consumer bump: `hft-ops/pyproject.toml` must pin
  `hft-contracts>=2.3.0` (ships in A.5.2 consumer commit).
- A.5.3a (trainer `LabelsConfig` → Pydantic BaseModel) is unblocked by
  this release. The `test_compute_label_strategy_hash_accepts_pydantic_basemodel`
  test is the cross-module byte-identity guarantee — if the real
  `LabelsConfig` fields ever diverge from the fixture's
  `_FixtureLabelsConfig` mirror, the A.5.3a parity test
  (`test_label_strategy_hash_real_pydantic_parity`) will fire.

---

## [2.2.1] — 2026-04-21

Phase V.A.1 — Activate Phase 0 forward-pass regression detection. Non-breaking
PATCH: pure fixture content addition + populate-script SSoT alignment. Contract
schema version unchanged at 2.2.

### Added (Phase V.A.1, 2026-04-21)

- **`tests/fixtures/phase0_benchmark/golden_values.json::forward_pass`** — now
  populated with two HMHP-family entries (previously `{}`, so
  `lob-models/tests/integration/test_phase0_forward_pass.py` silently
  `pytest.skip`ped its 14 regression tests):
    - `hmhp_classifier`: pinned classifier goldens — `logits_shape`/`logits_hash`,
      `horizon_logits_hashes` for horizons `[10, 60, 300]`, `agreement_hash`,
      `confidence_hash`, and `compute_loss.value` + per-horizon components
      (tolerance `rel=1e-6`).
    - `hmhp_regressor`: parallel goldens — `horizon_predictions_hashes` + loss
      components (regression head; no argmax).
    - Both pinned at HEAD `2026-04-20` (pre-Phase-I.A / pre-Phase-I.B state;
      `pinned_by_phase: "0.3"`). Phase I.A FRESH-2 fix will intentionally drift
      `agreement_hash` + add `nonzero_fraction`; Phase I.B `compute_loss`
      refactor (reduction + sample_weights + pooling) will drift loss values —
      both updates flow through the populate script + CHANGELOG delta, not
      silent golden overwrites. TLOB/XGBoost/DeepLOB/MLPLOB goldens deferred
      to Phase 0.5 per `populate_forward_pass_goldens.py:24-28`.
- **`forward_pass_populated_at_utc`** top-level timestamp — distinct from
  `generated_at_utc` (fixture generation). Operators triaging a drift can tell
  fixture bytes from goldens bytes apart.

### Changed (Phase V.A.1, 2026-04-21)

- **`lob-models/tests/integration/populate_forward_pass_goldens.py`** — final
  JSON write now delegates to `hft_contracts.atomic_io.atomic_write_json`
  (previously `golden_path.write_text(json.dumps(...))`, non-atomic). Resolves
  the v3.0-audit discrepancy: partial-file corruption on SIGKILL / Ctrl-C
  mid-write could previously leave an inconsistent goldens file that tests
  would then consume as truth. The SSoT also enforces canonical
  `sort_keys=True` + trailing-newline convention so goldens are diff-stable
  across regeneration. Matches the pattern used by `ExperimentRecord.save()`,
  `FeatureImportanceArtifact.save()`, and the feature-sets writer. No behavior
  change on happy path; failure mode is now "original bytes retained" rather
  than "partial write committed".

### Verification

- `lob-models/tests/integration/test_phase0_forward_pass.py`: **14 / 14 PASS**
  locally (Darwin ARM64, Python 3.14.2, PyTorch 2.x with
  `torch.use_deterministic_algorithms(True, warn_only=True)`). Tests were
  previously SKIP'd. Cross-platform drift risk (Mac ARM → Linux x86 CI)
  flagged in Phase V risk table as "Low likelihood / HIGH impact";
  escalation-to-tolerance-based-comparison is the pivot path if CI surfaces a
  bit-exact-hash mismatch.
- `hft-contracts` full suite: **467 pass, 3 warnings, 0 regressions** (vs 465
  at Phase II v2.22.1 close — the +2 delta is re-collected parametrized tests,
  not new tests; no test files were added or modified in hft-contracts for
  V.A.1).

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
