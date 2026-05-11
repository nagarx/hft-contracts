# Changelog

All notable changes to this project are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/) for the Python package; the
cross-module **contract schema version** is tracked independently via
`SCHEMA_VERSION` in `_generated.py` (locked at `2.2` for this release).

---

## [Unreleased]

## [2.7.0] — 2026-05-11

### Added (#PY-73 closure — atomic binary write SSoT)

- **`hft_contracts.atomic_io.atomic_write_binary(path, write_fn, *, min_bytes=1)`** —
  generic atomic binary-write primitive. Mirrors `atomic_write_json`
  Protocol (tmp + fsync + `os.replace` + BaseException-safe cleanup)
  but for binary file handles. Caller supplies `write_fn` callable
  that receives an opened-for-write binary handle. Empty-write guard
  (`tmp_path.stat().st_size < min_bytes` → `AtomicWriteError`) per
  hft-rules §8. **Closes #PY-73** — the SSoT primitive underlying 3
  typed wrappers + `atomic_copy`.

- **`atomic_write_torch(path, obj, *, _use_new_zipfile_serialization=True)`** —
  atomic `torch.save` wrapper. **Lazy-imports `torch` inside function
  body** to preserve hft-contracts' torch-free invariant at module
  load. Locked by `tests/test_atomic_io_imports.py::test_atomic_io_no_top_level_torch_import`
  AST regression test (mirrors the Cycle C1 `test_contract_preflight_module_imports_are_torch_free`
  precedent). Migrates 2 producer sites in lob-model-trainer
  (`trainer.py:1388` Trainer.save_checkpoint + `callbacks.py:684`
  ModelCheckpoint._save_checkpoint).

- **`atomic_write_npy(path, arr, *, allow_pickle=False)`** — atomic
  `np.save` wrapper. **Rejects non-ndarray inputs explicitly** via
  `TypeError` (no silent `np.asarray` conversion, per hft-rules §8).
  Migrates 14 producer sites: 4 in lob-model-trainer
  `simple_trainer.py:765-768`, 9 in `lobtrainer/export/exporter.py:502-534`,
  1 in lob-backtester `registry.py:124`. `numpy` was already a hft-contracts
  runtime dep (via `label_factory` + `signal_manifest`) — top-level
  import is safe.

- **`atomic_write_pickle(path, obj, *, protocol=pickle.DEFAULT_PROTOCOL)`** —
  atomic `pickle.dump` wrapper. **Default protocol is
  `pickle.DEFAULT_PROTOCOL` (stable across point releases), NOT
  `pickle.HIGHEST_PROTOCOL`** (the newest supported, which may rotate
  forward and break older readers). Caller may pass
  `HIGHEST_PROTOCOL` explicitly when reader Python version is
  controlled. **Both candidate consumer sites DEFERRED**:
  `lob-models/.../base_simple.py:80` (leaf-package architecture —
  lob-models cannot depend on hft-contracts per `pyproject.toml:39-40`)
  and `MBO-LOB-analyzer/.../orchestrator.py:224` (foreign-agent state
  per root CLAUDE.md banner). Tracked as sister #PY-73 follow-ups.

- **`atomic_copy(src, dst, *, min_bytes=1)`** — atomic file copy via
  tmp + `os.replace`. Reads `src` bytes into a tmp file alongside
  `dst`, then atomic-renames. Mirrors `shutil.copy` semantics but
  eliminates the partial-write window. Does NOT preserve file
  metadata (use existing `hft-ops/ledger/ledger.py:611` `shutil.copy2
  → tmp → os.replace` pattern if metadata needed; that path is
  Phase 8C-α-aligned and remains divergent on purpose). Migrates 1
  producer site in lob-model-trainer `callbacks.py:694`
  (ModelCheckpoint copies `<epoch>.pt` to `best.pt` — a LARGE
  100MB-1GB file in the hot per-epoch loop, where SIGKILL-mid-copy
  is the dominant corruption risk).

### Changed

- **Tmp-path generation hardened** with `secrets.token_hex(4)`
  suffix component. Previous 2-tuple suffix
  `<name>.tmp.<pid>.<time_ns>` is now `<name>.tmp.<pid>.<time_ns>.<rand4>`.
  Closes PID-recycle + coarse `time_ns()` granularity (~1µs on
  macOS) collision hazard surfaced by Adv-API-review. Internal
  helper `_make_tmp_path(path)` consolidates the pattern across all
  6 atomic primitives. Existing `atomic_write_json` callers see no
  observable change (tmp filenames are internal).

- **Module docstring** — extended to document the #PY-73 cycle
  rationale, the lazy-import discipline preserving hft-ops torch-free
  invariant, the empty-write guard policy, and the NFS / SMB / FUSE
  atomicity caveat.

### Tests

- **NEW `tests/test_atomic_io.py`** — 31 tests covering all 5 new
  primitives + extended `atomic_write_json` coverage:
  - **Round-trip equality** for torch/npy/pickle/copy (byte-identical
    vs direct serializer)
  - **Empty-write guard** fires on no-op `write_fn` (raises
    `AtomicWriteError`)
  - **`atomic_write_npy` TypeError** on non-ndarray (list/tuple/scalar)
  - **`atomic_write_pickle` protocol default** is `DEFAULT_PROTOCOL`
    not `HIGHEST_PROTOCOL` (forward-compat invariant)
  - **Tmp cleanup** when `write_fn` raises mid-write
  - **`atomic_copy` round-trip** preserves file contents
  - **`AtomicWriteError` IS-A `OSError`** (preserves except-OSError
    compat)
  - **Tmp-path uniqueness** under repeated rapid invocation (collision
    sanity)

- **NEW `tests/test_atomic_io_imports.py`** — AST regression test
  locking the lazy-torch-import discipline. Parses `atomic_io.py`
  module-level imports + asserts `torch` not in any top-level
  `Import`/`ImportFrom` node. Also runs a subprocess sanity check
  (`python -c "import hft_contracts.atomic_io; assert 'torch' not in
  sys.modules"`) to lock the runtime invariant alongside the AST
  invariant.

### Migration notes for consumers (this cycle: 17 of 18 sites migrated)

After hft-contracts v2.7.0 ships, consumer repos should bump
`hft-contracts>=2.7.0` in `pyproject.toml` and migrate non-atomic
write sites:

- **lob-model-trainer SHIPPED** (16 sites): 2 `torch.save` →
  `atomic_write_torch` (`trainer.py:1388` Trainer.save_checkpoint +
  `callbacks.py:684` ModelCheckpoint); 13 `np.save` → `atomic_write_npy`
  (`simple_trainer.py:765-768` 4 sklearn signals +
  `lobtrainer/export/exporter.py:502-534` 9 pytorch signals); 1
  `shutil.copy` → `atomic_copy` (`callbacks.py:694` best.pt
  duplication). Pin bump: `>=2.5.0` → `>=2.7.0`.

- **lob-backtester SHIPPED** (1 site): `registry.py:124` `np.save` →
  `atomic_write_npy`. Pin bump: unpinned → `>=2.7.0`.

- **lob-models DEFERRED** (1 site `base_simple.py:80`): leaf-package
  architecture per `pyproject.toml:39-41` — lob-models cannot depend
  on hft-contracts. Migrating this site requires either (a) lifting
  an inline-pickle atomic helper into lob-models, or (b) relaxing
  the leaf-package invariant. The site is bounded-impact (sklearn
  models are LOAD-once / SAVE-rarely; no per-epoch hot-loop SIGKILL
  hazard). Tracked as sister #PY-73 follow-up. Docstring at
  `base_simple.py:75-87` documents the deferral rationale in-place.

- **hft-ops**: no migrations needed (`ledger.py:611` already uses
  the tmp+os.replace pattern via `shutil.copy2` — preserves
  metadata, divergent-on-purpose vs the metadata-free `atomic_copy`
  per Phase 8C-α design). Pin bump optional (no new symbols
  consumed by hft-ops this cycle).

- **MBO-LOB-analyzer DEFERRED** (1 site `orchestrator.py:224`):
  separate coordination cycle — repo is in foreign-agent dirty
  state per root CLAUDE.md banner (8+ weeks old, no DESIGN-1
  cycle work). Tracked as sister #PY-73 follow-up.

## [2.6.0] — 2026-05-09

### Added (Phase X.3 / REFINED-PLUS Sub-cycle 3 — Phase Y `model_config_hash` top-level projection in `index_entry()`)

- **`ExperimentRecord.index_entry()` projects `model_config_hash` at top level.**
  The Phase Y composer reads `model_config_hash` from
  `training_config["model_config_hash"]` (per `_extract_provenance_components`
  at experiment_record.py:749). That nested value IS populated at trainer
  write time (sklearn at simple_trainer.py sidecar; PyTorch at
  `_build_checkpoint_dict`). Without this projection, `hft-ops ledger list
  --model-config-hash <hex>` queries cannot filter — #PY-94 surfaced this
  gap during γ-1 LITE empirical gate 2026-05-09 night (12 records had
  populated nested mch but 0 top-level projection because the field
  exists only nested in `training_config`). Same regex gate +
  graceful-degradation pattern as `compatibility_fingerprint` (Phase
  V.A.4) and `experiment_provenance_hash` (Phase X.3).

### Changed

- **`INDEX_SCHEMA_VERSION`: `"1.5.0"` → `"1.6.0"` (MINOR additive).**
  Drives the auto-invalidation substrate at `hft-ops/ledger/ledger.py`
  via the `_load_index` envelope writer — existing `index.json`
  on-disk envelopes will auto-rebuild from `records/*.json` on next
  load to pick up the new top-level projection. Operators can also
  run `hft-ops ledger rebuild-index` explicitly.

### Tests

- **`tests/test_experiment_record.py::TestModelConfigHashIndexEntryProjection`**
  — 3 NEW tests locking the projection contract:
  (a) populated nested mch surfaces at top level (uses actual γ-1 LITE
  TLOB arm value `de47c0ef...`);
  (b) unpopulated/missing → `""` graceful degradation across 3 record
  shapes (empty training_config, no training_config, training_config
  without `model_config_hash` key);
  (c) malformed (not lowercase 64-hex) → `""` across 7 invalid value
  shapes (too short / wrong charset / uppercase / length 63 / length 65
  / int / None).
- **`TestIndexEntryCompleteness::test_index_entry_top_level_key_set_frozen`**
  — extended `expected_top_level` set with `model_config_hash`; updated
  version comment to track 1.5.0 → 1.6.0 bump.

### Added (Phase X.3 / REFINED-PLUS Sub-cycle 2 — Phase Y composer fail-loud opt-in + structured provenance diagnostic)

- **`hft_contracts.experiment_record.ProvenanceDiagnostic`** (NEW frozen
  dataclass) — structured diagnostic describing which
  `experiment_provenance_hash` components are present + valid.
  Fields: `complete: bool`, `missing: FrozenSet[str]`,
  `invalid_format: FrozenSet[str]`. Class-level
  `COMPONENT_NAMES: ClassVar[FrozenSet[str]]` is the SSoT for the 4
  fingerprint sources (`data_export_fp` / `feature_set_content_hash` /
  `compatibility_fp` / `model_config_hash`) per hft-rules §1 — eliminates
  magic-string drift between composer + caller.
- **`hft_contracts.experiment_record.diagnose_provenance_completeness(record)`**
  (NEW function) — returns `ProvenanceDiagnostic`. Empty-string values count
  as `missing` (matches producer-side convention: un-populated 64-hex
  string surfaces as `None` or `""`). Invalid-format detection consumes
  `hft_contracts.signal_manifest.CONTENT_HASH_RE` (lowercase 64-hex
  SHA-256). Closes PHASE_P_BACKLOG #PY-49 mitigation (note at L891) — lifts the
  inline diagnostic from `hft-ops/src/hft_ops/cli.py:639-666` into a
  stable home; cli.py DRY refactor lands in Sub-cycle 4b alongside
  composer caller wiring).

### Changed (additive — back-compat preserved)

- **`compute_experiment_provenance_hash(record, *, required: Optional[FrozenSet[str]] = None)`** —
  added the keyword-only `required` parameter. When `None` (default),
  preserves existing silent-None graceful-degradation behavior for legacy
  records (zero back-compat risk). When set, raises `ValueError` on
  missing or invalid-format components in the required-set, plus on
  unknown component names. Closes the
  `lobmodels.registry.protocols.OrchestratorContract` `requires_*`
  contract pre-committed by Sub-cycle 1a — Sub-cycle 4b composer caller
  wiring will consume this with `required=ProvenanceDiagnostic.COMPONENT_NAMES`
  (or a subset per per-trainer `requires_*` flags).

### API surface

- **Package-level re-export**: `compute_experiment_provenance_hash` /
  `diagnose_provenance_completeness` / `ProvenanceDiagnostic` now
  importable via `from hft_contracts import ...` (matches the convention
  established for `CompatibilityContract`, `FeatureImportance`, etc.).
  Submodule import paths remain unchanged.

### Tests

- **`tests/test_experiment_record.py`** — 23 NEW tests collected in 4 NEW
  test classes (20 unique `def test_*` declarations; parametric expansion of
  `test_each_component_missing_via_none` over 4 components yields 4 cases).
  File total: 56 → 79 collected (verified 79 passed in 0.17s):
  - `TestProvenanceDiagnostic` (3) — `COMPONENT_NAMES` SSoT, frozen-dataclass
    immutability via `dataclasses.FrozenInstanceError`, `FrozenSet` field types
  - `TestDiagnoseProvenanceCompleteness` (9) — happy path + parametric
    each-component-missing (4) + empty-string-as-missing + invalid-format
    uppercase / too-short + `provenance=None` access guard
  - `TestComputeExperimentProvenanceHashRequiredArg` (9) — `required=None`
    back-compat, `required=frozenset()` equivalence, missing-component
    raises, invalid-format raises, unknown-name raises, all-required-passes,
    hash-identical-with-vs-without-required, partial-required-set behavior,
    multi-component error-message lists all
  - `TestSubCycle2PackageSurface` (2) — new symbols in `__all__` (module +
    package level)

### Adversarial validation

- 7-agent prep round (3 investigation + 4 adversarial) post-Cycle-C1
  2026-05-09 night converged on REFINED HYBRID resolution: ship
  pre-committed `required` arg AS PROMISED + ADD Alt-2 sibling
  `diagnose_provenance_completeness`. Doc-alignment-auditor verified the
  proposed signature does not break the single production caller
  (`hft-ops/cli.py:635`) and all back-compat tests in
  `hft-ops/tests/test_phase_y_harvest_compose.py`.
- Mid-impl + pre-commit gates per saved feedback memory
  `feedback_final_adversarial_validation_round.md` Standard 2-gate cadence.

### Migration

- **NO API breakage**. Existing callers passing `record` positionally
  continue to work. Existing tests in
  `hft-ops/tests/test_phase_y_harvest_compose.py` exercising the
  silent-None graceful-degradation path continue to pass.
- **Forward path**: callers wanting fail-loud opt-in pass
  `required=ProvenanceDiagnostic.COMPONENT_NAMES` (or a subset).
  Sub-cycle 4b will refactor `hft-ops/src/hft_ops/cli.py:639-666` to
  consume the new `diagnose_provenance_completeness` SSoT (DRY win +
  closes PHASE_P_BACKLOG #PY-49 mitigation (note at L891)).

---

## [2.5.0] — 2026-05-07

### Added (Phase 2 P2.C — Cyclelet B K-way pairwise-compare artifact contract)

- **`hft_contracts.pairwise_compare_artifact`** (NEW module) — frozen-
  dataclass contract for Phase 2 P2.C K-way pairwise-comparison artifacts
  produced by `lobtrainer.analysis.stat_rigor.pairwise.compare_k_way`.
  Mirrors the v2.4.0 `TestMetricsCIArtifact` pattern.
  - **`PairwiseResultRecord`** — per-pair (treatment_a_idx, treatment_b_idx,
    treatment_a_label, treatment_b_label, statistic_a, statistic_b, delta,
    delta_ci_low, delta_ci_high, p_value_raw, p_value_bh, n_nonfinite_replaced).
    `__post_init__` validates finiteness + index ordering + delta CI invariant
    + p-value range + treatment label non-empty. `from_hft_metrics_result`
    classmethod bridges runtime `hft_metrics.pairwise.PairwiseResult` → artifact.
  - **`PairwiseCompareArtifact`** — full K-way artifact. Fields:
    schema_version, method, metric_name, block_length (+source), n_bootstraps,
    alpha, seed, n_treatments (K>=2), n_samples_paired/raw, n_dropped_nonfinite,
    drop_fraction, primary_horizon_idx; **parallel-indexed Phase Y composability
    tuples**: `parent_experiment_ids`, `parent_compatibility_fingerprints`,
    `parent_model_config_hashes` (Optional[str] for pre-Phase-Q.6.5 sklearn);
    `paired_compat_fingerprint` (SHARED — all K must equal, validated);
    `paired_labels_sha256` (verifies all K consume same labels);
    `pairs: Tuple[PairwiseResultRecord, ...]` length K*(K-1)/2;
    `treatment_labels`; `timestamp_utc`; `method_caveats`.
    Construction-time validation rejects degenerate parameters per hft-rules
    §5/§8 (n_treatments<2, K-pair count mismatch, parallel-tuple length
    mismatch, alpha ∉ (0,1), invalid SHA-256 hex, mismatched compat_fp,
    drop_fraction inconsistency).
  - `content_hash()` delegates to `hft_contracts.canonical_hash` SSoT.
  - `save()` delegates to `hft_contracts.atomic_io.atomic_write_json` SSoT.
  - `from_dict()` migration shim accepts forward-compatible additive
    schema bumps; preserves `Optional[str]` None values for sklearn
    pre-Phase-Q.6.5 model_config_hashes.
  - `get_pair(a_idx, b_idx)` + `get_pair_by_labels(label_a, label_b)`
    O(K^2) lookup helpers.
- **`PAIRWISE_COMPARE_SCHEMA_VERSION = "1"`** module constant.
- **`pipeline_contract.toml::[artifacts.pairwise_compare_schema]`** —
  registers the artifact with `kind = "pairwise_compare"` for hft-ops
  ledger routing. Companion to the hft-ops `_POST_STAGE_ARTIFACT_PATTERNS`
  row (registered separately in hft-ops repo).
- **Phase Y composability**: artifact integrates with future
  `experiment_provenance_hash` graph as a "comparison node" with K
  parent provenance-hash references via parallel-indexed `parent_*` tuples.
- **30+ new tests** at `tests/test_pairwise_compare_artifact.py`.

### Architectural notes

- **K-arbitrary support** (Round 2 Agent B finding): K=2 is special case
  of K-way; K>=3 enables meaningful BH FDR correction (K=2 BH ≡ raw p).
- **Strict `compat_fp` invariant** (Round 2 Agent B HIGH finding): all K
  treatments must share `compatibility_fingerprint` (paired comparison
  requires shared paired-data) — `__post_init__` raises `ValueError`
  on divergence with diagnostic listing the mismatching fingerprints.
- **Effect size in artifact** (Round 2 Agent B HIGH finding): per-pair
  surface `statistic_a`, `statistic_b`, `delta`, `delta_ci_low`,
  `delta_ci_high` — without effect size, BH q-value alone tells nothing
  about practical relevance (per "many experiments empirically traceable"
  user mandate).

### Notes

- **No SCHEMA_VERSION (`_generated.py`) bump**: P2.C artifact is a
  POST-EXPERIMENT statistical-comparison artifact; does NOT modify the
  data contract. Schema version remains at `3.0`.
- **Cycle origin**: Plan v4 §4.3 (`PHASE_2_STAT_RIGOR_PLAN.md`) +
  Round 2 architectural critique (3 agents → 6 HIGH revisions applied).

## [2.4.0] — 2026-05-07

### Added (Phase 2 P2.A — Cyclelet B bootstrap-CI artifact contract)

- **`hft_contracts.test_metrics_ci_artifact`** (NEW module) — frozen-dataclass
  contract for Phase 2 P2.A bootstrap-CI artifacts produced by
  `lobtrainer.analysis.stat_rigor.ci`. Mirrors the Phase 8C-α Stage C.2
  `FeatureImportanceArtifact` precedent.
  - **`MetricCIBound`** — per-metric `(point, ci_low, ci_high, n_samples)`
    bound. `__post_init__` validates finiteness + `ci_low <= point <=
    ci_high` invariant + `n_samples > 0` (Round 1 mid-impl adversarial
    finding §2 HIGH — leaf-type validation surfaces clearer errors than
    parent artifact's cross-leaf check).
  - **`TestMetricsCIArtifact`** — full artifact carrying `schema_version`,
    `method`, `block_length` (+source string), `n_bootstraps`, `ci`, `seed`,
    `n_test_samples`, `metrics` dict, plus traceability fields
    (`compatibility_fingerprint`, `model_config_hash`,
    `normalization_stats_sha256`, `signal_export_output_dir`,
    `experiment_id`, `fingerprint`, `model_type`, `timestamp_utc`,
    `method_caveats`). Construction-time validation rejects degenerate
    parameters per hft-rules §5/§8 (n_test_samples<=0, n_bootstraps<100,
    block_length<2, ci ∉ (0,1), empty metrics, invalid SHA-256 hex).
  - `content_hash()` delegates to `hft_contracts.canonical_hash` SSoT
    (NO re-derivation per §0).
  - `save()` delegates to `hft_contracts.atomic_io.atomic_write_json`
    SSoT.
  - `from_dict()` migration shim accepts forward-compatible additive
    schema bumps; `load()` strict on required fields per §8.
- **`TEST_METRICS_CI_SCHEMA_VERSION = "1"`** module constant — bumped
  MAJOR for breaking field rename/remove; MINOR for additive new fields
  with `None` defaults.
- **`pipeline_contract.toml::[artifacts.test_metrics_ci_schema]`** —
  registers the artifact with `kind = "test_metrics_ci"` for hft-ops
  ledger routing. Required fields, optional fields, per-metric fields,
  content_hash algorithm + module documented inline.
- **Phase Y composability**: artifact integrates with future
  `experiment_provenance_hash` via `record.artifacts[].sha256` projection
  through hft-ops ledger router (matches `feature_importance_v1.json`
  precedent).
- **40 new tests** at `tests/test_test_metrics_ci_artifact.py`:
  TestPublicAPI (2), TestConstruction (3), TestRoundTrip (3),
  TestContentHash (5), TestSaveLoad (3), TestValidation (16 — incl.
  parametric hex-format check + 3 leaf MetricCIBound validation tests),
  TestGetMetric (2). Test count: 594 → 597 (+3 net) at this release.

### Notes

- **No SCHEMA_VERSION (`_generated.py`) bump**: P2.A artifact is a
  POST-EXPERIMENT statistical-analysis artifact; it does NOT modify the
  data contract (feature indices 0-97 stable). Schema version remains
  at `3.0` (locked since Phase G G.6.A).
- **Cycle origin**: Plan v4 §4.1 (`PHASE_2_STAT_RIGOR_PLAN.md`), Round 1
  ground-truth verification (5 agents) + Round 2 architectural critique
  (3 agents) + Round 1 mid-impl adversarial review (1 agent, 5 BLOCKING
  revisions applied) + Pre-commit gate (1 agent, 1 BLOCKING revision
  applied — version bump + this CHANGELOG entry).
- **First empirical use** (2026-05-07): 6 R-series candidates at v3p0
  corpus (TLOB no-CVML / TLOB+CVML / TLOB+GMADL+CVML / HMHP-R /
  TemporalGradBoost / TemporalRidge), 10K bootstraps each, ~4 min total
  compute. Empirically reproduced CLAUDE.md "GMADL collapse" finding
  (R11 IC CI = [-0.030, +0.019] includes 0) and "CVML doesn't transfer"
  ranking (R10 IC=0.346 < R9 IC=0.375).

## [Unreleased pre-Phase-2 work]

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
