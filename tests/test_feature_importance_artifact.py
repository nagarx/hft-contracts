"""Phase 8C-α Stage C.2 — tests for FeatureImportanceArtifact contract.

Locks the schema invariants BEFORE trainer produces + ledger routes.
Any future MAJOR-bump migration must update this file AND the consumer
code in the same commit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hft_contracts import (
    FeatureImportance,
    FeatureImportanceArtifact,
    FEATURE_IMPORTANCE_SCHEMA_VERSION,
)
from hft_contracts.feature_importance_artifact import compute_stability


def _make_feature(name: str = "f", index: int = 0) -> FeatureImportance:
    """Test fixture with non-trivial values so serialization catches bugs."""
    return FeatureImportance(
        feature_name=name,
        feature_index=index,
        importance_mean=0.025,
        importance_std=0.004,
        ci_lower_95=0.017,
        ci_upper_95=0.033,
        n_permutations=500,
        n_seeds_aggregated=5,
        stability=0.84,
    )


def _make_artifact(**overrides) -> FeatureImportanceArtifact:
    base = dict(
        schema_version=FEATURE_IMPORTANCE_SCHEMA_VERSION,
        method="permutation",
        baseline_metric="val_ic",
        baseline_value=0.245,
        block_size_days=1,
        n_permutations=500,
        n_seeds=5,
        seed=42,
        eval_split="test",
        features=(_make_feature("depth_norm_ofi", 85), _make_feature("spread_bps", 42)),
        feature_set_ref={"name": "nvda_short_term_40_src128_v1", "content_hash": "a" * 64},
        experiment_id="exp_20260420T120000_abc123",
        fingerprint="b" * 64,
        model_type="tlob",
        timestamp_utc="2026-04-20T12:00:00+00:00",
        method_caveats=("correlation-split",),
    )
    base.update(overrides)
    return FeatureImportanceArtifact(**base)


class TestFeatureImportanceArtifactRoundTrip:
    """Serialization fidelity: to_dict / from_dict / save / load preserve
    every field. If any field is dropped or mis-coerced, downstream
    feedback-merge decisions become unreliable.
    """

    def test_to_dict_and_from_dict_round_trip(self):
        original = _make_artifact()
        reloaded = FeatureImportanceArtifact.from_dict(original.to_dict())
        assert reloaded == original, (
            f"to_dict → from_dict round trip MUST preserve equality. "
            f"Any dropped/mis-coerced field breaks the contract."
        )

    def test_save_and_load_file_round_trip(self, tmp_path: Path):
        original = _make_artifact()
        path = tmp_path / "feature_importance_v1.json"
        original.save(path)
        assert path.exists()
        reloaded = FeatureImportanceArtifact.load(path)
        assert reloaded == original

    def test_load_malformed_json_raises(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("{ this is not valid JSON")
        with pytest.raises(json.JSONDecodeError):
            FeatureImportanceArtifact.load(path)

    def test_from_dict_tolerates_missing_optional_fields(self):
        """MINOR-bump new fields with defaults load gracefully from
        legacy artifacts that lack them."""
        legacy = _make_artifact().to_dict()
        # Simulate a legacy v1 artifact without method_caveats
        legacy.pop("method_caveats", None)
        reloaded = FeatureImportanceArtifact.from_dict(legacy)
        assert reloaded.method_caveats == ()


class TestContentHash:
    """Content-addressing: same data → same hash; any field change → different hash."""

    def test_same_data_produces_same_hash(self):
        a = _make_artifact()
        b = _make_artifact()
        assert a.content_hash() == b.content_hash(), (
            "Two artifacts with identical data MUST produce the same "
            "content hash (used by ledger routing for de-duplication)."
        )

    def test_hash_is_64_char_lowercase_hex(self):
        h = _make_artifact().content_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_any_field_mutation_changes_hash(self):
        base = _make_artifact()
        variants = [
            _make_artifact(baseline_value=0.246),  # tiny float diff
            _make_artifact(seed=43),
            _make_artifact(method="shap"),
            _make_artifact(feature_set_ref={"name": "other", "content_hash": "a" * 64}),
            _make_artifact(method_caveats=("different",)),
        ]
        base_hash = base.content_hash()
        for v in variants:
            assert v.content_hash() != base_hash, (
                f"Hash must change when any field mutates; {v} did not. "
                f"Missed-mutation breaks content-addressed de-duplication."
            )


class TestFeatureImportanceArtifactLookup:
    def test_get_by_name_returns_feature(self):
        a = _make_artifact()
        found = a.get_by_name("depth_norm_ofi")
        assert found is not None
        assert found.feature_index == 85

    def test_get_by_name_returns_none_on_missing(self):
        a = _make_artifact()
        assert a.get_by_name("nonexistent_feature") is None


class TestComputeStability:
    """Stability metric: 1 - std/|mean| clipped to [0,1]."""

    def test_perfect_stability_zero_std(self):
        assert compute_stability(mean=0.1, std=0.0) == 1.0

    def test_zero_stability_when_std_dominates(self):
        # std >> |mean| → stability saturated at 0
        assert compute_stability(mean=0.01, std=100.0) == 0.0

    def test_near_zero_mean_drives_stability_down(self):
        """Near-zero importance → low stability regardless of std,
        because the metric (std / |mean|) explodes. This is DESIRED —
        feedback-merge should NOT rely on near-zero-importance features.
        """
        s = compute_stability(mean=1e-15, std=0.001)
        assert s < 0.1, f"Near-zero mean must yield low stability; got {s}"

    def test_nan_inputs_yield_zero(self):
        assert compute_stability(mean=float("nan"), std=0.01) == 0.0
        assert compute_stability(mean=0.1, std=float("inf")) == 0.0


class TestExperimentRecordArtifactsField:
    """Phase 8C-α Stage C.2: ExperimentRecord.artifacts field + index_entry
    projects artifact_kinds. Tests the integration between the artifact
    surface and the ledger record.
    """

    def _record_with_artifacts(self, artifacts):
        from hft_contracts.experiment_record import ExperimentRecord
        from hft_contracts.provenance import GitInfo, Provenance
        return ExperimentRecord(
            experiment_id="e",
            name="e",
            fingerprint="f" * 64,
            contract_version="2.2",
            status="completed",
            created_at="2026-04-20T00:00:00+00:00",
            artifacts=artifacts,
            provenance=Provenance(
                git=GitInfo(commit_hash="x", branch="main", dirty=False),
                contract_version="2.2",
            ),
        )

    def test_default_empty_list(self):
        """Back-compat: records without artifacts default to []."""
        rec = self._record_with_artifacts([])
        assert rec.artifacts == []

    def test_artifact_kinds_projection_empty(self):
        """Empty artifacts → empty artifact_kinds list (not None)."""
        rec = self._record_with_artifacts([])
        entry = rec.index_entry()
        assert "artifact_kinds" in entry
        assert entry["artifact_kinds"] == []

    def test_artifact_kinds_projection_extracts_distinct_sorted(self):
        """Multiple artifacts → sorted set of distinct kinds."""
        rec = self._record_with_artifacts([
            {"kind": "feature_importance", "path": "a", "sha256": "x", "bytes": 1},
            {"kind": "feature_importance", "path": "b", "sha256": "y", "bytes": 2},
            {"kind": "shap", "path": "c", "sha256": "z", "bytes": 3},
        ])
        entry = rec.index_entry()
        assert entry["artifact_kinds"] == ["feature_importance", "shap"], (
            "Must be sorted (hft-rules §7 no dict/set-ordering reliance) "
            "and deduplicated"
        )

    def test_artifact_kinds_ignores_malformed_entries(self):
        """Defensive: artifacts[] is Dict[str, Any]; filter out entries
        without a ``kind`` field or with non-dict shape.
        """
        rec = self._record_with_artifacts([
            {"kind": "feature_importance", "path": "a", "sha256": "x", "bytes": 1},
            {},  # empty dict — no kind
            "not_a_dict",  # wrong shape
            {"path": "b"},  # no kind
        ])
        entry = rec.index_entry()
        assert entry["artifact_kinds"] == ["feature_importance"]


class TestIndexSchemaVersion1_3_0:
    """Phase 8C-α Stage C.2: version bump lock."""

    def test_index_schema_version_is_1_3_0(self):
        from hft_contracts import INDEX_SCHEMA_VERSION
        assert INDEX_SCHEMA_VERSION == "1.3.0", (
            f"Expected INDEX_SCHEMA_VERSION='1.3.0' after Phase 8C-α "
            f"Stage C.2 bump (added ``artifact_kinds`` projection + "
            f"``artifacts`` field). Got {INDEX_SCHEMA_VERSION!r}."
        )

    def test_feature_importance_schema_version_is_1(self):
        assert FEATURE_IMPORTANCE_SCHEMA_VERSION == "1"
