"""Regression locks for the from_dict null-collection crash family.

Audit finding #1 (2026-05-28 fresh-eye adversarial review): the H-2 fix
(ExperimentRecord.from_dict on ``provenance: null``) was the shallowest
instance of a 7-site bug class. A key PRESENT with a literal ``null``
value flows past ``dict.get(key, default)`` (the default applies only to
ABSENT keys) and then crashes at ``dict(None)`` / ``None.items()`` /
``tuple(None)`` / ``for x in None``.

CONTRACT LOCKED HERE: every ``from_dict`` (and the deep nested case +
``index_entry`` projection) must tolerate present-but-null collection
fields by coercing them to their empty default — the same ``x or {}`` /
``x or []`` / ``x or ()`` pattern the H-2 fix established.

Each test would crash (AttributeError / TypeError) if its fix were
reverted.
"""

from __future__ import annotations

import pytest

from hft_contracts.provenance import Provenance, GitInfo
from hft_contracts.experiment_record import ExperimentRecord


def _fia_dict(**overrides):
    d = dict(
        schema_version="2", method="permutation", baseline_metric="ic",
        baseline_value=0.1, block_length_samples=1, n_permutations=1,
        n_seeds=1, seed=1, eval_split="test", features=[],
        feature_set_ref=None, experiment_id="e", fingerprint="", model_type="m",
        timestamp_utc="t", method_caveats=[],
    )
    d.update(overrides)
    return d


class TestProvenanceFromDictNull:
    def test_git_null_coerced(self):
        """provenance.git: null → default GitInfo (was AttributeError)."""
        p = Provenance.from_dict({"git": None})
        assert isinstance(p.git, GitInfo)

    def test_config_hashes_null_coerced(self):
        p = Provenance.from_dict({"config_hashes": None})
        assert p.config_hashes == {}

    def test_experiment_record_nested_git_null(self):
        """The proven end-to-end crash: ExperimentRecord.load on a record
        with provenance.git: null crashed the whole ledger index build."""
        rec = ExperimentRecord.from_dict({
            "experiment_id": "x", "name": "n", "status": "completed",
            "provenance": {"git": None, "contract_version": "3.0"},
        })
        assert isinstance(rec.provenance.git, GitInfo)


class TestExperimentRecordIndexEntryNull:
    def test_training_config_model_null(self):
        """training_config={'model': null} crashed index_entry() at
        projection time (delayed failure, loads fine then crashes)."""
        rec = ExperimentRecord.from_dict({
            "experiment_id": "x", "name": "n", "status": "completed",
            "training_config": {"model": None, "data": None},
        })
        entry = rec.index_entry()
        assert entry["model_type"] == ""
        assert entry["labeling_strategy"] == ""


class TestFeatureImportanceArtifactFromDictNull:
    def test_features_null_coerced(self):
        from hft_contracts.feature_importance_artifact import FeatureImportanceArtifact
        art = FeatureImportanceArtifact.from_dict(_fia_dict(features=None))
        assert art.features == ()

    def test_method_caveats_null_coerced(self):
        from hft_contracts.feature_importance_artifact import FeatureImportanceArtifact
        art = FeatureImportanceArtifact.from_dict(_fia_dict(method_caveats=None))
        assert art.method_caveats == ()


class TestTestMetricsCIArtifactFromDictNull:
    def _base(self, **overrides):
        d = dict(
            schema_version="1", method="moving_block_bootstrap",
            block_length=10, block_length_source="auto", n_bootstraps=1000,
            ci=0.95, seed=1, n_test_samples=100, metrics={},
            method_caveats=[],
        )
        d.update(overrides)
        return d

    def test_metrics_null_coerced(self):
        from hft_contracts.test_metrics_ci_artifact import TestMetricsCIArtifact
        # metrics=None coerces to {} → __post_init__ then raises a CLEAN
        # ValueError (non-empty required), not a cryptic None.items() crash.
        with pytest.raises(ValueError):
            TestMetricsCIArtifact.from_dict(self._base(metrics=None))

    def test_method_caveats_null_coerced(self):
        from hft_contracts.test_metrics_ci_artifact import TestMetricsCIArtifact, MetricCIBound
        art = TestMetricsCIArtifact.from_dict(self._base(
            metrics={"ic": {"point": 0.1, "ci_low": 0.0, "ci_high": 0.2, "n_samples": 100}},
            method_caveats=None,
        ))
        assert art.method_caveats == ()


class TestFeatureSetFromDictNull:
    def _base(self, **overrides):
        # A minimal valid FeatureSet dict; content_hash recomputed via build
        # is bypassed by from_dict(verify=False) so we can test coercion only.
        from hft_contracts.feature_sets.hashing import compute_feature_set_hash
        indices = [0, 5, 12]
        ch = compute_feature_set_hash(indices, 98, "3.0")
        d = dict(
            schema_version="1.0", name="t", content_hash=ch,
            contract_version="3.0", source_feature_count=98,
            applies_to={"assets": ["NVDA"], "horizons": [10]},
            feature_indices=indices, feature_names=[],
            produced_by={
                "tool": "x", "tool_version": "1", "config_path": "c.yaml",
                "config_hash": "h", "source_profile_hash": "s",
                "data_export": "exp", "data_dir_hash": "d",
            },
            criteria={}, created_at="2026-01-01",
        )
        d.update(overrides)
        return d

    def test_criteria_null_coerced(self):
        from hft_contracts.feature_sets.schema import FeatureSet
        fs = FeatureSet.from_dict(self._base(criteria=None), verify=False)
        assert fs.criteria == {}

    def test_feature_names_null_coerced(self):
        from hft_contracts.feature_sets.schema import FeatureSet
        fs = FeatureSet.from_dict(self._base(feature_names=None), verify=False)
        assert fs.feature_names == ()


class TestSha256RegexParity:
    """Lock the intentional duplication between _validators._SHA256_HEX_RE
    and signal_manifest.CONTENT_HASH_RE (audit finding #5). They are
    deliberately separate (dependency-direction hygiene) but MUST stay
    semantically identical."""

    def test_patterns_identical(self):
        from hft_contracts._validators import _SHA256_HEX_RE
        from hft_contracts.signal_manifest import CONTENT_HASH_RE
        assert _SHA256_HEX_RE.pattern == CONTENT_HASH_RE.pattern

    def test_same_accept_reject_across_vectors(self):
        from hft_contracts._validators import _SHA256_HEX_RE
        from hft_contracts.signal_manifest import CONTENT_HASH_RE
        vectors = ["a" * 64, "A" * 64, "a" * 63, "a" * 65, "",
                   "0123456789abcdef" * 4, "g" * 64, "  " + "a" * 62]
        for v in vectors:
            assert bool(_SHA256_HEX_RE.match(v)) == bool(CONTENT_HASH_RE.match(v)), v
