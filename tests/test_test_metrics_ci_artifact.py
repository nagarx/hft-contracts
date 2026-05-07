"""Tests for hft_contracts.test_metrics_ci_artifact (Phase 2 P2.A — 2026-05-07).

Mirrors the structure of `test_feature_importance_artifact.py` (Phase 8C-α
Stage C.2 precedent). Locks the contract for `TestMetricsCIArtifact` +
`MetricCIBound` so future schema evolution + content-addressed routing
are stable.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hft_contracts.test_metrics_ci_artifact import (
    TEST_METRICS_CI_SCHEMA_VERSION,
    MetricCIBound,
    TestMetricsCIArtifact,
)


def _make_bound(
    point: float = 0.5,
    ci_low: float = 0.4,
    ci_high: float = 0.6,
    n_samples: int = 8085,
) -> MetricCIBound:
    return MetricCIBound(
        point=point,
        ci_low=ci_low,
        ci_high=ci_high,
        n_samples=n_samples,
    )


def _make_artifact(
    *,
    metrics: dict | None = None,
    n_test_samples: int = 8085,
    n_bootstraps: int = 10_000,
    block_length: int = 21,
    ci: float = 0.95,
    compatibility_fingerprint: str | None = "0" * 64,
    model_config_hash: str | None = None,
    normalization_stats_sha256: str | None = None,
    method_caveats: tuple = (),
) -> TestMetricsCIArtifact:
    if metrics is None:
        metrics = {
            "test_ic": _make_bound(0.37466, 0.36, 0.39, n_test_samples),
            "test_directional_accuracy": _make_bound(0.642, 0.62, 0.66, n_test_samples),
        }
    return TestMetricsCIArtifact(
        schema_version="1",
        method="block_bootstrap",
        block_length=block_length,
        block_length_source="auto-derive ceil(n^(1/3))",
        n_bootstraps=n_bootstraps,
        ci=ci,
        seed=42,
        n_test_samples=n_test_samples,
        metrics=metrics,
        compatibility_fingerprint=compatibility_fingerprint,
        model_config_hash=model_config_hash,
        normalization_stats_sha256=normalization_stats_sha256,
        signal_export_output_dir=None,
        experiment_id="nvda_first_pytorch_v3p0",
        fingerprint="abc123",
        model_type="tlob",
        timestamp_utc="2026-05-07T20:30:00Z",
        method_caveats=method_caveats,
    )


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_schema_version_constant_value(self) -> None:
        assert TEST_METRICS_CI_SCHEMA_VERSION == "1"
        assert isinstance(TEST_METRICS_CI_SCHEMA_VERSION, str)

    def test_top_level_imports_from_hft_contracts(self) -> None:
        """TestMetricsCIArtifact must be importable directly from
        ``hft_contracts`` (mirror FeatureImportanceArtifact pattern)."""
        import hft_contracts

        assert hasattr(hft_contracts, "TestMetricsCIArtifact")
        assert hasattr(hft_contracts, "MetricCIBound")
        assert hasattr(hft_contracts, "TEST_METRICS_CI_SCHEMA_VERSION")
        assert "TestMetricsCIArtifact" in hft_contracts.__all__
        assert "MetricCIBound" in hft_contracts.__all__
        assert "TEST_METRICS_CI_SCHEMA_VERSION" in hft_contracts.__all__


# ---------------------------------------------------------------------------
# Construction + round-trip
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_minimal_valid_construction(self) -> None:
        art = _make_artifact()
        assert art.schema_version == "1"
        assert art.method == "block_bootstrap"
        assert art.n_test_samples == 8085
        assert "test_ic" in art.metrics
        # Frozen dataclass — assignment must raise
        with pytest.raises(Exception):
            art.schema_version = "2"  # type: ignore[misc]

    def test_method_caveats_default_empty_tuple(self) -> None:
        art = _make_artifact(method_caveats=())
        assert art.method_caveats == ()

    def test_optional_fingerprint_fields_accept_none(self) -> None:
        art = _make_artifact(
            compatibility_fingerprint=None,
            model_config_hash=None,
            normalization_stats_sha256=None,
        )
        assert art.compatibility_fingerprint is None
        assert art.model_config_hash is None
        assert art.normalization_stats_sha256 is None


class TestRoundTrip:
    def test_to_dict_then_from_dict_preserves_artifact(self) -> None:
        art = _make_artifact()
        data = art.to_dict()
        art2 = TestMetricsCIArtifact.from_dict(data)
        assert art == art2

    def test_to_dict_metrics_sorted_for_canonical_stability(self) -> None:
        """metrics dict must be serialized sorted by key so canonical-JSON
        bypass produces stable content_hash regardless of insertion order."""
        # Insertion-order: directional_accuracy first, ic second
        metrics_unordered = {}
        metrics_unordered["test_directional_accuracy"] = _make_bound(0.6, 0.55, 0.65)
        metrics_unordered["test_ic"] = _make_bound(0.4, 0.35, 0.45)
        art = _make_artifact(metrics=metrics_unordered)
        data = art.to_dict()
        # Sorted output: ic before directional_accuracy alphabetically
        assert list(data["metrics"].keys()) == sorted(data["metrics"].keys())

    def test_from_dict_handles_legacy_method_caveats_as_list(self) -> None:
        """method_caveats stored as list in JSON must be coerced to tuple
        on load (frozen dataclass requires hashable type)."""
        art = _make_artifact(method_caveats=("ic_silent_sanitize",))
        data = art.to_dict()
        # Verify list serialization
        assert isinstance(data["method_caveats"], list)
        # Verify tuple round-trip
        art2 = TestMetricsCIArtifact.from_dict(data)
        assert art2.method_caveats == ("ic_silent_sanitize",)
        assert isinstance(art2.method_caveats, tuple)


# ---------------------------------------------------------------------------
# Content addressing (canonical_hash SSoT delegation)
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_content_hash_deterministic_across_calls(self) -> None:
        art = _make_artifact()
        assert art.content_hash() == art.content_hash()

    def test_content_hash_is_64_hex(self) -> None:
        art = _make_artifact()
        h = art.content_hash()
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_content_hash_mutation_sensitive_seed(self) -> None:
        a1 = _make_artifact()
        a2 = TestMetricsCIArtifact(
            **{**a1.to_dict(), "metrics": a1.metrics, "seed": 999}
        )
        assert a1.content_hash() != a2.content_hash()

    def test_content_hash_mutation_sensitive_metric_value(self) -> None:
        a1 = _make_artifact()
        new_metrics = {**a1.metrics, "test_ic": _make_bound(0.99, 0.95, 1.0)}
        a2 = _make_artifact(metrics=new_metrics)
        assert a1.content_hash() != a2.content_hash()

    def test_content_hash_insensitive_to_metric_insertion_order(self) -> None:
        """Same metrics in different insertion order must produce identical
        content_hash (sorted-keys discipline in to_dict)."""
        m1 = {}
        m1["test_ic"] = _make_bound(0.4, 0.35, 0.45)
        m1["test_directional_accuracy"] = _make_bound(0.6, 0.55, 0.65)
        m2 = {}
        m2["test_directional_accuracy"] = _make_bound(0.6, 0.55, 0.65)
        m2["test_ic"] = _make_bound(0.4, 0.35, 0.45)
        a1 = _make_artifact(metrics=m1)
        a2 = _make_artifact(metrics=m2)
        assert a1.content_hash() == a2.content_hash()


# ---------------------------------------------------------------------------
# Atomic save/load (atomic_io SSoT delegation)
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_then_load_round_trip(self, tmp_path: Path) -> None:
        art = _make_artifact()
        path = tmp_path / "test_metrics_ci_v1.json"
        art.save(path)
        art2 = TestMetricsCIArtifact.load(path)
        assert art == art2
        assert art.content_hash() == art2.content_hash()

    def test_save_uses_canonical_json_format(self, tmp_path: Path) -> None:
        """Atomic_write_json default has sort_keys=True + trailing newline.
        Verify the written file matches canonical convention."""
        art = _make_artifact()
        path = tmp_path / "test_metrics_ci_v1.json"
        art.save(path)
        text = path.read_text()
        # Trailing newline (atomic_io convention)
        assert text.endswith("\n")
        # Re-parse to verify well-formed JSON
        data = json.loads(text)
        assert data["schema_version"] == "1"

    def test_load_raises_keyerror_on_missing_required_field(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "broken.json"
        # Missing 'method' field — required scalar
        path.write_text(json.dumps({
            "schema_version": "1",
            "block_length": 21,
            "block_length_source": "x",
            "n_bootstraps": 100,
            "ci": 0.95,
            "seed": 42,
            "n_test_samples": 100,
            "metrics": {},
        }))
        with pytest.raises(KeyError):
            TestMetricsCIArtifact.load(path)


# ---------------------------------------------------------------------------
# Validation invariants (__post_init__ fail-loud)
# ---------------------------------------------------------------------------


class TestValidation:
    def test_n_test_samples_zero_raises(self) -> None:
        # Bypass _make_artifact's helper (which would fail at MetricCIBound
        # leaf-level validation first) — construct artifact directly with
        # n_test_samples=0 + a separately-built bound that does NOT share
        # the bad n_samples value (mismatch then triggers artifact-level
        # check). Exercises the artifact's own n_test_samples > 0 invariant.
        valid_bound = MetricCIBound(point=0.5, ci_low=0.4, ci_high=0.6, n_samples=10)
        with pytest.raises(ValueError, match="n_test_samples"):
            TestMetricsCIArtifact(
                schema_version="1",
                method="block_bootstrap",
                block_length=21,
                block_length_source="auto-derive ceil(n^(1/3))",
                n_bootstraps=10000,
                ci=0.95,
                seed=42,
                n_test_samples=0,
                metrics={"test_ic": valid_bound},
                compatibility_fingerprint="0" * 64,
                model_config_hash=None,
                normalization_stats_sha256=None,
                signal_export_output_dir=None,
                experiment_id="x",
                fingerprint="x",
                model_type="x",
                timestamp_utc="2026-05-07T20:30:00Z",
            )

    def test_n_test_samples_negative_raises(self) -> None:
        # Same pattern as test_n_test_samples_zero_raises — bypass helper.
        valid_bound = MetricCIBound(point=0.5, ci_low=0.4, ci_high=0.6, n_samples=10)
        with pytest.raises(ValueError, match="n_test_samples"):
            TestMetricsCIArtifact(
                schema_version="1",
                method="block_bootstrap",
                block_length=21,
                block_length_source="auto-derive ceil(n^(1/3))",
                n_bootstraps=10000,
                ci=0.95,
                seed=42,
                n_test_samples=-1,
                metrics={"test_ic": valid_bound},
                compatibility_fingerprint="0" * 64,
                model_config_hash=None,
                normalization_stats_sha256=None,
                signal_export_output_dir=None,
                experiment_id="x",
                fingerprint="x",
                model_type="x",
                timestamp_utc="2026-05-07T20:30:00Z",
            )

    def test_metric_ci_bound_n_samples_zero_raises(self) -> None:
        """MetricCIBound leaf-type validation (Round 1 §2 HIGH fix)."""
        with pytest.raises(ValueError, match="n_samples=0 must be > 0"):
            MetricCIBound(point=0.5, ci_low=0.4, ci_high=0.6, n_samples=0)

    def test_metric_ci_bound_nan_point_raises(self) -> None:
        """MetricCIBound leaf-type validation: finite check."""
        with pytest.raises(ValueError, match="point.*not finite"):
            MetricCIBound(
                point=float("nan"), ci_low=0.4, ci_high=0.6, n_samples=10
            )

    def test_metric_ci_bound_inf_ci_low_raises(self) -> None:
        with pytest.raises(ValueError, match="ci_low.*not finite"):
            MetricCIBound(
                point=0.5, ci_low=float("-inf"), ci_high=0.6, n_samples=10
            )

    def test_n_bootstraps_below_100_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bootstraps"):
            _make_artifact(n_bootstraps=99)

    def test_block_length_below_2_raises(self) -> None:
        with pytest.raises(ValueError, match="block_length"):
            _make_artifact(block_length=1)

    @pytest.mark.parametrize("ci", [0.0, 1.0, 1.5, -0.1])
    def test_ci_out_of_range_raises(self, ci: float) -> None:
        with pytest.raises(ValueError, match="ci="):
            _make_artifact(ci=ci)

    def test_empty_metrics_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="metrics dict must not be empty"):
            _make_artifact(metrics={})

    def test_metric_value_not_metric_ci_bound_raises(self) -> None:
        with pytest.raises(TypeError, match="MetricCIBound"):
            _make_artifact(metrics={"test_ic": (0.4, 0.35, 0.45, 8085)})  # type: ignore[dict-item]

    def test_point_above_ci_high_raises_at_metric_ci_bound_layer(self) -> None:
        """Round 1 §2 HIGH fix: MetricCIBound itself catches the invariant
        BEFORE the artifact-level check (better diagnostic)."""
        with pytest.raises(
            ValueError, match="ci_low <= point <= ci_high"
        ):
            MetricCIBound(
                point=0.5, ci_low=0.3, ci_high=0.4, n_samples=10
            )

    def test_point_below_ci_low_raises_at_metric_ci_bound_layer(self) -> None:
        with pytest.raises(
            ValueError, match="ci_low <= point <= ci_high"
        ):
            MetricCIBound(
                point=0.1, ci_low=0.3, ci_high=0.4, n_samples=10
            )

    def test_n_samples_mismatch_across_metrics_raises(self) -> None:
        bad_metrics = {
            "test_ic": _make_bound(0.4, 0.35, 0.45, n_samples=8085),
            "test_da": _make_bound(0.6, 0.55, 0.65, n_samples=7000),
        }
        with pytest.raises(ValueError, match="cross-metric n must agree"):
            _make_artifact(metrics=bad_metrics, n_test_samples=8085)

    @pytest.mark.parametrize(
        "bad_fp",
        [
            "abc",  # too short
            "0" * 63,  # 63 chars
            "0" * 65,  # 65 chars
            "ABCDEF" + "0" * 58,  # uppercase
            "g" * 64,  # non-hex char
        ],
    )
    def test_compatibility_fingerprint_format_validation(
        self, bad_fp: str
    ) -> None:
        with pytest.raises(ValueError, match="compatibility_fingerprint"):
            _make_artifact(compatibility_fingerprint=bad_fp)

    def test_valid_64_hex_fingerprint_accepted(self) -> None:
        valid_fp = "67c8ff36949d6809" + "0" * 48
        assert len(valid_fp) == 64
        art = _make_artifact(compatibility_fingerprint=valid_fp)
        assert art.compatibility_fingerprint == valid_fp


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


class TestGetMetric:
    def test_get_metric_returns_bound_for_present_key(self) -> None:
        art = _make_artifact()
        bound = art.get_metric("test_ic")
        assert bound is not None
        assert bound.point == pytest.approx(0.37466)

    def test_get_metric_returns_none_for_missing_key(self) -> None:
        art = _make_artifact()
        assert art.get_metric("nonexistent_metric") is None
