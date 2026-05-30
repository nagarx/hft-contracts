"""Tests for hft_contracts.pairwise_compare_artifact (Phase 2 P2.C — 2026-05-07).

Mirrors test_test_metrics_ci_artifact.py structure (Phase 2 P2.A precedent).
Locks PairwiseCompareArtifact + PairwiseResultRecord contracts for K-way
pairwise comparison via paired moving-block bootstrap.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hft_contracts.pairwise_compare_artifact import (
    PAIRWISE_COMPARE_SCHEMA_VERSION,
    PairwiseCompareArtifact,
    PairwiseResultRecord,
)


def _make_record(
    *,
    treatment_a_idx: int = 0,
    treatment_b_idx: int = 1,
    treatment_a_label: str = "R9",
    treatment_b_label: str = "R10",
    statistic_a: float = 0.375,
    statistic_b: float = 0.346,
    delta: float = 0.029,
    delta_ci_low: float = 0.005,
    delta_ci_high: float = 0.052,
    p_value_raw: float = 0.022,
    p_value_bh: float = 0.066,
    n_nonfinite_replaced: int = 0,
) -> PairwiseResultRecord:
    return PairwiseResultRecord(
        treatment_a_idx=treatment_a_idx,
        treatment_b_idx=treatment_b_idx,
        treatment_a_label=treatment_a_label,
        treatment_b_label=treatment_b_label,
        statistic_a=statistic_a,
        statistic_b=statistic_b,
        delta=delta,
        delta_ci_low=delta_ci_low,
        delta_ci_high=delta_ci_high,
        p_value_raw=p_value_raw,
        p_value_bh=p_value_bh,
        n_nonfinite_replaced=n_nonfinite_replaced,
    )


def _make_k3_artifact(
    *,
    n_treatments: int = 3,
    parent_compatibility_fingerprints: tuple = ("a" * 64,) * 3,
    parent_model_config_hashes: tuple = ("b" * 64, "c" * 64, "d" * 64),
    treatment_labels: tuple = ("R9", "R10", "R11"),
    parent_experiment_ids: tuple = ("R9", "R10", "R11"),
    pairs: tuple | None = None,
    n_samples_paired: int = 8085,
    n_samples_raw: int = 8085,
    n_dropped_nonfinite: int = 0,
    drop_fraction: float = 0.0,
    paired_compat_fingerprint: str = "a" * 64,
    paired_labels_sha256: str = "1" * 64,
) -> PairwiseCompareArtifact:
    if pairs is None:
        pairs = (
            _make_record(
                treatment_a_idx=0, treatment_b_idx=1,
                treatment_a_label="R9", treatment_b_label="R10",
            ),
            _make_record(
                treatment_a_idx=0, treatment_b_idx=2,
                treatment_a_label="R9", treatment_b_label="R11",
                statistic_b=-0.005, delta=0.380,
                delta_ci_low=0.350, delta_ci_high=0.410,
                p_value_raw=0.0001, p_value_bh=0.0003,
            ),
            _make_record(
                treatment_a_idx=1, treatment_b_idx=2,
                treatment_a_label="R10", treatment_b_label="R11",
                statistic_a=0.346, statistic_b=-0.005, delta=0.351,
                delta_ci_low=0.320, delta_ci_high=0.380,
                p_value_raw=0.0001, p_value_bh=0.0003,
            ),
        )
    return PairwiseCompareArtifact(
        schema_version="1",
        method="paired_block_bootstrap",
        metric_name="spearman_ic",
        block_length=21,
        block_length_source="auto-derive ceil(n^(1/3))",
        n_bootstraps=10000,
        alpha=0.05,
        seed=42,
        n_treatments=n_treatments,
        n_samples_paired=n_samples_paired,
        n_samples_raw=n_samples_raw,
        n_dropped_nonfinite=n_dropped_nonfinite,
        drop_fraction=drop_fraction,
        primary_horizon_idx=0,
        parent_experiment_ids=parent_experiment_ids,
        parent_compatibility_fingerprints=parent_compatibility_fingerprints,
        parent_model_config_hashes=parent_model_config_hashes,
        paired_compat_fingerprint=paired_compat_fingerprint,
        paired_labels_sha256=paired_labels_sha256,
        pairs=pairs,
        treatment_labels=treatment_labels,
        timestamp_utc="2026-05-07T20:30:00Z",
    )


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_schema_version_constant(self) -> None:
        assert PAIRWISE_COMPARE_SCHEMA_VERSION == "1"

    def test_top_level_imports_from_hft_contracts(self) -> None:
        import hft_contracts
        assert hasattr(hft_contracts, "PairwiseCompareArtifact")
        assert hasattr(hft_contracts, "PairwiseResultRecord")
        assert hasattr(hft_contracts, "PAIRWISE_COMPARE_SCHEMA_VERSION")
        assert "PairwiseCompareArtifact" in hft_contracts.__all__
        assert "PairwiseResultRecord" in hft_contracts.__all__
        assert "PAIRWISE_COMPARE_SCHEMA_VERSION" in hft_contracts.__all__


# ---------------------------------------------------------------------------
# PairwiseResultRecord leaf validation
# ---------------------------------------------------------------------------


class TestPairwiseResultRecordValidation:
    def test_valid_record_constructs_ok(self) -> None:
        r = _make_record()
        assert r.delta == pytest.approx(0.029)
        assert r.treatment_a_idx == 0
        assert r.treatment_b_idx == 1

    def test_a_idx_equals_b_idx_raises(self) -> None:
        with pytest.raises(ValueError, match="treatment_a_idx.*<"):
            _make_record(treatment_a_idx=2, treatment_b_idx=2)

    def test_a_idx_greater_than_b_idx_raises(self) -> None:
        with pytest.raises(ValueError, match="treatment_a_idx.*<"):
            _make_record(treatment_a_idx=3, treatment_b_idx=1)

    def test_negative_a_idx_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            _make_record(treatment_a_idx=-1, treatment_b_idx=0)

    def test_nan_statistic_raises(self) -> None:
        with pytest.raises(ValueError, match="not finite"):
            _make_record(statistic_a=float("nan"))

    def test_inf_delta_raises(self) -> None:
        with pytest.raises(ValueError, match="not finite"):
            _make_record(delta=float("inf"))

    def test_delta_outside_ci_raises(self) -> None:
        with pytest.raises(ValueError, match="delta_ci_low <= delta <= delta_ci_high"):
            _make_record(delta=0.1, delta_ci_low=0.5, delta_ci_high=0.6)

    def test_p_value_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="p_value_raw"):
            _make_record(p_value_raw=1.5)

    def test_p_value_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="p_value_bh"):
            _make_record(p_value_bh=-0.1)

    def test_negative_n_nonfinite_replaced_raises(self) -> None:
        with pytest.raises(ValueError, match="n_nonfinite_replaced"):
            _make_record(n_nonfinite_replaced=-1)

    def test_empty_label_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _make_record(treatment_a_label="")


# ---------------------------------------------------------------------------
# Construction + Round-trip
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_k3_artifact_constructs_ok(self) -> None:
        art = _make_k3_artifact()
        assert art.n_treatments == 3
        assert len(art.pairs) == 3  # K*(K-1)/2 = 3
        assert art.method == "paired_block_bootstrap"

    def test_k2_artifact_constructs_ok(self) -> None:
        art = _make_k3_artifact(
            n_treatments=2,
            parent_experiment_ids=("R9", "R10"),
            parent_compatibility_fingerprints=("a" * 64,) * 2,
            parent_model_config_hashes=("b" * 64, "c" * 64),
            treatment_labels=("R9", "R10"),
            pairs=(_make_record(),),
        )
        assert art.n_treatments == 2
        assert len(art.pairs) == 1


class TestRoundTrip:
    def test_to_dict_then_from_dict_preserves_artifact(self) -> None:
        art = _make_k3_artifact()
        data = art.to_dict()
        art2 = PairwiseCompareArtifact.from_dict(data)
        assert art == art2

    def test_optional_model_config_hashes_none_preserved(self) -> None:
        """sklearn pre-Phase-Q.6.5 may have None model_config_hash."""
        art = _make_k3_artifact(
            parent_model_config_hashes=("b" * 64, None, "d" * 64),
        )
        data = art.to_dict()
        art2 = PairwiseCompareArtifact.from_dict(data)
        assert art2.parent_model_config_hashes[1] is None
        assert art == art2


# ---------------------------------------------------------------------------
# Content addressing
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_content_hash_deterministic(self) -> None:
        art = _make_k3_artifact()
        assert art.content_hash() == art.content_hash()

    def test_content_hash_64_hex(self) -> None:
        art = _make_k3_artifact()
        h = art.content_hash()
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_content_hash_mutation_sensitive(self) -> None:
        a1 = _make_k3_artifact()
        a2 = _make_k3_artifact(treatment_labels=("R9", "R10", "R11_v2"))
        assert a1.content_hash() != a2.content_hash()


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        art = _make_k3_artifact()
        path = tmp_path / "pairwise_compare_v1.json"
        art.save(path)
        art2 = PairwiseCompareArtifact.load(path)
        assert art == art2
        assert art.content_hash() == art2.content_hash()

    def test_save_uses_atomic_write(self, tmp_path: Path) -> None:
        art = _make_k3_artifact()
        path = tmp_path / "out.json"
        art.save(path)
        text = path.read_text()
        assert text.endswith("\n")  # atomic_write_json convention
        assert json.loads(text)["schema_version"] == "1"


# ---------------------------------------------------------------------------
# Artifact-level invariant validations
# ---------------------------------------------------------------------------


class TestArtifactValidation:
    def test_n_treatments_below_2_raises(self) -> None:
        with pytest.raises(ValueError, match="n_treatments"):
            _make_k3_artifact(
                n_treatments=1,
                parent_experiment_ids=("R9",),
                parent_compatibility_fingerprints=("a" * 64,),
                parent_model_config_hashes=("b" * 64,),
                treatment_labels=("R9",),
                pairs=(),
            )

    def test_n_pairs_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="K\\*\\(K-1\\)/2"):
            # K=3 should have 3 pairs; supply 2
            _make_k3_artifact(
                pairs=(
                    _make_record(treatment_a_idx=0, treatment_b_idx=1),
                    _make_record(treatment_a_idx=0, treatment_b_idx=2,
                                 treatment_a_label="R9", treatment_b_label="R11"),
                ),
            )

    def test_parent_experiment_ids_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="parent_experiment_ids"):
            _make_k3_artifact(parent_experiment_ids=("R9", "R10"))  # 2 != K=3

    def test_alpha_at_zero_raises(self) -> None:
        art = _make_k3_artifact()
        with pytest.raises(ValueError, match="alpha"):
            PairwiseCompareArtifact(
                **{**art.to_dict(),
                   "alpha": 0.0,
                   "pairs": art.pairs,
                   "parent_experiment_ids": art.parent_experiment_ids,
                   "parent_compatibility_fingerprints": art.parent_compatibility_fingerprints,
                   "parent_model_config_hashes": art.parent_model_config_hashes,
                   "treatment_labels": art.treatment_labels,
                   "method_caveats": art.method_caveats}
            )

    def test_n_bootstraps_below_100_raises(self) -> None:
        art = _make_k3_artifact()
        with pytest.raises(ValueError, match="n_bootstraps"):
            PairwiseCompareArtifact(
                **{**art.to_dict(),
                   "n_bootstraps": 50,
                   "pairs": art.pairs,
                   "parent_experiment_ids": art.parent_experiment_ids,
                   "parent_compatibility_fingerprints": art.parent_compatibility_fingerprints,
                   "parent_model_config_hashes": art.parent_model_config_hashes,
                   "treatment_labels": art.treatment_labels,
                   "method_caveats": art.method_caveats}
            )

    def test_block_length_below_2_raises(self) -> None:
        art = _make_k3_artifact()
        with pytest.raises(ValueError, match="block_length"):
            PairwiseCompareArtifact(
                **{**art.to_dict(),
                   "block_length": 1,
                   "pairs": art.pairs,
                   "parent_experiment_ids": art.parent_experiment_ids,
                   "parent_compatibility_fingerprints": art.parent_compatibility_fingerprints,
                   "parent_model_config_hashes": art.parent_model_config_hashes,
                   "treatment_labels": art.treatment_labels,
                   "method_caveats": art.method_caveats}
            )

    def test_compat_fps_not_all_equal_raises(self) -> None:
        with pytest.raises(ValueError, match="must be identical"):
            _make_k3_artifact(
                parent_compatibility_fingerprints=("a" * 64, "b" * 64, "a" * 64),
            )

    def test_paired_compat_fp_mismatch_with_parent_raises(self) -> None:
        with pytest.raises(ValueError, match="paired_compat_fingerprint"):
            _make_k3_artifact(
                # All parent compat fps = "a"*64
                paired_compat_fingerprint="b" * 64,  # mismatch
            )

    def test_invalid_compat_fp_format_raises(self) -> None:
        with pytest.raises(ValueError, match="compat"):
            _make_k3_artifact(
                parent_compatibility_fingerprints=("invalid",) * 3,
                paired_compat_fingerprint="invalid",
            )

    def test_uppercase_hex_rejected(self) -> None:
        with pytest.raises(ValueError, match="hex"):
            _make_k3_artifact(
                paired_compat_fingerprint="A" * 64,
                parent_compatibility_fingerprints=("A" * 64,) * 3,
            )

    def test_drop_fraction_inconsistent_raises(self) -> None:
        with pytest.raises(ValueError, match="drop_fraction"):
            _make_k3_artifact(
                n_samples_paired=8000,
                n_samples_raw=8085,
                n_dropped_nonfinite=85,
                drop_fraction=0.5,  # actual = 85/8085 ≈ 0.0105
            )

    def test_n_dropped_plus_paired_neq_raw_raises(self) -> None:
        with pytest.raises(ValueError, match="!="):
            _make_k3_artifact(
                n_samples_paired=8000,
                n_samples_raw=8085,
                n_dropped_nonfinite=50,  # 50 + 8000 = 8050, not 8085
                drop_fraction=50/8085,
            )

    def test_negative_primary_horizon_idx_raises(self) -> None:
        art = _make_k3_artifact()
        with pytest.raises(ValueError, match="primary_horizon_idx"):
            PairwiseCompareArtifact(
                **{**art.to_dict(),
                   "primary_horizon_idx": -1,
                   "pairs": art.pairs,
                   "parent_experiment_ids": art.parent_experiment_ids,
                   "parent_compatibility_fingerprints": art.parent_compatibility_fingerprints,
                   "parent_model_config_hashes": art.parent_model_config_hashes,
                   "treatment_labels": art.treatment_labels,
                   "method_caveats": art.method_caveats}
            )

    def test_empty_treatment_label_raises(self) -> None:
        with pytest.raises(ValueError, match="treatment_labels"):
            _make_k3_artifact(treatment_labels=("R9", "", "R11"))


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


class TestLookupHelpers:
    def test_get_pair_by_indices(self) -> None:
        art = _make_k3_artifact()
        p = art.get_pair(0, 2)
        assert p is not None
        assert p.delta == pytest.approx(0.380)

    def test_get_pair_canonical_ordering(self) -> None:
        art = _make_k3_artifact()
        # Reverse order returns None — caller must canonicalize
        assert art.get_pair(2, 0) is None

    def test_get_pair_missing_returns_none(self) -> None:
        art = _make_k3_artifact()
        assert art.get_pair(99, 100) is None

    def test_get_pair_by_labels_forward_order(self) -> None:
        art = _make_k3_artifact()
        p = art.get_pair_by_labels("R9", "R10")
        assert p is not None
        assert p.delta == pytest.approx(0.029)

    def test_get_pair_by_labels_reverse_order(self) -> None:
        """Symmetric label search — both orderings find the pair."""
        art = _make_k3_artifact()
        p = art.get_pair_by_labels("R10", "R9")
        assert p is not None  # symmetric


# ---------------------------------------------------------------------------
# from_hft_metrics_result conversion
# ---------------------------------------------------------------------------


class TestFromHftMetricsResult:
    def test_converts_pairwise_result_with_labels(self) -> None:
        from hft_metrics.pairwise import PairwiseResult
        # Construct synthetic PairwiseResult (mirrors hft-metrics primitive's output)
        hm_result = PairwiseResult(
            i=0, j=1,
            statistic_i=0.375, statistic_j=0.346,
            delta=0.029,
            ci_lower=0.005, ci_upper=0.052,
            p_value_raw=0.022, p_value_bh=0.066,
            n_bootstraps=10000, block_length=21, seed=42,
            n_nonfinite_replaced=2,
        )
        labels = ("R9_TLOB_no_CVML", "R10_TLOB_CVML", "R11_GMADL")
        record = PairwiseResultRecord.from_hft_metrics_result(hm_result, labels)
        assert record.treatment_a_idx == 0
        assert record.treatment_b_idx == 1
        assert record.treatment_a_label == "R9_TLOB_no_CVML"
        assert record.treatment_b_label == "R10_TLOB_CVML"
        assert record.delta == pytest.approx(0.029)
        assert record.n_nonfinite_replaced == 2


class TestFromDictNullHardening:
    """Audit finding #1 (2026-05-28): from_dict must tolerate present-but-null
    collection fields (the H-2 bug class). pairs:null and method_caveats:null
    would crash with TypeError before the `or [] / or ()` coercion fix."""

    def test_method_caveats_null_coerced(self):
        d = _make_k3_artifact().to_dict()
        d["method_caveats"] = None
        art = PairwiseCompareArtifact.from_dict(d)
        assert art.method_caveats == ()

    def test_pairs_null_coerced_then_count_check_raises_clean(self):
        # pairs=None coerces to [] → __post_init__ raises a CLEAN ValueError
        # (K*(K-1)/2 count mismatch), not a cryptic `for p in None` TypeError.
        d = _make_k3_artifact().to_dict()
        d["pairs"] = None
        with pytest.raises(ValueError):
            PairwiseCompareArtifact.from_dict(d)

    @pytest.mark.parametrize(
        "field",
        [
            "parent_experiment_ids",
            "parent_compatibility_fingerprints",
            "parent_model_config_hashes",
            "treatment_labels",
        ],
    )
    def test_required_collection_null_raises_clean_valueerror(self, field):
        # Audit F1 (2026-05-30 re-validation): the 4 REQUIRED K-length collection
        # fields use bracket-access in from_dict. A present-but-null value would
        # crash with a cryptic `tuple(None)` / `for v in None` TypeError. The
        # `or ()` coercion turns null into () so the existing __post_init__
        # `len(field) != n_treatments` invariant raises a CLEAN ValueError that
        # NAMES the offending field. Round-2 hardened only the `.get()`-default
        # collection sites (pairs/method_caveats) and left these four required
        # bracket-access fields — they were the remaining null-collection
        # family members.
        d = _make_k3_artifact().to_dict()
        d[field] = None
        with pytest.raises(ValueError, match=field):
            PairwiseCompareArtifact.from_dict(d)
