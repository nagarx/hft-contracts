"""Tests for hft_contracts.feature_sets (Phase 6 6B.3 co-move).

Focused contract-plane tests — the full producer/writer/registry suite
stays in hft-ops. This file exercises the hashed schema (schema.py +
hashing.py), which is the contract surface consumed by every module
via the content_hash.
"""

from __future__ import annotations

import importlib.util

import pytest

from hft_contracts.feature_sets import (
    FEATURE_SET_SCHEMA_VERSION,
    FeatureSet,
    FeatureSetAppliesTo,
    FeatureSetIntegrityError,
    FeatureSetProducedBy,
    FeatureSetRef,
    FeatureSetValidationError,
    compute_feature_set_hash,
    validate_feature_set_dict,
)


def _base_produced_by() -> FeatureSetProducedBy:
    return FeatureSetProducedBy(
        tool="hft-feature-evaluator",
        tool_version="0.1.0",
        config_path="configs/test.yaml",
        config_hash="a" * 64,
        source_profile_hash="b" * 64,
        data_export="data/exports/test",
        data_dir_hash="c" * 64,
    )


class TestComputeFeatureSetHash:
    def test_deterministic(self):
        h1 = compute_feature_set_hash([0, 5, 12], 98, "2.2")
        h2 = compute_feature_set_hash([0, 5, 12], 98, "2.2")
        assert h1 == h2
        assert len(h1) == 64
        assert all(c in "0123456789abcdef" for c in h1)

    def test_order_and_duplicate_invariant(self):
        h1 = compute_feature_set_hash([12, 0, 5, 5], 98, "2.2")
        h2 = compute_feature_set_hash([0, 5, 12], 98, "2.2")
        assert h1 == h2

    def test_different_indices_different_hash(self):
        assert compute_feature_set_hash([0, 5], 98, "2.2") != compute_feature_set_hash(
            [0, 5, 12], 98, "2.2"
        )

    def test_different_source_width_different_hash(self):
        assert compute_feature_set_hash([0, 5], 98, "2.2") != compute_feature_set_hash(
            [0, 5], 128, "2.2"
        )

    def test_different_contract_version_different_hash(self):
        assert compute_feature_set_hash([0, 5], 98, "2.2") != compute_feature_set_hash(
            [0, 5], 98, "2.3"
        )

    def test_empty_indices_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_feature_set_hash([], 98, "2.2")

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            compute_feature_set_hash([-1, 0], 98, "2.2")

    def test_nonpositive_source_width_raises(self):
        with pytest.raises(ValueError, match="positive"):
            compute_feature_set_hash([0], 0, "2.2")


class TestFeatureSetBuild:
    def test_build_auto_computes_hash(self):
        fs = FeatureSet.build(
            name="test_v1",
            feature_indices=[0, 5, 12],
            feature_names=["ask_price_L1", "ask_price_L6", "bid_price_L3"],
            source_feature_count=98,
            contract_version="2.2",
            applies_to=FeatureSetAppliesTo(assets=("NVDA",), horizons=(60,)),
            produced_by=_base_produced_by(),
            criteria={},
            criteria_schema_version="1.0",
            created_at="2026-04-17T00:00:00Z",
        )
        assert fs.content_hash == compute_feature_set_hash([0, 5, 12], 98, "2.2")
        assert fs.feature_indices == (0, 5, 12)  # sorted-unique tuple
        assert fs.schema_version == FEATURE_SET_SCHEMA_VERSION

    def test_verify_integrity_passes_on_built_fs(self):
        fs = FeatureSet.build(
            name="test_v1",
            feature_indices=[0, 5, 12],
            feature_names=["a", "b", "c"],
            source_feature_count=98,
            contract_version="2.2",
            applies_to=FeatureSetAppliesTo(assets=("NVDA",), horizons=(60,)),
            produced_by=_base_produced_by(),
            criteria={},
            criteria_schema_version="1.0",
            created_at="2026-04-17T00:00:00Z",
        )
        fs.verify_integrity()  # must not raise

    def test_verify_integrity_detects_tamper(self):
        """Directly constructing with wrong content_hash → integrity fail."""
        fs = FeatureSet(
            schema_version=FEATURE_SET_SCHEMA_VERSION,
            name="tampered",
            content_hash="d" * 64,  # wrong
            contract_version="2.2",
            source_feature_count=98,
            applies_to=FeatureSetAppliesTo(assets=("NVDA",), horizons=(60,)),
            feature_indices=(0, 5),
            feature_names=("a", "b"),
            produced_by=_base_produced_by(),
            criteria={},
            criteria_schema_version="1.0",
            description="",
            notes="",
            created_at="2026-04-17T00:00:00Z",
            created_by="",
        )
        with pytest.raises(FeatureSetIntegrityError, match="integrity check failed"):
            fs.verify_integrity()

    def test_ref_returns_pointer(self):
        fs = FeatureSet.build(
            name="test_v1",
            feature_indices=[0, 5, 12],
            feature_names=["a", "b", "c"],
            source_feature_count=98,
            contract_version="2.2",
            applies_to=FeatureSetAppliesTo(assets=("NVDA",), horizons=(60,)),
            produced_by=_base_produced_by(),
            criteria={},
            criteria_schema_version="1.0",
            created_at="2026-04-17T00:00:00Z",
        )
        ref = fs.ref()
        assert isinstance(ref, FeatureSetRef)
        assert ref.name == "test_v1"
        assert ref.content_hash == fs.content_hash


class TestRoundtripJson:
    def test_to_dict_from_dict_roundtrip(self):
        fs1 = FeatureSet.build(
            name="test_v1",
            feature_indices=[0, 5, 12],
            feature_names=["a", "b", "c"],
            source_feature_count=98,
            contract_version="2.2",
            applies_to=FeatureSetAppliesTo(assets=("NVDA",), horizons=(60,)),
            produced_by=_base_produced_by(),
            criteria={"mode": "ic_screening", "min_abs_ic": 0.05},
            criteria_schema_version="1.0",
            created_at="2026-04-17T00:00:00Z",
        )
        fs2 = FeatureSet.from_dict(fs1.to_dict())
        assert fs2.name == fs1.name
        assert fs2.content_hash == fs1.content_hash
        assert fs2.feature_indices == fs1.feature_indices
        assert fs2.criteria == fs1.criteria


class TestValidateFeatureSetDict:
    def _valid_dict(self) -> dict:
        return {
            "schema_version": FEATURE_SET_SCHEMA_VERSION,
            "name": "test_v1",
            "content_hash": compute_feature_set_hash([0, 5], 98, "2.2"),
            "contract_version": "2.2",
            "source_feature_count": 98,
            "applies_to": {"assets": ["NVDA"], "horizons": [60]},
            "feature_indices": [0, 5],
            "feature_names": ["a", "b"],
            "produced_by": {
                "tool": "hft-feature-evaluator",
                "tool_version": "0.1.0",
                "config_path": "x",
                "config_hash": "a" * 64,
                "source_profile_hash": "b" * 64,
                "data_export": "x",
                "data_dir_hash": "c" * 64,
            },
            "criteria": {},
            "criteria_schema_version": "1.0",
            "description": "",
            "notes": "",
            "created_at": "2026-04-17T00:00:00Z",
            "created_by": "",
        }

    def test_valid_passes(self):
        validate_feature_set_dict(self._valid_dict())

    def test_missing_required_key_raises(self):
        d = self._valid_dict()
        del d["name"]
        with pytest.raises(FeatureSetValidationError, match="missing required keys"):
            validate_feature_set_dict(d)

    def test_invalid_content_hash_raises(self):
        d = self._valid_dict()
        d["content_hash"] = "invalid"
        with pytest.raises(FeatureSetValidationError, match="content_hash must be"):
            validate_feature_set_dict(d)

    def test_index_beyond_source_width_raises(self):
        d = self._valid_dict()
        d["feature_indices"] = [0, 200]  # 200 > 98
        d["content_hash"] = compute_feature_set_hash([0, 200], 98, "2.2")
        with pytest.raises(FeatureSetValidationError, match="must be <"):
            validate_feature_set_dict(d)

    def test_duplicate_index_raises(self):
        d = self._valid_dict()
        d["feature_indices"] = [0, 0, 5]
        with pytest.raises(FeatureSetValidationError, match="unique"):
            validate_feature_set_dict(d)


@pytest.mark.skipif(
    importlib.util.find_spec("hft_ops") is None,
    reason=(
        "hft_ops not installed — shim-parity regression guard skipped on "
        "fresh-clone installs of hft-contracts. Runs in authoring env."
    ),
)
class TestHftOpsShimCompat:
    """Back-compat: pre-6B.3 imports through hft_ops.feature_sets must
    still work, returning the SAME classes (not a copy).

    REV 2 pre-push (2026-04-20): class-level skip marker so fresh-clone
    users (who install only hft-contracts) get `pytest -q` → all tests
    SKIP gracefully rather than ERROR with ModuleNotFoundError. Running
    `pip install hft-ops` re-enables these guards automatically.
    """

    def test_hft_ops_feature_sets_reexport_is_identical_class(self):
        from hft_ops.feature_sets.schema import FeatureSet as FromHftOps
        from hft_ops.feature_sets import FeatureSet as FromHftOpsTop

        assert FromHftOps is FeatureSet
        assert FromHftOpsTop is FeatureSet

    def test_hft_ops_hashing_reexport_is_identical_function(self):
        from hft_ops.feature_sets.hashing import (
            compute_feature_set_hash as fn_from_hft_ops,
        )
        assert fn_from_hft_ops is compute_feature_set_hash
