"""
Self-consistency tests for the HFT pipeline contract.

These tests verify that the generated constants are internally consistent:
enum member counts match declared counts, slices are contiguous, categorical
indices are valid, and label contracts are well-formed.

Run: pytest hft-contracts/tests/ -v
"""

import pytest

from hft_contracts import (
    # Enums
    FeatureIndex,
    ExperimentalFeatureIndex,
    SignalIndex,
    # Counts
    LOB_LEVELS,
    LOB_FEATURE_COUNT,
    DERIVED_FEATURE_COUNT,
    MBO_FEATURE_COUNT,
    SIGNAL_FEATURE_COUNT,
    EXPERIMENTAL_INSTITUTIONAL_V2_COUNT,
    EXPERIMENTAL_VOLATILITY_COUNT,
    EXPERIMENTAL_SEASONALITY_COUNT,
    EXPERIMENTAL_MLOFI_COUNT,
    EXPERIMENTAL_KOLM_OF_COUNT,
    EXPERIMENTAL_FEATURE_COUNT,
    FEATURE_COUNT,
    STANDARD_FEATURE_COUNT,
    EXTENDED_FEATURE_COUNT,
    FULL_FEATURE_COUNT,
    FEATURE_COUNT_WITH_EXPERIMENTAL,
    # Slices
    LOB_ASK_PRICES,
    LOB_ASK_SIZES,
    LOB_BID_PRICES,
    LOB_BID_SIZES,
    LOB_ALL,
    DERIVED_ALL,
    MBO_ALL,
    SIGNALS_ALL,
    EXPERIMENTAL_ALL,
    EXPERIMENTAL_INSTITUTIONAL_V2_SLICE,
    EXPERIMENTAL_VOLATILITY_SLICE,
    EXPERIMENTAL_SEASONALITY_SLICE,
    EXPERIMENTAL_MLOFI_SLICE,
    EXPERIMENTAL_KOLM_OF_SLICE,
    # Layout
    GROUPED_PRICE_INDICES,
    GROUPED_SIZE_INDICES,
    LOBSTER_PRICE_INDICES,
    LOBSTER_SIZE_INDICES,
    # Classification
    CATEGORICAL_INDICES,
    UNSIGNED_FEATURES,
    SAFETY_GATES,
    PRIMARY_SIGNALS,
    ASYMMETRY_SIGNALS,
    SIGNAL_NAMES,
    EXPERIMENTAL_FEATURE_NAMES,
    # Labels
    LABEL_DOWN,
    LABEL_STABLE,
    LABEL_UP,
    NUM_CLASSES,
    TLOB_CONTRACT,
    TB_CONTRACT,
    OPPORTUNITY_CONTRACT,
    REGRESSION_CONTRACT,
    RegressionLabelContract,
    LabelingStrategy,
    get_contract,
    get_label_name,
    is_regression_strategy,
    # Validation
    validate_feature_indices,
    ContractError,
    SCHEMA_VERSION,
    # Off-Exchange
    OffExchangeFeatureIndex,
    OFF_EXCHANGE_FEATURE_COUNT,
    OFF_EXCHANGE_ACTIVE_FEATURE_COUNT,
    OFF_EXCHANGE_SCHEMA_VERSION,
    OFF_EXCHANGE_SIGNED_FLOW,
    OFF_EXCHANGE_VENUE_METRICS,
    OFF_EXCHANGE_RETAIL_METRICS,
    OFF_EXCHANGE_BBO_DYNAMICS,
    OFF_EXCHANGE_VPIN,
    OFF_EXCHANGE_TRADE_SIZE,
    OFF_EXCHANGE_CROSS_VENUE,
    OFF_EXCHANGE_ACTIVITY,
    OFF_EXCHANGE_SAFETY_GATES_SLICE,
    OFF_EXCHANGE_CONTEXT,
    OFF_EXCHANGE_CATEGORICAL_INDICES,
    OFF_EXCHANGE_NON_NORMALIZABLE_INDICES,
    OFF_EXCHANGE_UNSIGNED_FEATURES,
    OFF_EXCHANGE_SAFETY_GATES,
    OFF_EXCHANGE_FEATURE_NAMES,
)


class TestFeatureEnumCounts:
    """Verify enum member counts match declared constants."""

    def test_feature_index_count(self):
        assert len(FeatureIndex) == FEATURE_COUNT, (
            f"FeatureIndex has {len(FeatureIndex)} members but "
            f"FEATURE_COUNT is {FEATURE_COUNT}"
        )

    def test_experimental_index_count(self):
        assert len(ExperimentalFeatureIndex) == EXPERIMENTAL_FEATURE_COUNT, (
            f"ExperimentalFeatureIndex has {len(ExperimentalFeatureIndex)} members "
            f"but EXPERIMENTAL_FEATURE_COUNT is {EXPERIMENTAL_FEATURE_COUNT}"
        )

    def test_signal_index_count(self):
        assert len(SignalIndex) == SIGNAL_FEATURE_COUNT, (
            f"SignalIndex has {len(SignalIndex)} members but "
            f"SIGNAL_FEATURE_COUNT is {SIGNAL_FEATURE_COUNT}"
        )

    def test_no_duplicate_values_in_feature_index(self):
        values = [m.value for m in FeatureIndex]
        assert len(values) == len(set(values)), "FeatureIndex has duplicate values"

    def test_no_duplicate_values_in_experimental_index(self):
        values = [m.value for m in ExperimentalFeatureIndex]
        assert len(values) == len(set(values)), "ExperimentalFeatureIndex has duplicate values"

    def test_no_duplicate_values_in_signal_index(self):
        values = [m.value for m in SignalIndex]
        assert len(values) == len(set(values)), "SignalIndex has duplicate values"


class TestFeatureCountArithmetic:
    """Verify count constants are consistent with each other."""

    def test_lob_count(self):
        assert LOB_FEATURE_COUNT == LOB_LEVELS * 4

    def test_stable_sum(self):
        computed = (
            LOB_FEATURE_COUNT
            + DERIVED_FEATURE_COUNT
            + MBO_FEATURE_COUNT
            + SIGNAL_FEATURE_COUNT
        )
        assert computed == FEATURE_COUNT, (
            f"Sum of group counts ({computed}) != FEATURE_COUNT ({FEATURE_COUNT})"
        )

    def test_stable_aliases(self):
        assert STANDARD_FEATURE_COUNT == FEATURE_COUNT

    def test_extended_sum(self):
        assert EXTENDED_FEATURE_COUNT == FEATURE_COUNT + EXPERIMENTAL_FEATURE_COUNT

    def test_full_aliases(self):
        assert FULL_FEATURE_COUNT == EXTENDED_FEATURE_COUNT
        assert FEATURE_COUNT_WITH_EXPERIMENTAL == EXTENDED_FEATURE_COUNT

    def test_experimental_group_sum(self):
        computed = (
            EXPERIMENTAL_INSTITUTIONAL_V2_COUNT
            + EXPERIMENTAL_VOLATILITY_COUNT
            + EXPERIMENTAL_SEASONALITY_COUNT
            + EXPERIMENTAL_MLOFI_COUNT
            + EXPERIMENTAL_KOLM_OF_COUNT
        )
        assert computed == EXPERIMENTAL_FEATURE_COUNT, (
            f"Sum of experimental groups ({computed}) != "
            f"EXPERIMENTAL_FEATURE_COUNT ({EXPERIMENTAL_FEATURE_COUNT})"
        )


class TestFeatureIndexRange:
    """Verify enum values cover the expected contiguous ranges."""

    def test_stable_range(self):
        values = sorted(m.value for m in FeatureIndex)
        assert values == list(range(FEATURE_COUNT)), (
            f"FeatureIndex values are not contiguous 0..{FEATURE_COUNT - 1}"
        )

    def test_experimental_range(self):
        values = sorted(m.value for m in ExperimentalFeatureIndex)
        expected = list(range(FEATURE_COUNT, FULL_FEATURE_COUNT))
        assert values == expected, (
            f"ExperimentalFeatureIndex values are not contiguous "
            f"{FEATURE_COUNT}..{FULL_FEATURE_COUNT - 1}"
        )

    def test_signal_subset_of_feature_index(self):
        signal_values = set(m.value for m in SignalIndex)
        feature_values = set(m.value for m in FeatureIndex)
        assert signal_values.issubset(feature_values), (
            "SignalIndex values must be a subset of FeatureIndex"
        )


class TestSliceConsistency:
    """Verify slices are contiguous and cover the correct ranges."""

    def test_lob_subslices(self):
        assert LOB_ASK_PRICES == slice(0, LOB_LEVELS)
        assert LOB_ASK_SIZES == slice(LOB_LEVELS, LOB_LEVELS * 2)
        assert LOB_BID_PRICES == slice(LOB_LEVELS * 2, LOB_LEVELS * 3)
        assert LOB_BID_SIZES == slice(LOB_LEVELS * 3, LOB_FEATURE_COUNT)

    def test_lob_all(self):
        assert LOB_ALL == slice(0, LOB_FEATURE_COUNT)

    def test_category_slices_contiguous(self):
        assert LOB_ALL.stop == DERIVED_ALL.start
        assert DERIVED_ALL.stop == MBO_ALL.start
        assert MBO_ALL.stop == SIGNALS_ALL.start
        assert SIGNALS_ALL.stop == FEATURE_COUNT

    def test_experimental_slices_contiguous(self):
        assert EXPERIMENTAL_ALL.start == FEATURE_COUNT
        assert EXPERIMENTAL_ALL.stop == FULL_FEATURE_COUNT
        assert EXPERIMENTAL_INSTITUTIONAL_V2_SLICE.stop == EXPERIMENTAL_VOLATILITY_SLICE.start
        assert EXPERIMENTAL_VOLATILITY_SLICE.stop == EXPERIMENTAL_SEASONALITY_SLICE.start
        assert EXPERIMENTAL_SEASONALITY_SLICE.stop == EXPERIMENTAL_MLOFI_SLICE.start
        assert EXPERIMENTAL_MLOFI_SLICE.stop == EXPERIMENTAL_KOLM_OF_SLICE.start
        assert EXPERIMENTAL_KOLM_OF_SLICE.stop == FULL_FEATURE_COUNT

    def test_experimental_slice_sizes(self):
        inst = EXPERIMENTAL_INSTITUTIONAL_V2_SLICE
        vol = EXPERIMENTAL_VOLATILITY_SLICE
        seas = EXPERIMENTAL_SEASONALITY_SLICE
        mlofi = EXPERIMENTAL_MLOFI_SLICE
        kolm = EXPERIMENTAL_KOLM_OF_SLICE
        assert inst.stop - inst.start == EXPERIMENTAL_INSTITUTIONAL_V2_COUNT
        assert vol.stop - vol.start == EXPERIMENTAL_VOLATILITY_COUNT
        assert seas.stop - seas.start == EXPERIMENTAL_SEASONALITY_COUNT
        assert mlofi.stop - mlofi.start == EXPERIMENTAL_MLOFI_COUNT
        assert kolm.stop - kolm.start == EXPERIMENTAL_KOLM_OF_COUNT


class TestLayoutIndices:
    """Verify layout index tuples are correct."""

    def test_grouped_price_count(self):
        assert len(GROUPED_PRICE_INDICES) == LOB_LEVELS * 2

    def test_grouped_size_count(self):
        assert len(GROUPED_SIZE_INDICES) == LOB_LEVELS * 2

    def test_grouped_no_overlap(self):
        assert not set(GROUPED_PRICE_INDICES) & set(GROUPED_SIZE_INDICES)

    def test_grouped_union_covers_lob(self):
        union = set(GROUPED_PRICE_INDICES) | set(GROUPED_SIZE_INDICES)
        assert union == set(range(LOB_FEATURE_COUNT))

    def test_lobster_price_count(self):
        assert len(LOBSTER_PRICE_INDICES) == LOB_LEVELS * 2

    def test_lobster_size_count(self):
        assert len(LOBSTER_SIZE_INDICES) == LOB_LEVELS * 2

    def test_lobster_no_overlap(self):
        assert not set(LOBSTER_PRICE_INDICES) & set(LOBSTER_SIZE_INDICES)

    def test_lobster_union_covers_lob(self):
        union = set(LOBSTER_PRICE_INDICES) | set(LOBSTER_SIZE_INDICES)
        assert union == set(range(LOB_FEATURE_COUNT))


class TestClassificationSets:
    """Verify classification sets reference valid indices."""

    def test_categorical_indices_valid(self):
        all_valid = set(range(FULL_FEATURE_COUNT))
        assert CATEGORICAL_INDICES.issubset(all_valid), (
            f"CATEGORICAL_INDICES contains invalid indices: "
            f"{CATEGORICAL_INDICES - all_valid}"
        )

    def test_unsigned_features_valid(self):
        all_valid = set(m.value for m in FeatureIndex)
        assert set(UNSIGNED_FEATURES).issubset(all_valid)

    def test_safety_gates_valid(self):
        all_valid = set(m.value for m in FeatureIndex)
        assert set(SAFETY_GATES).issubset(all_valid)

    def test_primary_signals_valid(self):
        all_valid = set(m.value for m in FeatureIndex)
        assert set(PRIMARY_SIGNALS).issubset(all_valid)

    def test_asymmetry_signals_valid(self):
        all_valid = set(m.value for m in FeatureIndex)
        assert set(ASYMMETRY_SIGNALS).issubset(all_valid)


class TestNameDictionaries:
    """Verify name lookup dictionaries are consistent."""

    def test_signal_names_count(self):
        assert len(SIGNAL_NAMES) == SIGNAL_FEATURE_COUNT

    def test_signal_names_keys_match_signal_index(self):
        enum_values = set(m.value for m in SignalIndex)
        assert set(SIGNAL_NAMES.keys()) == enum_values

    def test_experimental_names_count(self):
        assert len(EXPERIMENTAL_FEATURE_NAMES) == EXPERIMENTAL_FEATURE_COUNT

    def test_experimental_names_keys_match_enum(self):
        enum_values = set(m.value for m in ExperimentalFeatureIndex)
        assert set(EXPERIMENTAL_FEATURE_NAMES.keys()) == enum_values


class TestLabelContracts:
    """Verify label contracts are well-formed."""

    def test_tlob_contract(self):
        c = TLOB_CONTRACT
        assert c.strategy == LabelingStrategy.TLOB
        assert c.num_classes == NUM_CLASSES
        assert c.values == (-1, 0, 1)
        assert c.shift_for_crossentropy is True

    def test_tb_contract(self):
        c = TB_CONTRACT
        assert c.strategy == LabelingStrategy.TRIPLE_BARRIER
        assert c.num_classes == NUM_CLASSES
        assert c.values == (0, 1, 2)
        assert c.shift_for_crossentropy is False

    def test_opportunity_contract(self):
        c = OPPORTUNITY_CONTRACT
        assert c.strategy == LabelingStrategy.OPPORTUNITY
        assert c.num_classes == NUM_CLASSES
        assert c.values == (-1, 0, 1)
        assert c.shift_for_crossentropy is True

    def test_get_contract_lookup(self):
        assert get_contract("tlob") is TLOB_CONTRACT
        assert get_contract("triple_barrier") is TB_CONTRACT
        assert get_contract("opportunity") is OPPORTUNITY_CONTRACT
        assert get_contract("regression") is REGRESSION_CONTRACT

    def test_get_contract_invalid(self):
        with pytest.raises(ValueError, match="Unknown labeling strategy"):
            get_contract("nonexistent")

    def test_regression_contract(self):
        c = REGRESSION_CONTRACT
        assert c.strategy == LabelingStrategy.REGRESSION
        assert c.encoding == "continuous_bps"
        assert c.dtype == "float64"
        assert c.unit == "basis_points"
        assert isinstance(c, RegressionLabelContract)

    def test_is_regression_strategy(self):
        assert is_regression_strategy("regression") is True
        assert is_regression_strategy("Regression") is True
        assert is_regression_strategy("tlob") is False
        assert is_regression_strategy("triple_barrier") is False

    def test_label_constants(self):
        assert LABEL_DOWN == -1
        assert LABEL_STABLE == 0
        assert LABEL_UP == 1

    def test_get_label_name_original(self):
        assert get_label_name(-1) == "Down"
        assert get_label_name(0) == "Stable"
        assert get_label_name(1) == "Up"

    def test_get_label_name_shifted(self):
        assert get_label_name(0, shifted=True) == "Down"
        assert get_label_name(1, shifted=True) == "Stable"
        assert get_label_name(2, shifted=True) == "Up"


class TestValidation:
    """Verify validation functions behave correctly."""

    def test_validate_feature_indices_valid(self):
        validate_feature_indices((0, 1, 84, 97), 98, "test")

    def test_validate_feature_indices_empty(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_feature_indices((), 98, "test")

    def test_validate_feature_indices_out_of_bounds(self):
        with pytest.raises(ValueError, match="contains index 100"):
            validate_feature_indices((0, 100), 98, "test")

    def test_validate_feature_indices_negative(self):
        with pytest.raises(ValueError, match="negative index"):
            validate_feature_indices((-1, 0), 98, "test")

    def test_validate_feature_indices_duplicate(self):
        with pytest.raises(ValueError, match="duplicates"):
            validate_feature_indices((0, 1, 0), 98, "test")


class TestSchemaVersion:
    """Verify schema version is the expected value."""

    def test_schema_version_value(self):
        assert SCHEMA_VERSION == "2.2"

    def test_known_v22_features(self):
        """Verify the v2.2-specific MBO Core names exist (the drift fix)."""
        assert FeatureIndex.AVG_ORDER_AGE == 78
        assert FeatureIndex.MEDIAN_ORDER_LIFETIME == 79
        assert FeatureIndex.AVG_FILL_RATIO == 80
        assert FeatureIndex.AVG_TIME_TO_FIRST_FILL == 81
        assert FeatureIndex.CANCEL_TO_ADD_RATIO == 82
        assert FeatureIndex.ACTIVE_ORDER_COUNT == 83


# =============================================================================
# Off-Exchange Feature Contract Tests
# =============================================================================


class TestOffExchangeEnumCount:
    """Verify OffExchangeFeatureIndex has the correct number of members."""

    def test_enum_count(self):
        assert len(OffExchangeFeatureIndex) == OFF_EXCHANGE_FEATURE_COUNT

    def test_count_is_34(self):
        assert OFF_EXCHANGE_FEATURE_COUNT == 34

    def test_active_count(self):
        assert OFF_EXCHANGE_ACTIVE_FEATURE_COUNT == 30

    def test_no_duplicate_values(self):
        values = [m.value for m in OffExchangeFeatureIndex]
        assert len(values) == len(set(values)), "Duplicate values in OffExchangeFeatureIndex"


class TestOffExchangeIndexRange:
    """Verify off-exchange enum values are contiguous 0..33."""

    def test_contiguous_range(self):
        values = sorted(m.value for m in OffExchangeFeatureIndex)
        assert values == list(range(OFF_EXCHANGE_FEATURE_COUNT)), (
            f"OffExchangeFeatureIndex values are not contiguous 0..{OFF_EXCHANGE_FEATURE_COUNT - 1}"
        )


class TestOffExchangeGroupSum:
    """Verify off-exchange group counts sum to total."""

    def test_group_sum(self):
        # signed_flow(4) + venue_metrics(4) + retail_metrics(4) + bbo_dynamics(6)
        # + vpin(2) + trade_size(4) + cross_venue(3) + activity(2)
        # + safety_gates(2) + context(3) = 34
        group_slices = [
            OFF_EXCHANGE_SIGNED_FLOW, OFF_EXCHANGE_VENUE_METRICS,
            OFF_EXCHANGE_RETAIL_METRICS, OFF_EXCHANGE_BBO_DYNAMICS,
            OFF_EXCHANGE_VPIN, OFF_EXCHANGE_TRADE_SIZE,
            OFF_EXCHANGE_CROSS_VENUE, OFF_EXCHANGE_ACTIVITY,
            OFF_EXCHANGE_SAFETY_GATES_SLICE, OFF_EXCHANGE_CONTEXT,
        ]
        total = sum(s.stop - s.start for s in group_slices)
        assert total == OFF_EXCHANGE_FEATURE_COUNT, (
            f"Sum of off-exchange group sizes ({total}) != "
            f"OFF_EXCHANGE_FEATURE_COUNT ({OFF_EXCHANGE_FEATURE_COUNT})"
        )


class TestOffExchangeSlicesContiguous:
    """Verify off-exchange group slices are contiguous."""

    def test_contiguous(self):
        slices = [
            OFF_EXCHANGE_SIGNED_FLOW, OFF_EXCHANGE_VENUE_METRICS,
            OFF_EXCHANGE_RETAIL_METRICS, OFF_EXCHANGE_BBO_DYNAMICS,
            OFF_EXCHANGE_VPIN, OFF_EXCHANGE_TRADE_SIZE,
            OFF_EXCHANGE_CROSS_VENUE, OFF_EXCHANGE_ACTIVITY,
            OFF_EXCHANGE_SAFETY_GATES_SLICE, OFF_EXCHANGE_CONTEXT,
        ]
        assert slices[0].start == 0
        for i in range(1, len(slices)):
            assert slices[i].start == slices[i - 1].stop, (
                f"Gap between group {i-1} (stop={slices[i-1].stop}) "
                f"and group {i} (start={slices[i].start})"
            )
        assert slices[-1].stop == OFF_EXCHANGE_FEATURE_COUNT


class TestOffExchangeClassificationSets:
    """Verify off-exchange classification sets are valid."""

    def test_categorical_valid(self):
        valid = set(range(OFF_EXCHANGE_FEATURE_COUNT))
        assert OFF_EXCHANGE_CATEGORICAL_INDICES.issubset(valid)

    def test_categorical_expected(self):
        # bin_valid=29, bbo_valid=30, time_bucket=32, schema_version=33
        assert OFF_EXCHANGE_CATEGORICAL_INDICES == frozenset({29, 30, 32, 33})

    def test_non_normalizable_superset_of_categorical(self):
        assert OFF_EXCHANGE_CATEGORICAL_INDICES.issubset(OFF_EXCHANGE_NON_NORMALIZABLE_INDICES)

    def test_safety_gates_valid(self):
        valid = set(range(OFF_EXCHANGE_FEATURE_COUNT))
        assert set(OFF_EXCHANGE_SAFETY_GATES).issubset(valid)

    def test_safety_gates_expected(self):
        assert OFF_EXCHANGE_SAFETY_GATES == (29, 30)

    def test_unsigned_valid(self):
        valid = set(range(OFF_EXCHANGE_FEATURE_COUNT))
        assert OFF_EXCHANGE_UNSIGNED_FEATURES.issubset(valid)

    def test_session_progress_is_not_categorical(self):
        """session_progress (31) IS an active model feature, NOT categorical."""
        assert 31 not in OFF_EXCHANGE_CATEGORICAL_INDICES


class TestOffExchangeNameDict:
    """Verify OFF_EXCHANGE_FEATURE_NAMES dictionary."""

    def test_count(self):
        assert len(OFF_EXCHANGE_FEATURE_NAMES) == OFF_EXCHANGE_FEATURE_COUNT

    def test_keys_match_enum(self):
        enum_values = set(m.value for m in OffExchangeFeatureIndex)
        assert set(OFF_EXCHANGE_FEATURE_NAMES.keys()) == enum_values

    def test_names_are_lowercase(self):
        for idx, name in OFF_EXCHANGE_FEATURE_NAMES.items():
            assert name == name.lower(), f"Name at index {idx} is not lowercase: {name}"


class TestOffExchangeSpecificMembers:
    """Spot-check specific enum members and their values."""

    def test_first_feature(self):
        assert OffExchangeFeatureIndex.TRF_SIGNED_IMBALANCE == 0

    def test_spread_bps(self):
        assert OffExchangeFeatureIndex.SPREAD_BPS == 12

    def test_bin_valid(self):
        assert OffExchangeFeatureIndex.BIN_VALID == 29

    def test_session_progress(self):
        assert OffExchangeFeatureIndex.SESSION_PROGRESS == 31

    def test_schema_version(self):
        assert OffExchangeFeatureIndex.SCHEMA_VERSION == 33

    def test_schema_version_string(self):
        assert OFF_EXCHANGE_SCHEMA_VERSION == "1.0"
