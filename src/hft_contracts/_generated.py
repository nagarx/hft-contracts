# AUTO-GENERATED from contracts/pipeline_contract.toml
# DO NOT EDIT — run: python contracts/generate_python_contract.py
# Schema version: 2.2 | Generated: 2026-04-21 01:27 UTC
"""
Pipeline contract constants generated from the authoritative TOML definition.

This module is the single source of truth for all feature indices, counts,
slices, layout index tuples, and classification sets used across the
HFT pipeline's Python modules.

Sign convention (RULE.md §9):
    > 0 = Bullish / Buy pressure
    < 0 = Bearish / Sell pressure
    = 0 = Neutral / No signal
"""

from enum import IntEnum
from typing import Dict, Final


# =====================================================================
# Schema Version
# =====================================================================

SCHEMA_VERSION: Final[str] = "2.2"
"""
Schema version for the feature export format.

A change to any stable feature index, label encoding, or normalization
semantic constitutes a breaking change and requires a version bump here.
"""

SCHEMA_VERSION_FLOAT: Final[float] = 2.2
"""Numeric form for backward-compatible comparisons with older metadata."""


# =====================================================================
# Feature Counts
# =====================================================================

LOB_LEVELS: Final[int] = 10
"""Number of LOB price levels per side."""

LOB_FEATURE_COUNT: Final[int] = 40
"""Raw LOB features: 10 levels x 4 (ask_prices, ask_sizes, bid_prices, bid_sizes)."""

DERIVED_FEATURE_COUNT: Final[int] = 8
"""Derived features: mid_price, spread, spread_bps, volumes, microprice, price_impact."""

MBO_FEATURE_COUNT: Final[int] = 36
"""MBO features: order flow, size distribution, queue/depth, institutional, core."""

SIGNAL_FEATURE_COUNT: Final[int] = 14
"""Trading signals: OFI, asymmetry, regime, safety gates."""

EXPERIMENTAL_INSTITUTIONAL_V2_COUNT: Final[int] = 8
"""Experimental institutional_v2 features."""

EXPERIMENTAL_VOLATILITY_COUNT: Final[int] = 6
"""Experimental volatility features."""

EXPERIMENTAL_SEASONALITY_COUNT: Final[int] = 4
"""Experimental seasonality features."""

EXPERIMENTAL_MLOFI_COUNT: Final[int] = 12
"""Experimental mlofi features."""

EXPERIMENTAL_KOLM_OF_COUNT: Final[int] = 20
"""Experimental kolm_of features."""

EXPERIMENTAL_FEATURE_COUNT: Final[int] = 50
"""Total experimental features (all groups)."""

FEATURE_COUNT: Final[int] = 98
"""Standard (stable) feature count. Backward-compatible alias."""

STANDARD_FEATURE_COUNT: Final[int] = 98
"""Standard (stable) feature count. Explicit name."""

EXTENDED_FEATURE_COUNT: Final[int] = 148
"""Extended feature count: stable + experimental."""

FULL_FEATURE_COUNT: Final[int] = 148
"""Alias for EXTENDED_FEATURE_COUNT."""

FEATURE_COUNT_WITH_EXPERIMENTAL: Final[int] = 148
"""Alias for backward compatibility."""


# =====================================================================
# FeatureIndex Enum (98 stable features)
# =====================================================================


class FeatureIndex(IntEnum):
    """
    Complete feature index mapping for the stable 98-feature export.

    Usage:
        >>> features[:, FeatureIndex.TRUE_OFI]  # Access OFI signal
        >>> features[:, FeatureIndex.MID_PRICE]  # Access mid price

    Sign conventions (RULE.md §9):
        All directional signals: > 0 = BULLISH, < 0 = BEARISH.
        Exception: PRICE_IMPACT (47) is unsigned.

    LOB layout (matches Rust pipeline):
        0-9: Ask prices | 10-19: Ask sizes | 20-29: Bid prices | 30-39: Bid sizes
    """

    ASK_PRICE_L0 = 0
    ASK_PRICE_L1 = 1
    ASK_PRICE_L2 = 2
    ASK_PRICE_L3 = 3
    ASK_PRICE_L4 = 4
    ASK_PRICE_L5 = 5
    ASK_PRICE_L6 = 6
    ASK_PRICE_L7 = 7
    ASK_PRICE_L8 = 8
    ASK_PRICE_L9 = 9
    ASK_SIZE_L0 = 10
    ASK_SIZE_L1 = 11
    ASK_SIZE_L2 = 12
    ASK_SIZE_L3 = 13
    ASK_SIZE_L4 = 14
    ASK_SIZE_L5 = 15
    ASK_SIZE_L6 = 16
    ASK_SIZE_L7 = 17
    ASK_SIZE_L8 = 18
    ASK_SIZE_L9 = 19
    BID_PRICE_L0 = 20
    BID_PRICE_L1 = 21
    BID_PRICE_L2 = 22
    BID_PRICE_L3 = 23
    BID_PRICE_L4 = 24
    BID_PRICE_L5 = 25
    BID_PRICE_L6 = 26
    BID_PRICE_L7 = 27
    BID_PRICE_L8 = 28
    BID_PRICE_L9 = 29
    BID_SIZE_L0 = 30
    BID_SIZE_L1 = 31
    BID_SIZE_L2 = 32
    BID_SIZE_L3 = 33
    BID_SIZE_L4 = 34
    BID_SIZE_L5 = 35
    BID_SIZE_L6 = 36
    BID_SIZE_L7 = 37
    BID_SIZE_L8 = 38
    BID_SIZE_L9 = 39
    MID_PRICE = 40
    SPREAD = 41
    SPREAD_BPS = 42
    TOTAL_BID_VOLUME = 43
    TOTAL_ASK_VOLUME = 44
    VOLUME_IMBALANCE = 45
    WEIGHTED_MID_PRICE = 46
    PRICE_IMPACT = 47
    ADD_RATE_BID = 48
    ADD_RATE_ASK = 49
    CANCEL_RATE_BID = 50
    CANCEL_RATE_ASK = 51
    TRADE_RATE_BID = 52
    TRADE_RATE_ASK = 53
    NET_ORDER_FLOW = 54
    NET_CANCEL_FLOW = 55
    NET_TRADE_FLOW = 56
    AGGRESSIVE_ORDER_RATIO = 57
    ORDER_FLOW_VOLATILITY = 58
    FLOW_REGIME_INDICATOR = 59
    SIZE_P25 = 60
    SIZE_P50 = 61
    SIZE_P75 = 62
    SIZE_P90 = 63
    SIZE_ZSCORE = 64
    LARGE_ORDER_RATIO = 65
    SIZE_SKEWNESS = 66
    SIZE_CONCENTRATION = 67
    AVG_QUEUE_POSITION = 68
    QUEUE_SIZE_AHEAD = 69
    ORDERS_PER_LEVEL = 70
    LEVEL_CONCENTRATION = 71
    DEPTH_TICKS_BID = 72
    DEPTH_TICKS_ASK = 73
    LARGE_ORDER_FREQUENCY = 74
    LARGE_ORDER_IMBALANCE = 75
    MODIFICATION_SCORE = 76
    ICEBERG_PROXY = 77
    AVG_ORDER_AGE = 78
    MEDIAN_ORDER_LIFETIME = 79
    AVG_FILL_RATIO = 80
    AVG_TIME_TO_FIRST_FILL = 81
    CANCEL_TO_ADD_RATIO = 82
    ACTIVE_ORDER_COUNT = 83
    TRUE_OFI = 84
    DEPTH_NORM_OFI = 85
    EXECUTED_PRESSURE = 86
    SIGNED_MP_DELTA_BPS = 87
    TRADE_ASYMMETRY = 88
    CANCEL_ASYMMETRY = 89
    FRAGILITY_SCORE = 90
    DEPTH_ASYMMETRY = 91
    BOOK_VALID = 92
    TIME_REGIME = 93
    MBO_READY = 94
    DT_SECONDS = 95
    INVALIDITY_DELTA = 96
    SCHEMA_VERSION = 97


# =====================================================================
# ExperimentalFeatureIndex Enum (50 experimental features)
# =====================================================================


class ExperimentalFeatureIndex(IntEnum):
    """
    Experimental feature indices (98-147).

    NOT part of the stable schema. These may change without a version bump.
    Check feature array shape before using:

        if features.shape[-1] >= 148:
            # Experimental features available
    """

    ROUND_LOT_RATIO = 98
    ODD_LOT_RATIO = 99
    SIZE_CLUSTERING = 100
    PRICE_CLUSTERING = 101
    MOD_BEFORE_CANCEL = 102
    SWEEP_RATIO = 103
    FILL_PATIENCE_BID = 104
    FILL_PATIENCE_ASK = 105
    REALIZED_VOL_FAST = 106
    REALIZED_VOL_SLOW = 107
    VOL_RATIO = 108
    VOL_MOMENTUM = 109
    RETURN_AUTOCORR = 110
    VOL_OF_VOL = 111
    MINUTES_SINCE_OPEN = 112
    MINUTES_UNTIL_CLOSE = 113
    SESSION_PROGRESS = 114
    TIME_BUCKET = 115
    TOTAL_MLOFI = 116
    WEIGHTED_MLOFI = 117
    OFI_LEVEL_1 = 118
    OFI_LEVEL_2 = 119
    OFI_LEVEL_3 = 120
    OFI_LEVEL_4 = 121
    OFI_LEVEL_5 = 122
    OFI_LEVEL_6 = 123
    OFI_LEVEL_7 = 124
    OFI_LEVEL_8 = 125
    OFI_LEVEL_9 = 126
    OFI_LEVEL_10 = 127
    BOF_LEVEL_1 = 128
    BOF_LEVEL_2 = 129
    BOF_LEVEL_3 = 130
    BOF_LEVEL_4 = 131
    BOF_LEVEL_5 = 132
    BOF_LEVEL_6 = 133
    BOF_LEVEL_7 = 134
    BOF_LEVEL_8 = 135
    BOF_LEVEL_9 = 136
    BOF_LEVEL_10 = 137
    AOF_LEVEL_1 = 138
    AOF_LEVEL_2 = 139
    AOF_LEVEL_3 = 140
    AOF_LEVEL_4 = 141
    AOF_LEVEL_5 = 142
    AOF_LEVEL_6 = 143
    AOF_LEVEL_7 = 144
    AOF_LEVEL_8 = 145
    AOF_LEVEL_9 = 146
    AOF_LEVEL_10 = 147


# =====================================================================
# SignalIndex Enum (convenience alias for indices 84-97)
# =====================================================================


class SignalIndex(IntEnum):
    """Convenience enum for the 14 trading signals (indices 84-97)."""

    TRUE_OFI = 84
    DEPTH_NORM_OFI = 85
    EXECUTED_PRESSURE = 86
    SIGNED_MP_DELTA_BPS = 87
    TRADE_ASYMMETRY = 88
    CANCEL_ASYMMETRY = 89
    FRAGILITY_SCORE = 90
    DEPTH_ASYMMETRY = 91
    BOOK_VALID = 92
    TIME_REGIME = 93
    MBO_READY = 94
    DT_SECONDS = 95
    INVALIDITY_DELTA = 96
    SCHEMA_VERSION = 97


# =====================================================================
# Feature Group Slices
# =====================================================================

LOB_ASK_PRICES = slice(0, 10)
LOB_ASK_SIZES = slice(10, 20)
LOB_BID_PRICES = slice(20, 30)
LOB_BID_SIZES = slice(30, 40)
LOB_ALL = slice(0, 40)

DERIVED_ALL = slice(40, 48)
MBO_ALL = slice(48, 84)
SIGNALS_ALL = slice(84, 98)

EXPERIMENTAL_ALL = slice(98, 148)
EXPERIMENTAL_ALL_SLICE = slice(98, 148)

EXPERIMENTAL_INSTITUTIONAL_V2_SLICE = slice(98, 106)
EXPERIMENTAL_INSTITUTIONAL_V2 = EXPERIMENTAL_INSTITUTIONAL_V2_SLICE
EXPERIMENTAL_VOLATILITY_SLICE = slice(106, 112)
EXPERIMENTAL_VOLATILITY = EXPERIMENTAL_VOLATILITY_SLICE
EXPERIMENTAL_SEASONALITY_SLICE = slice(112, 116)
EXPERIMENTAL_SEASONALITY = EXPERIMENTAL_SEASONALITY_SLICE
EXPERIMENTAL_MLOFI_SLICE = slice(116, 128)
EXPERIMENTAL_MLOFI = EXPERIMENTAL_MLOFI_SLICE
EXPERIMENTAL_KOLM_OF_SLICE = slice(128, 148)
EXPERIMENTAL_KOLM_OF = EXPERIMENTAL_KOLM_OF_SLICE


# =====================================================================
# Layout Index Tuples (for normalization)
# =====================================================================

GROUPED_PRICE_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29)
"""Price column indices for GROUPED layout (our Rust pipeline)."""

GROUPED_SIZE_INDICES: tuple[int, ...] = (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39)
"""Size column indices for GROUPED layout."""

LOBSTER_PRICE_INDICES: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38)
"""Price column indices for LOBSTER/FI2010 interleaved layout."""

LOBSTER_SIZE_INDICES: tuple[int, ...] = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39)
"""Size column indices for LOBSTER/FI2010 interleaved layout."""


# =====================================================================
# Feature Classification Sets
# =====================================================================

CATEGORICAL_INDICES: Final[frozenset[int]] = frozenset({92, 93, 94, 97, 115})
"""Indices of categorical/binary features. Must NEVER be normalized."""

UNSIGNED_FEATURES: Final[frozenset[int]] = frozenset({47})
"""Unsigned features. Cannot be used for directional signals."""

SAFETY_GATES: tuple[int, ...] = (92, 94, 96)
"""Indices that must be checked before using other signals."""

PRIMARY_SIGNALS: tuple[int, ...] = (84, 86, 92)
"""Most important signals per Cont et al. (2014)."""

ASYMMETRY_SIGNALS: tuple[int, ...] = (88, 89, 91)
"""Normalized asymmetry signals, all in range [-1, 1]."""


# =====================================================================
# Name Lookup Dictionaries
# =====================================================================

SIGNAL_NAMES: Final[Dict[int, str]] = {
    84: "true_ofi",
    85: "depth_norm_ofi",
    86: "executed_pressure",
    87: "signed_mp_delta_bps",
    88: "trade_asymmetry",
    89: "cancel_asymmetry",
    90: "fragility_score",
    91: "depth_asymmetry",
    92: "book_valid",
    93: "time_regime",
    94: "mbo_ready",
    95: "dt_seconds",
    96: "invalidity_delta",
    97: "schema_version",
}
"""Human-readable signal names."""

EXPERIMENTAL_FEATURE_NAMES: Final[Dict[int, str]] = {
    98: "round_lot_ratio",
    99: "odd_lot_ratio",
    100: "size_clustering",
    101: "price_clustering",
    102: "mod_before_cancel",
    103: "sweep_ratio",
    104: "fill_patience_bid",
    105: "fill_patience_ask",
    106: "realized_vol_fast",
    107: "realized_vol_slow",
    108: "vol_ratio",
    109: "vol_momentum",
    110: "return_autocorr",
    111: "vol_of_vol",
    112: "minutes_since_open",
    113: "minutes_until_close",
    114: "session_progress",
    115: "time_bucket",
    116: "total_mlofi",
    117: "weighted_mlofi",
    118: "ofi_level_1",
    119: "ofi_level_2",
    120: "ofi_level_3",
    121: "ofi_level_4",
    122: "ofi_level_5",
    123: "ofi_level_6",
    124: "ofi_level_7",
    125: "ofi_level_8",
    126: "ofi_level_9",
    127: "ofi_level_10",
    128: "bof_level_1",
    129: "bof_level_2",
    130: "bof_level_3",
    131: "bof_level_4",
    132: "bof_level_5",
    133: "bof_level_6",
    134: "bof_level_7",
    135: "bof_level_8",
    136: "bof_level_9",
    137: "bof_level_10",
    138: "aof_level_1",
    139: "aof_level_2",
    140: "aof_level_3",
    141: "aof_level_4",
    142: "aof_level_5",
    143: "aof_level_6",
    144: "aof_level_7",
    145: "aof_level_8",
    146: "aof_level_9",
    147: "aof_level_10",
}
"""Human-readable experimental feature names."""


# =====================================================================
# Normalization Contract
# =====================================================================

NON_NORMALIZABLE_INDICES: Final[frozenset[int]] = frozenset({92, 93, 94, 95, 96, 97, 115})
"""
Indices that must NOT be normalized. Superset of CATEGORICAL_INDICES,
also includes special-semantic features like invalidity_delta (counter).
"""


# =====================================================================
# Export Metadata Contract
# =====================================================================

EXPORT_METADATA_REQUIRED_FIELDS: Final[tuple[str, ...]] = ('day', 'n_sequences', 'window_size', 'n_features', 'schema_version', 'contract_version', 'label_strategy', 'label_encoding', 'normalization', 'provenance', 'export_timestamp')
"""Required top-level fields in every {day}_metadata.json."""

EXPORT_METADATA_NORMALIZATION_FIELDS: Final[tuple[str, ...]] = ('strategy', 'applied', 'params_file')
"""Required fields inside the metadata normalization block."""

EXPORT_METADATA_PROVENANCE_FIELDS: Final[tuple[str, ...]] = ('extractor_version', 'git_commit', 'git_dirty', 'config_hash', 'contract_version', 'export_timestamp_utc')
"""Required fields inside the metadata provenance block."""

EXPORT_MANIFEST_REQUIRED_FIELDS: Final[tuple[str, ...]] = ('experiment', 'symbol', 'feature_count', 'days_processed', 'export_timestamp', 'config_hash', 'schema_version', 'sequence_length', 'stride', 'labeling_strategy', 'horizons', 'splits')
"""Required fields in dataset_manifest.json."""


# =====================================================================
# Signal Export Contract (Trainer -> Backtester)
# =====================================================================

SIGNAL_EXPORT_FILES: Final[tuple[str, ...]] = ('predictions.npy', 'agreement_ratio.npy', 'confirmation_score.npy', 'spreads.npy', 'prices.npy', 'labels.npy', 'signal_metadata.json')
"""Files produced by export_hmhp_signals.py for backtester consumption."""

SIGNAL_SPREAD_FEATURE_INDEX: Final[int] = 42
"""Feature index for spread used in signal export (from features.derived.spread_bps)."""

SIGNAL_PRICE_FEATURE_INDEX: Final[int] = 40
"""Feature index for mid price used in signal export (from features.derived.mid_price)."""

SIGNAL_CLASS_DOWN: Final[int] = 0
SIGNAL_CLASS_STABLE: Final[int] = 1
SIGNAL_CLASS_UP: Final[int] = 2
"""Signal prediction class encoding."""



# =====================================================================
# Off-Exchange Feature Contract (basic-quote-processor)
# Independent index space (0-33) — NOT an extension of MBO 0-147.
# Source: [features.off_exchange] in pipeline_contract.toml
# =====================================================================

OFF_EXCHANGE_SCHEMA_VERSION: Final[str] = "1.0"
"""Schema version for the off-exchange feature export format."""

OFF_EXCHANGE_FEATURE_COUNT: Final[int] = 34
"""Total off-exchange feature count (34)."""

OFF_EXCHANGE_ACTIVE_FEATURE_COUNT: Final[int] = 30
"""Active (model-usable) off-exchange features: total minus safety gates and categoricals."""


class OffExchangeFeatureIndex(IntEnum):
    """Off-exchange feature indices (0-33). Independent of MBO FeatureIndex."""

    TRF_SIGNED_IMBALANCE = 0
    MROIB = 1
    INV_INST_DIRECTION = 2
    BVC_IMBALANCE = 3
    DARK_SHARE = 4
    TRF_VOLUME = 5
    LIT_VOLUME = 6
    TOTAL_VOLUME = 7
    SUBPENNY_INTENSITY = 8
    ODD_LOT_RATIO = 9
    RETAIL_TRADE_RATE = 10
    RETAIL_VOLUME_FRACTION = 11
    SPREAD_BPS = 12
    BID_PRESSURE = 13
    ASK_PRESSURE = 14
    BBO_UPDATE_RATE = 15
    QUOTE_IMBALANCE = 16
    SPREAD_CHANGE_RATE = 17
    TRF_VPIN = 18
    LIT_VPIN = 19
    MEAN_TRADE_SIZE = 20
    BLOCK_TRADE_RATIO = 21
    TRADE_COUNT = 22
    SIZE_CONCENTRATION = 23
    TRF_BURST_INTENSITY = 24
    TIME_SINCE_BURST = 25
    TRF_LIT_VOLUME_RATIO = 26
    BIN_TRADE_COUNT = 27
    BIN_TRF_TRADE_COUNT = 28
    BIN_VALID = 29
    BBO_VALID = 30
    SESSION_PROGRESS = 31
    TIME_BUCKET = 32
    SCHEMA_VERSION = 33


# Off-Exchange Group Slices
OFF_EXCHANGE_SIGNED_FLOW = slice(0, 4)
OFF_EXCHANGE_VENUE_METRICS = slice(4, 8)
OFF_EXCHANGE_RETAIL_METRICS = slice(8, 12)
OFF_EXCHANGE_BBO_DYNAMICS = slice(12, 18)
OFF_EXCHANGE_VPIN = slice(18, 20)
OFF_EXCHANGE_TRADE_SIZE = slice(20, 24)
OFF_EXCHANGE_CROSS_VENUE = slice(24, 27)
OFF_EXCHANGE_ACTIVITY = slice(27, 29)
OFF_EXCHANGE_SAFETY_GATES_SLICE = slice(29, 31)
OFF_EXCHANGE_CONTEXT = slice(31, 34)


# Off-Exchange Classification Sets
OFF_EXCHANGE_CATEGORICAL_INDICES: Final[frozenset[int]] = frozenset({29, 30, 32, 33})
"""Off-exchange features excluded from normalization/evaluation (categorical/binary)."""

OFF_EXCHANGE_NON_NORMALIZABLE_INDICES: Final[frozenset[int]] = frozenset({29, 30, 32, 33})
"""Off-exchange features that must not be normalized."""

OFF_EXCHANGE_UNSIGNED_FEATURES: Final[frozenset[int]] = frozenset({4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28})
"""Off-exchange features with no sign convention (volumes, ratios, counts)."""

OFF_EXCHANGE_SAFETY_GATES: Final[tuple[int, ...]] = (29, 30)
"""Off-exchange safety gate feature indices (bin_valid, bbo_valid)."""


# Off-Exchange Feature Names
OFF_EXCHANGE_FEATURE_NAMES: Final[Dict[int, str]] = {
    0: "trf_signed_imbalance",
    1: "mroib",
    2: "inv_inst_direction",
    3: "bvc_imbalance",
    4: "dark_share",
    5: "trf_volume",
    6: "lit_volume",
    7: "total_volume",
    8: "subpenny_intensity",
    9: "odd_lot_ratio",
    10: "retail_trade_rate",
    11: "retail_volume_fraction",
    12: "spread_bps",
    13: "bid_pressure",
    14: "ask_pressure",
    15: "bbo_update_rate",
    16: "quote_imbalance",
    17: "spread_change_rate",
    18: "trf_vpin",
    19: "lit_vpin",
    20: "mean_trade_size",
    21: "block_trade_ratio",
    22: "trade_count",
    23: "size_concentration",
    24: "trf_burst_intensity",
    25: "time_since_burst",
    26: "trf_lit_volume_ratio",
    27: "bin_trade_count",
    28: "bin_trf_trade_count",
    29: "bin_valid",
    30: "bbo_valid",
    31: "session_progress",
    32: "time_bucket",
    33: "schema_version",
}
"""Maps off-exchange feature index -> feature name."""

