"""
HFT Pipeline Contracts — Single Source of Truth.

This package provides the authoritative definitions for all cross-module
data contracts in the HFT pipeline. All Python consumers (lobtrainer,
lobanalyzer, lobbacktest) import from here instead of maintaining
independent copies.

Generated from: contracts/pipeline_contract.toml
Regenerate with: python contracts/generate_python_contract.py

Usage:
    from hft_contracts import FeatureIndex, FEATURE_COUNT, TLOB_CONTRACT
    from hft_contracts import validate_export_contract, ContractError
"""

# -- Generated contract constants (from pipeline_contract.toml) --
from hft_contracts._generated import (
    # Schema
    SCHEMA_VERSION,
    SCHEMA_VERSION_FLOAT,
    # Feature counts
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
    # Enums
    FeatureIndex,
    ExperimentalFeatureIndex,
    SignalIndex,
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
    EXPERIMENTAL_ALL_SLICE,
    EXPERIMENTAL_INSTITUTIONAL_V2,
    EXPERIMENTAL_VOLATILITY,
    EXPERIMENTAL_SEASONALITY,
    EXPERIMENTAL_MLOFI,
    EXPERIMENTAL_KOLM_OF,
    # Layout index tuples
    GROUPED_PRICE_INDICES,
    GROUPED_SIZE_INDICES,
    LOBSTER_PRICE_INDICES,
    LOBSTER_SIZE_INDICES,
    # Classification sets
    CATEGORICAL_INDICES,
    UNSIGNED_FEATURES,
    SAFETY_GATES,
    PRIMARY_SIGNALS,
    ASYMMETRY_SIGNALS,
    # Name lookups
    SIGNAL_NAMES,
    EXPERIMENTAL_FEATURE_NAMES,
    # Normalization contract
    NON_NORMALIZABLE_INDICES,
    # Export metadata contract
    EXPORT_METADATA_REQUIRED_FIELDS,
    EXPORT_METADATA_NORMALIZATION_FIELDS,
    EXPORT_METADATA_PROVENANCE_FIELDS,
    EXPORT_MANIFEST_REQUIRED_FIELDS,
    # Signal export contract (trainer -> backtester)
    SIGNAL_EXPORT_FILES,
    SIGNAL_SPREAD_FEATURE_INDEX,
    SIGNAL_PRICE_FEATURE_INDEX,
    SIGNAL_CLASS_DOWN,
    SIGNAL_CLASS_STABLE,
    SIGNAL_CLASS_UP,
    # Off-Exchange Feature Contract
    OFF_EXCHANGE_SCHEMA_VERSION,
    OFF_EXCHANGE_FEATURE_COUNT,
    OFF_EXCHANGE_ACTIVE_FEATURE_COUNT,
    OffExchangeFeatureIndex,
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

# -- Label contracts --
from hft_contracts.labels import (
    LABEL_DOWN,
    LABEL_STABLE,
    LABEL_UP,
    NUM_CLASSES,
    LABEL_NAMES,
    SHIFTED_LABEL_DOWN,
    SHIFTED_LABEL_STABLE,
    SHIFTED_LABEL_UP,
    SHIFTED_LABEL_NAMES,
    get_label_name,
    LabelingStrategy,
    LabelContract,
    RegressionLabelContract,
    TLOB_CONTRACT,
    TB_CONTRACT,
    OPPORTUNITY_CONTRACT,
    REGRESSION_CONTRACT,
    get_contract,
    is_regression_strategy,
)

# -- Label computation from forward prices --
from hft_contracts.label_factory import (
    DIVISION_GUARD_EPS,
    ForwardPriceContract,
    LabelFactory,
)

# -- Validation utilities --
from hft_contracts.validation import (
    ContractError,
    validate_feature_indices,
    validate_schema_version,
    validate_export_contract,
    validate_normalization_not_applied,
    validate_metadata_completeness,
    validate_label_encoding,
    validate_provenance_present,
    validate_off_exchange_export_contract,
    validate_any_export_contract,
)

__all__ = [
    # Schema
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_FLOAT",
    # Feature counts
    "LOB_LEVELS",
    "LOB_FEATURE_COUNT",
    "DERIVED_FEATURE_COUNT",
    "MBO_FEATURE_COUNT",
    "SIGNAL_FEATURE_COUNT",
    "EXPERIMENTAL_INSTITUTIONAL_V2_COUNT",
    "EXPERIMENTAL_VOLATILITY_COUNT",
    "EXPERIMENTAL_SEASONALITY_COUNT",
    "EXPERIMENTAL_FEATURE_COUNT",
    "FEATURE_COUNT",
    "STANDARD_FEATURE_COUNT",
    "EXTENDED_FEATURE_COUNT",
    "FULL_FEATURE_COUNT",
    "FEATURE_COUNT_WITH_EXPERIMENTAL",
    # Enums
    "FeatureIndex",
    "ExperimentalFeatureIndex",
    "SignalIndex",
    # Slices
    "LOB_ASK_PRICES",
    "LOB_ASK_SIZES",
    "LOB_BID_PRICES",
    "LOB_BID_SIZES",
    "LOB_ALL",
    "DERIVED_ALL",
    "MBO_ALL",
    "SIGNALS_ALL",
    "EXPERIMENTAL_ALL",
    "EXPERIMENTAL_INSTITUTIONAL_V2_SLICE",
    "EXPERIMENTAL_VOLATILITY_SLICE",
    "EXPERIMENTAL_SEASONALITY_SLICE",
    "EXPERIMENTAL_ALL_SLICE",
    "EXPERIMENTAL_INSTITUTIONAL_V2",
    "EXPERIMENTAL_VOLATILITY",
    "EXPERIMENTAL_SEASONALITY",
    # Layout
    "GROUPED_PRICE_INDICES",
    "GROUPED_SIZE_INDICES",
    "LOBSTER_PRICE_INDICES",
    "LOBSTER_SIZE_INDICES",
    # Classification
    "CATEGORICAL_INDICES",
    "UNSIGNED_FEATURES",
    "SAFETY_GATES",
    "PRIMARY_SIGNALS",
    "ASYMMETRY_SIGNALS",
    # Name lookups
    "SIGNAL_NAMES",
    "EXPERIMENTAL_FEATURE_NAMES",
    # Normalization contract
    "NON_NORMALIZABLE_INDICES",
    # Export metadata contract
    "EXPORT_METADATA_REQUIRED_FIELDS",
    "EXPORT_METADATA_NORMALIZATION_FIELDS",
    "EXPORT_METADATA_PROVENANCE_FIELDS",
    "EXPORT_MANIFEST_REQUIRED_FIELDS",
    # Signal export contract
    "SIGNAL_EXPORT_FILES",
    "SIGNAL_SPREAD_FEATURE_INDEX",
    "SIGNAL_PRICE_FEATURE_INDEX",
    "SIGNAL_CLASS_DOWN",
    "SIGNAL_CLASS_STABLE",
    "SIGNAL_CLASS_UP",
    # Labels
    "LABEL_DOWN",
    "LABEL_STABLE",
    "LABEL_UP",
    "NUM_CLASSES",
    "LABEL_NAMES",
    "SHIFTED_LABEL_DOWN",
    "SHIFTED_LABEL_STABLE",
    "SHIFTED_LABEL_UP",
    "SHIFTED_LABEL_NAMES",
    "get_label_name",
    "LabelingStrategy",
    "LabelContract",
    "RegressionLabelContract",
    "TLOB_CONTRACT",
    "TB_CONTRACT",
    "OPPORTUNITY_CONTRACT",
    "REGRESSION_CONTRACT",
    "get_contract",
    "is_regression_strategy",
    # Label computation
    "DIVISION_GUARD_EPS",
    "ForwardPriceContract",
    "LabelFactory",
    # Validation
    "ContractError",
    "validate_feature_indices",
    "validate_schema_version",
    "validate_export_contract",
    "validate_normalization_not_applied",
    "validate_metadata_completeness",
    "validate_label_encoding",
    "validate_provenance_present",
    "validate_off_exchange_export_contract",
    "validate_any_export_contract",
    # Experimental MLOFI/KOLM_OF (added Phase 0)
    "EXPERIMENTAL_MLOFI_COUNT",
    "EXPERIMENTAL_KOLM_OF_COUNT",
    "EXPERIMENTAL_MLOFI_SLICE",
    "EXPERIMENTAL_KOLM_OF_SLICE",
    "EXPERIMENTAL_MLOFI",
    "EXPERIMENTAL_KOLM_OF",
    # Off-Exchange Feature Contract (added Phase 0)
    "OFF_EXCHANGE_SCHEMA_VERSION",
    "OFF_EXCHANGE_FEATURE_COUNT",
    "OFF_EXCHANGE_ACTIVE_FEATURE_COUNT",
    "OffExchangeFeatureIndex",
    "OFF_EXCHANGE_SIGNED_FLOW",
    "OFF_EXCHANGE_VENUE_METRICS",
    "OFF_EXCHANGE_RETAIL_METRICS",
    "OFF_EXCHANGE_BBO_DYNAMICS",
    "OFF_EXCHANGE_VPIN",
    "OFF_EXCHANGE_TRADE_SIZE",
    "OFF_EXCHANGE_CROSS_VENUE",
    "OFF_EXCHANGE_ACTIVITY",
    "OFF_EXCHANGE_SAFETY_GATES_SLICE",
    "OFF_EXCHANGE_CONTEXT",
    "OFF_EXCHANGE_CATEGORICAL_INDICES",
    "OFF_EXCHANGE_NON_NORMALIZABLE_INDICES",
    "OFF_EXCHANGE_UNSIGNED_FEATURES",
    "OFF_EXCHANGE_SAFETY_GATES",
    "OFF_EXCHANGE_FEATURE_NAMES",
    # Canonical JSON + SHA-256 hashing (Phase 4 Batch 4c hardening)
    "canonical_json_blob",
    "sanitize_for_hash",
    "sha256_hex",
]

# -- Canonical hashing primitives (Phase 4 Batch 4c hardening, 2026-04-15) --
# Single source of truth for the canonical JSON + SHA-256 convention used
# across hft-ops ledger, hft-ops provenance, hft-ops feature_sets,
# hft-feature-evaluator, and (via inlined copy + parity test) lob-model-trainer.
from hft_contracts.canonical_hash import (
    canonical_json_blob,
    sanitize_for_hash,
    sha256_hex,
)
