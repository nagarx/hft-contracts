# hft-contracts

Single source of truth for cross-module data contracts in the HFT pipeline.

All Python modules (lobtrainer, lobanalyzer, lobbacktest) depend on this package
for feature indices, label contracts, and validation utilities. The constants are
auto-generated from `contracts/pipeline_contract.toml`.

## Installation

```bash
# In each module's venv:
uv pip install -e ../hft-contracts/
# or
pip install -e ../hft-contracts/
```

## Usage

```python
from hft_contracts import (
    # Feature indices
    FeatureIndex, ExperimentalFeatureIndex, SignalIndex,
    FEATURE_COUNT, FULL_FEATURE_COUNT, SCHEMA_VERSION,
    # Slices
    LOB_ALL, MBO_ALL, SIGNALS_ALL, EXPERIMENTAL_ALL,
    # Layout for normalization
    GROUPED_PRICE_INDICES, GROUPED_SIZE_INDICES,
    NON_NORMALIZABLE_INDICES,
    # Classification
    CATEGORICAL_INDICES, UNSIGNED_FEATURES, SAFETY_GATES,
    # Export metadata contract
    EXPORT_METADATA_REQUIRED_FIELDS,
    EXPORT_METADATA_NORMALIZATION_FIELDS,
    EXPORT_METADATA_PROVENANCE_FIELDS,
    EXPORT_MANIFEST_REQUIRED_FIELDS,
    # Labels
    TLOB_CONTRACT, TB_CONTRACT, OPPORTUNITY_CONTRACT,
    get_contract, get_label_name,
    LABEL_DOWN, LABEL_STABLE, LABEL_UP, NUM_CLASSES,
    # Validation
    ContractError, validate_export_contract,
    validate_schema_version, validate_normalization_not_applied,
    validate_metadata_completeness, validate_label_encoding,
    validate_provenance_present,
)
```

## Package structure

```
src/hft_contracts/
    __init__.py         # Re-exports all public symbols
    _generated.py       # AUTO-GENERATED (never edit manually)
    labels.py           # LabelingStrategy, LabelContract, canonical contracts
    validation.py       # ContractError, validation utilities
```

## Regeneration

After editing `contracts/pipeline_contract.toml`:

```bash
python contracts/generate_python_contract.py          # Regenerate
python contracts/generate_python_contract.py --check  # CI: exit 1 if stale
```

## Testing

```bash
pytest hft-contracts/tests/ -v
```

82 tests in total: 52 self-consistency tests (`test_contract_self_consistency.py`)
verify enum counts, slice ranges, categorical indices, label contracts, and name
dictionaries; 30 validation gate tests (`test_validation_gates.py`) verify the
validation utilities against edge cases.

## Validation

Validation utilities should be called at dataset load time, once per dataset:

| Function | Purpose |
|----------|---------|
| `validate_export_contract` | Validates feature count and schema version match the contract |
| `validate_schema_version` | Validates schema version compatibility |
| `validate_normalization_not_applied` | Ensures raw data has not been pre-normalized |
| `validate_metadata_completeness` | Checks export metadata has required fields |
| `validate_label_encoding` | Validates label encoding matches the labeling strategy |
| `validate_provenance_present` | Ensures provenance fields are present for reproducibility |

## Label Factory

Compute any label type from forward mid-price trajectories (`{day}_forward_prices.npy`):

```python
from hft_contracts import LabelFactory, ForwardPriceContract
import numpy as np

# Load forward prices (exported by Rust feature extractor)
fwd = np.load("20251114_forward_prices.npy")  # [N, k+max_H+1] float64

# Compute different label types from the SAME prices
smoothed = LabelFactory.smoothed_return(fwd, horizon=10, smoothing_window=10)
point = LabelFactory.point_return(fwd, horizon=10, smoothing_window=10)
peak = LabelFactory.peak_return(fwd, horizon=10, smoothing_window=10)

# Multi-horizon labels
multi = LabelFactory.multi_horizon(fwd, horizons=[10, 60, 300], smoothing_window=10)

# Classify continuous returns into {-1, 0, +1}
classes = LabelFactory.classify(smoothed, threshold_bps=8.0)

# Parse forward_prices metadata from export JSON
contract = ForwardPriceContract.from_metadata(metadata_dict)
contract.validate_shape(fwd)  # Raises ValueError if shape mismatch
```

All methods are pure static functions — thread-safe, deterministic, no state.
Cross-validated against Rust labels: max diff = 7.56e-12.

28 tests in `tests/test_label_factory.py`.

---

Last updated: March 17, 2026
