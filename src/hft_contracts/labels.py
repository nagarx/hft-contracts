"""
Labeling strategy contracts for the HFT pipeline.

Provides a unified abstraction over different labeling schemes, enabling
strategy-aware analysis without hardcoded label values.

Supported strategies:
    Classification:
    - TLOB: {-1, 0, 1} = Down, Stable, Up (trend-based)
    - TRIPLE_BARRIER: {0, 1, 2} = StopLoss, Timeout, ProfitTarget (event-based)
    - OPPORTUNITY: {-1, 0, 1} = BigDown, NoOpportunity, BigUp (opportunity-based)

    Regression:
    - REGRESSION: continuous float64 bps returns at each horizon

Design principles (RULE.md §1):
    - Single source of truth for label encoding contracts
    - Explicit contracts: every strategy has a frozen contract
    - Forward-compatible: new strategies can be added without breaking existing code
    - Classification and regression use separate contract types (no optional fields)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Final, Tuple, Union


# =========================================================================
# Label Encoding Constants
# =========================================================================

LABEL_DOWN: Final[int] = -1
"""Price moved down (bearish)."""

LABEL_STABLE: Final[int] = 0
"""Price stayed within threshold (neutral)."""

LABEL_UP: Final[int] = 1
"""Price moved up (bullish)."""

NUM_CLASSES: Final[int] = 3
"""Number of label classes for classification."""

LABEL_NAMES: Final[Dict[int, str]] = {
    LABEL_DOWN: "Down",
    LABEL_STABLE: "Stable",
    LABEL_UP: "Up",
}
"""Human-readable label names (original encoding: {-1, 0, 1})."""

# =========================================================================
# Shifted Label Encoding (for PyTorch CrossEntropyLoss)
# =========================================================================
# PyTorch CrossEntropyLoss requires labels in {0, 1, ..., num_classes-1}.
# Shift: original + 1 = shifted
#   -1 (Down)   -> 0
#    0 (Stable) -> 1
#   +1 (Up)     -> 2

SHIFTED_LABEL_DOWN: Final[int] = 0
"""Down label after shift (original: -1)."""

SHIFTED_LABEL_STABLE: Final[int] = 1
"""Stable label after shift (original: 0)."""

SHIFTED_LABEL_UP: Final[int] = 2
"""Up label after shift (original: +1)."""

SHIFTED_LABEL_NAMES: Final[Dict[int, str]] = {
    SHIFTED_LABEL_DOWN: "Down",
    SHIFTED_LABEL_STABLE: "Stable",
    SHIFTED_LABEL_UP: "Up",
}
"""Human-readable label names (shifted encoding: {0, 1, 2})."""


def get_label_name(label: int, shifted: bool = False) -> str:
    """
    Get human-readable name for a label value.

    Args:
        label: Label value (-1/0/1 for original, 0/1/2 for shifted)
        shifted: True if using shifted encoding (PyTorch), False for original

    Returns:
        Label name: "Down", "Stable", or "Up"
    """
    mapping = SHIFTED_LABEL_NAMES if shifted else LABEL_NAMES
    return mapping.get(label, str(label))


# =========================================================================
# Strategy Enum and Contract Dataclass
# =========================================================================


class LabelingStrategy(Enum):
    """Labeling strategy identifier."""
    TLOB = "tlob"
    TRIPLE_BARRIER = "triple_barrier"
    OPPORTUNITY = "opportunity"
    REGRESSION = "regression"


@dataclass(frozen=True)
class LabelContract:
    """
    Immutable contract defining a labeling strategy's encoding.

    Attributes:
        strategy: Which labeling strategy this contract represents
        class_names: Mapping from label value to human-readable name
        num_classes: Number of distinct classes
        value_range: (min_value, max_value) inclusive range of valid label values
        shift_for_crossentropy: Whether labels need +1 shift for CrossEntropyLoss
    """
    strategy: LabelingStrategy
    class_names: Dict[int, str]
    num_classes: int
    value_range: Tuple[int, int]
    shift_for_crossentropy: bool

    @property
    def values(self) -> Tuple[int, ...]:
        """All valid label values in sorted order."""
        return tuple(sorted(self.class_names.keys()))

    def class_name(self, value: int) -> str:
        """Get human-readable name for a label value."""
        return self.class_names.get(value, f"Unknown({value})")

    def is_valid(self, value: int) -> bool:
        """Check if a label value is valid for this contract."""
        return self.value_range[0] <= value <= self.value_range[1]


@dataclass(frozen=True)
class RegressionLabelContract:
    """
    Immutable contract for regression (continuous) labeling strategies.

    Separate from LabelContract because regression has no discrete classes,
    no class_names, no value_range, and no shift logic. Keeping them as
    distinct types prevents accidental misuse (RULE.md SS1: explicit contracts).

    Attributes:
        strategy: Always LabelingStrategy.REGRESSION
        encoding: Label encoding identifier (e.g. "continuous_bps")
        dtype: NumPy dtype string (e.g. "float64")
        unit: Physical unit of the label values (e.g. "basis_points")
        file_pattern: Expected filename pattern for label files
        shape_description: Human-readable shape description
    """
    strategy: LabelingStrategy
    encoding: str
    dtype: str
    unit: str
    file_pattern: str
    shape_description: str


def is_regression_strategy(strategy: str) -> bool:
    """Check if a strategy name refers to regression labeling."""
    return strategy.lower().strip() == "regression"


# =========================================================================
# Canonical Contracts (frozen singletons)
# =========================================================================

TLOB_CONTRACT: Final[LabelContract] = LabelContract(
    strategy=LabelingStrategy.TLOB,
    class_names={-1: "Down", 0: "Stable", 1: "Up"},
    num_classes=3,
    value_range=(-1, 1),
    shift_for_crossentropy=True,
)
"""TLOB labeling: {-1: Down, 0: Stable, 1: Up}. Requires +1 shift."""

TB_CONTRACT: Final[LabelContract] = LabelContract(
    strategy=LabelingStrategy.TRIPLE_BARRIER,
    class_names={0: "StopLoss", 1: "Timeout", 2: "ProfitTarget"},
    num_classes=3,
    value_range=(0, 2),
    shift_for_crossentropy=False,
)
"""Triple Barrier labeling: {0: StopLoss, 1: Timeout, 2: ProfitTarget}. No shift needed."""

OPPORTUNITY_CONTRACT: Final[LabelContract] = LabelContract(
    strategy=LabelingStrategy.OPPORTUNITY,
    class_names={-1: "BigDown", 0: "NoOpportunity", 1: "BigUp"},
    num_classes=3,
    value_range=(-1, 1),
    shift_for_crossentropy=True,
)
"""Opportunity labeling: {-1: BigDown, 0: NoOpportunity, 1: BigUp}. Requires +1 shift."""

REGRESSION_CONTRACT: Final[RegressionLabelContract] = RegressionLabelContract(
    strategy=LabelingStrategy.REGRESSION,
    encoding="continuous_bps",
    dtype="float64",
    unit="basis_points",
    file_pattern="{day}_regression_labels.npy",
    shape_description="[N, num_horizons]",
)
"""Regression labeling: continuous float64 forward returns in basis points."""

_CONTRACTS: Dict[str, Union[LabelContract, RegressionLabelContract]] = {
    "tlob": TLOB_CONTRACT,
    "trend": TLOB_CONTRACT,
    "triple_barrier": TB_CONTRACT,
    "opportunity": OPPORTUNITY_CONTRACT,
    "regression": REGRESSION_CONTRACT,
}


def get_contract(strategy: str) -> Union[LabelContract, RegressionLabelContract]:
    """
    Look up a label contract by strategy name.

    Args:
        strategy: One of "tlob", "triple_barrier", "opportunity", "regression"

    Returns:
        LabelContract for classification strategies, or
        RegressionLabelContract for regression.

    Raises:
        ValueError: If strategy is not recognized.
    """
    key = strategy.lower().strip()
    if key not in _CONTRACTS:
        raise ValueError(
            f"Unknown labeling strategy '{strategy}'. "
            f"Valid: {list(_CONTRACTS.keys())}"
        )
    return _CONTRACTS[key]
