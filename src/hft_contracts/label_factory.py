"""Label computation from forward mid-price trajectories.

All methods are pure functions operating on the forward_prices array
exported by the Rust feature extractor ({day}_forward_prices.npy).

Array layout (with smoothing_window_offset k):
    Column 0:     mid_price at t-k  (k events before sequence end)
    Column k:     mid_price at t    (base price / prediction point)
    Column k+h:   mid_price at t+h  (h events forward)

All return values are in basis points (return * 10000).

Reference: Matches Rust implementations in
    feature-extractor-MBO-LOB/src/labeling/multi_horizon.rs (lines 1072-1098)
    feature-extractor-MBO-LOB/src/labeling/magnitude.rs (lines 50-132)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

# Division guard — matches Rust contract.rs DIVISION_GUARD_EPS (1e-8)
DIVISION_GUARD_EPS = 1e-8


def _validate_fp_horizon_k(
    forward_prices: np.ndarray,
    horizon: int,
    smoothing_window: int,
) -> None:
    """Validate forward_prices shape and horizon/smoothing_window bounds.

    The universal predicate ``horizon + smoothing_window < n_cols`` works for
    all four return types:

        - smoothed_return: ``fp[:, h : h+k+1]``  needs ``h+k+1 <= n_cols``
        - point_return:    ``fp[:, k+h]``         needs ``k+h   <  n_cols``
        - mean_return:     ``fp[:, k+1 : k+h+1]`` needs ``k+h+1 <= n_cols``
        - peak_return:     ``fp[:, k+1 : k+h+1]`` needs ``k+h+1 <= n_cols``

    All four collapse to: ``k + h <= n_cols - 1``, i.e. ``k + h < n_cols``.

    Raises:
        ValueError: With a message naming the offending value and the legal bound.
    """
    if forward_prices.ndim != 2:
        raise ValueError(
            f"forward_prices must be 2D [N, cols], got shape {forward_prices.shape}"
        )
    n_cols = forward_prices.shape[1]
    if n_cols == 0:
        raise ValueError(
            f"forward_prices has 0 columns, shape={forward_prices.shape}"
        )
    if smoothing_window < 0:
        raise ValueError(
            f"smoothing_window must be >= 0, got {smoothing_window}"
        )
    if smoothing_window >= n_cols:
        raise ValueError(
            f"smoothing_window {smoothing_window} must be < forward_prices "
            f"cols {n_cols}"
        )
    if horizon < 1:
        raise ValueError(
            f"horizon must be >= 1, got {horizon}"
        )
    if horizon + smoothing_window >= n_cols:
        max_h = n_cols - smoothing_window - 1
        raise ValueError(
            f"horizon {horizon} + smoothing_window {smoothing_window} "
            f"= {horizon + smoothing_window} exceeds forward_prices cols "
            f"{n_cols} (max valid horizon for k={smoothing_window}: {max_h})"
        )


@dataclass(frozen=True)
class ForwardPriceContract:
    """Metadata contract for the forward_prices array.

    Parsed from the 'forward_prices' section of {day}_metadata.json.
    Defines the column layout so consumers know which column corresponds
    to which time offset.

    Attributes:
        smoothing_window_offset: k — column index of the base price at t.
            Columns [0, k) contain past prices at t-k, t-k+1, ..., t-1.
        max_horizon: H — maximum forward horizon in events.
            Column k+H contains mid_price at t+H.
        n_columns: Total columns = k + H + 1.
        units: Price unit (always "USD" for raw mid_prices).
    """

    smoothing_window_offset: int
    max_horizon: int
    n_columns: int
    units: str = "USD"

    def __post_init__(self) -> None:
        expected = self.smoothing_window_offset + self.max_horizon + 1
        if self.n_columns != expected:
            raise ValueError(
                f"ForwardPriceContract invariant violated: "
                f"n_columns ({self.n_columns}) != "
                f"smoothing_window_offset ({self.smoothing_window_offset}) + "
                f"max_horizon ({self.max_horizon}) + 1 = {expected}"
            )

    @property
    def base_price_column(self) -> int:
        """Column index of the base price at time t (sequence end)."""
        return self.smoothing_window_offset

    def horizon_column(self, h: int) -> int:
        """Column index for mid_price at t+h events forward.

        Args:
            h: Forward horizon in events.

        Returns:
            Column index into the forward_prices array.

        Raises:
            ValueError: If h > max_horizon.
        """
        if h > self.max_horizon:
            raise ValueError(
                f"Horizon {h} exceeds max_horizon {self.max_horizon}"
            )
        return self.smoothing_window_offset + h

    def validate_shape(self, forward_prices: np.ndarray) -> None:
        """Validate that forward_prices array matches this contract.

        Args:
            forward_prices: Array to validate.

        Raises:
            ValueError: If shape doesn't match contract.
        """
        if forward_prices.ndim != 2:
            raise ValueError(
                f"forward_prices must be 2D, got {forward_prices.ndim}D"
            )
        if forward_prices.shape[1] != self.n_columns:
            raise ValueError(
                f"forward_prices has {forward_prices.shape[1]} columns, "
                f"contract expects {self.n_columns}"
            )

    @classmethod
    def from_metadata(cls, metadata: dict) -> "ForwardPriceContract":
        """Parse from {day}_metadata.json forward_prices section.

        Args:
            metadata: Parsed metadata JSON dict.

        Returns:
            ForwardPriceContract instance.

        Raises:
            KeyError: If forward_prices section is missing or incomplete.
        """
        fp = metadata["forward_prices"]
        if not fp.get("exported", False):
            raise KeyError(
                "forward_prices not exported (exported=false in metadata)"
            )
        return cls(
            smoothing_window_offset=fp["smoothing_window_offset"],
            max_horizon=fp["max_horizon"],
            n_columns=fp["n_columns"],
        )


class LabelFactory:
    """Compute labels from forward mid-price trajectories.

    All methods are static pure functions with no side effects.
    Thread-safe, deterministic, no random state.

    The forward_prices array has shape [N, k + max_H + 1] where:
        - k = smoothing_window (number of past price columns)
        - max_H = maximum forward horizon
        - Column k = base price at time t (sequence end / prediction point)

    Usage:
        fwd = np.load("{day}_forward_prices.npy")  # [N, k + max_H + 1]
        smoothed = LabelFactory.smoothed_return(fwd, horizon=10, smoothing_window=5)
        point = LabelFactory.point_return(fwd, horizon=10, smoothing_window=5)
        # Compare: smoothed vs point on identical samples
    """

    EPS = DIVISION_GUARD_EPS

    @staticmethod
    def smoothed_return(
        forward_prices: np.ndarray,
        horizon: int,
        smoothing_window: int,
    ) -> np.ndarray:
        """TLOB smoothed-average return in basis points.

        Formula (Rust: multi_horizon.rs lines 1072-1098):
            past_smooth  = mean(prices[t-k : t+1])     → k+1 prices
            future_smooth = mean(prices[t+h-k : t+h+1]) → k+1 prices
            return_bps = (future_smooth - past_smooth) / past_smooth * 10000

        In forward_prices array (offset k = smoothing_window):
            past_smooth  = mean(forward_prices[:, 0 : k+1])
            future_smooth = mean(forward_prices[:, h : h+k+1])

        Args:
            forward_prices: [N, k + max_H + 1] float64 USD prices.
            horizon: Forward horizon h (events).
            smoothing_window: Smoothing parameter k.

        Returns:
            [N] float64 — smoothed returns in basis points.

        Raises:
            ValueError: If inputs are malformed or horizon exceeds array bounds.
        """
        _validate_fp_horizon_k(forward_prices, horizon, smoothing_window)
        k = smoothing_window
        h = horizon
        # Past smoothed: columns [0, k] → prices at t-k, ..., t (k+1 terms)
        past_smooth = np.mean(forward_prices[:, 0 : k + 1], axis=1)
        # Future smoothed: columns [h, h+k] → prices at t+h-k, ..., t+h (k+1 terms)
        future_smooth = np.mean(forward_prices[:, h : h + k + 1], axis=1)
        # Return in bps with division guard (avoid RuntimeWarning)
        safe_past = np.where(np.abs(past_smooth) > DIVISION_GUARD_EPS, past_smooth, 1.0)
        result = (future_smooth - past_smooth) / safe_past * 10000.0
        return np.where(np.abs(past_smooth) > DIVISION_GUARD_EPS, result, 0.0)

    @staticmethod
    def point_return(
        forward_prices: np.ndarray,
        horizon: int,
        smoothing_window: int,
    ) -> np.ndarray:
        """Point-to-point return in basis points.

        Formula (Rust: magnitude.rs ~line 60):
            return_bps = (price[t+h] - price[t]) / price[t] * 10000

        In forward_prices array (offset k = smoothing_window):
            base = forward_prices[:, k]        → price at t
            future = forward_prices[:, k + h]  → price at t+h

        Args:
            forward_prices: [N, k + max_H + 1] float64 USD prices.
            horizon: Forward horizon h (events).
            smoothing_window: Offset k to locate base price column.

        Returns:
            [N] float64 — point-to-point returns in basis points.

        Raises:
            ValueError: If inputs are malformed or horizon exceeds array bounds.
        """
        _validate_fp_horizon_k(forward_prices, horizon, smoothing_window)
        k = smoothing_window
        base = forward_prices[:, k]
        future = forward_prices[:, k + horizon]
        # Use safe_base to avoid RuntimeWarning on divide-by-zero
        safe_base = np.where(np.abs(base) > DIVISION_GUARD_EPS, base, 1.0)
        result = (future - base) / safe_base * 10000.0
        # Zero out where base was too small
        return np.where(np.abs(base) > DIVISION_GUARD_EPS, result, 0.0)

    @staticmethod
    def mean_return(
        forward_prices: np.ndarray,
        horizon: int,
        smoothing_window: int,
    ) -> np.ndarray:
        """Mean forward return in basis points.

        Formula (Rust: magnitude.rs ~line 70):
            return_bps = (mean(price[t+1:t+h+1]) / price[t] - 1) * 10000

        In forward_prices array (offset k = smoothing_window):
            base = forward_prices[:, k]
            future_mean = mean(forward_prices[:, k+1 : k+h+1])

        Args:
            forward_prices: [N, k + max_H + 1] float64 USD prices.
            horizon: Forward horizon h (events).
            smoothing_window: Offset k to locate base price column.

        Returns:
            [N] float64 — mean forward returns in basis points.

        Raises:
            ValueError: If inputs are malformed or horizon exceeds array bounds.
        """
        _validate_fp_horizon_k(forward_prices, horizon, smoothing_window)
        k = smoothing_window
        base = forward_prices[:, k]
        future_mean = np.mean(forward_prices[:, k + 1 : k + horizon + 1], axis=1)
        safe_base = np.where(np.abs(base) > DIVISION_GUARD_EPS, base, 1.0)
        result = (future_mean - base) / safe_base * 10000.0
        return np.where(np.abs(base) > DIVISION_GUARD_EPS, result, 0.0)

    @staticmethod
    def peak_return(
        forward_prices: np.ndarray,
        horizon: int,
        smoothing_window: int,
    ) -> np.ndarray:
        """Peak return (maximum absolute return within horizon) in basis points.

        Formula (Rust: magnitude.rs ~line 80):
            max_ret = (max(price[t+1:t+h+1]) - price[t]) / price[t]
            min_ret = (min(price[t+1:t+h+1]) - price[t]) / price[t]
            return = max_ret if |max_ret| >= |min_ret| else min_ret
            return_bps = return * 10000

        Args:
            forward_prices: [N, k + max_H + 1] float64 USD prices.
            horizon: Forward horizon h (events).
            smoothing_window: Offset k to locate base price column.

        Returns:
            [N] float64 — peak returns in basis points (signed, dominant direction).

        Raises:
            ValueError: If inputs are malformed or horizon exceeds array bounds.
        """
        _validate_fp_horizon_k(forward_prices, horizon, smoothing_window)
        k = smoothing_window
        base = forward_prices[:, k]
        future_window = forward_prices[:, k + 1 : k + horizon + 1]
        max_prices = np.max(future_window, axis=1)
        min_prices = np.min(future_window, axis=1)

        safe_base = np.where(np.abs(base) > DIVISION_GUARD_EPS, base, 1.0)
        max_ret = (max_prices - base) / safe_base
        min_ret = (min_prices - base) / safe_base

        # Pick whichever has larger absolute magnitude
        dominant = np.where(np.abs(max_ret) >= np.abs(min_ret), max_ret, min_ret)
        # Zero out where base was too small
        dominant = np.where(np.abs(base) > DIVISION_GUARD_EPS, dominant, 0.0)
        return dominant * 10000.0

    @staticmethod
    def multi_horizon(
        forward_prices: np.ndarray,
        horizons: List[int],
        smoothing_window: int,
        return_type: str = "smoothed_return",
    ) -> np.ndarray:
        """Compute labels at multiple horizons.

        Args:
            forward_prices: [N, k + max_H + 1] float64 USD prices.
            horizons: List of forward horizons [h1, h2, ...]. Must be
                non-empty, all integers >= 1, with
                max(horizons) + smoothing_window < n_cols.
            smoothing_window: Smoothing parameter / offset k.
            return_type: One of "smoothed_return", "point_return",
                "mean_return", "peak_return".

        Returns:
            [N, len(horizons)] float64 — returns in basis points per horizon.

        Raises:
            ValueError: If inputs are malformed, bounds are exceeded, or
                the computation produces non-finite values (defense in depth).
        """
        if not horizons:
            raise ValueError("horizons list is empty")
        if not all(isinstance(h, (int, np.integer)) for h in horizons):
            raise ValueError(
                f"horizons must all be integers, got types "
                f"{[type(h).__name__ for h in horizons]}"
            )

        fn = _RETURN_FUNCTIONS.get(return_type)
        if fn is None:
            raise ValueError(
                f"Unknown return_type '{return_type}'. "
                f"Valid: {list(_RETURN_FUNCTIONS.keys())}"
            )

        # Validate worst-case horizon upfront for a clear top-level error
        _validate_fp_horizon_k(forward_prices, max(horizons), smoothing_window)
        # Each horizon must also pass the h >= 1 check (max only catches largest)
        for h in horizons:
            if h < 1:
                raise ValueError(
                    f"all horizons must be >= 1, got {h} in {horizons}"
                )

        # Each primitive revalidates defensively; cost is negligible
        results = [
            fn(forward_prices, h, smoothing_window) for h in horizons
        ]
        out = np.column_stack(results)

        # Defense in depth: catch non-finite from NaN/Inf in forward_prices
        if not np.isfinite(out).all():
            nan_count = int(np.isnan(out).sum())
            inf_count = int(np.isinf(out).sum())
            raise ValueError(
                f"multi_horizon produced non-finite values ({nan_count} NaN, "
                f"{inf_count} Inf) with horizons={horizons}, "
                f"k={smoothing_window}, fp.shape={forward_prices.shape}. "
                f"Check forward_prices for NaN/Inf entries."
            )

        return out

    @staticmethod
    def classify(
        returns_bps: np.ndarray,
        threshold_bps: float,
    ) -> np.ndarray:
        """Classify continuous returns into {-1, 0, +1} TLOB labels.

        Formula: +1 if return > threshold, -1 if return < -threshold, 0 otherwise.

        Matches Rust TlobLabelGenerator classification.

        Args:
            returns_bps: [N] or [N, H] float64 returns in basis points.
            threshold_bps: Classification boundary in basis points.

        Returns:
            Same shape as input, int8, values in {-1, 0, +1}.
        """
        labels = np.zeros_like(returns_bps, dtype=np.int8)
        labels[returns_bps > threshold_bps] = 1
        labels[returns_bps < -threshold_bps] = -1
        return labels


# Dispatch table for return type functions.
# All share the same signature: (forward_prices, horizon, smoothing_window) -> np.ndarray
_RETURN_FUNCTIONS = {
    "smoothed_return": LabelFactory.smoothed_return,
    "point_return": LabelFactory.point_return,
    "mean_return": LabelFactory.mean_return,
    "peak_return": LabelFactory.peak_return,
}
