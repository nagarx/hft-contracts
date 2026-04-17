"""Tests for LabelFactory — label computation from forward mid-price trajectories.

Each test validates formulas against hand-calculated expected values,
citing the exact Rust source that defines the authoritative formula.

Reference:
    feature-extractor-MBO-LOB/src/labeling/multi_horizon.rs (smoothed return)
    feature-extractor-MBO-LOB/src/labeling/magnitude.rs (point, mean, peak return)
"""

import numpy as np
import pytest

from hft_contracts.label_factory import (
    DIVISION_GUARD_EPS,
    ForwardPriceContract,
    LabelFactory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_forward_prices(prices: list, k: int = 0) -> np.ndarray:
    """Build a single-row forward_prices array from a list of prices.

    The list should contain prices at [t-k, t-k+1, ..., t, t+1, ..., t+H].
    Returns shape [1, len(prices)].
    """
    return np.array([prices], dtype=np.float64)


# ---------------------------------------------------------------------------
# ForwardPriceContract
# ---------------------------------------------------------------------------

class TestForwardPriceContract:
    def test_base_price_column(self):
        c = ForwardPriceContract(smoothing_window_offset=5, max_horizon=300, n_columns=306)
        assert c.base_price_column == 5

    def test_horizon_column(self):
        c = ForwardPriceContract(smoothing_window_offset=5, max_horizon=300, n_columns=306)
        assert c.horizon_column(0) == 5   # t+0 = t
        assert c.horizon_column(10) == 15  # t+10
        assert c.horizon_column(300) == 305  # t+300

    def test_horizon_column_exceeds_max(self):
        c = ForwardPriceContract(smoothing_window_offset=5, max_horizon=300, n_columns=306)
        with pytest.raises(ValueError, match="exceeds max_horizon"):
            c.horizon_column(301)

    def test_validate_shape_correct(self):
        c = ForwardPriceContract(smoothing_window_offset=5, max_horizon=10, n_columns=16)
        fwd = np.zeros((100, 16))
        c.validate_shape(fwd)  # Should not raise

    def test_validate_shape_wrong_columns(self):
        c = ForwardPriceContract(smoothing_window_offset=5, max_horizon=10, n_columns=16)
        fwd = np.zeros((100, 20))
        with pytest.raises(ValueError, match="columns"):
            c.validate_shape(fwd)

    def test_validate_shape_wrong_dims(self):
        c = ForwardPriceContract(smoothing_window_offset=5, max_horizon=10, n_columns=16)
        fwd = np.zeros((100,))
        with pytest.raises(ValueError, match="2D"):
            c.validate_shape(fwd)

    def test_from_metadata(self):
        metadata = {
            "forward_prices": {
                "exported": True,
                "smoothing_window_offset": 5,
                "max_horizon": 300,
                "n_columns": 306,
            }
        }
        c = ForwardPriceContract.from_metadata(metadata)
        assert c.smoothing_window_offset == 5
        assert c.max_horizon == 300
        assert c.n_columns == 306

    def test_from_metadata_not_exported(self):
        metadata = {"forward_prices": {"exported": False}}
        with pytest.raises(KeyError, match="not exported"):
            ForwardPriceContract.from_metadata(metadata)


# ---------------------------------------------------------------------------
# Smoothed Return
# ---------------------------------------------------------------------------

class TestSmoothedReturn:
    def test_hand_calculated_k2_h2(self):
        """Hand-calculated smoothed return with k=2, h=2.

        Prices at: [t-2, t-1, t, t+1, t+2]  (5 columns, k=2)
        Values:    [100, 101, 102, 103, 104]

        past_smooth  = mean([100, 101, 102]) = 101.0
        future_smooth = mean([102, 103, 104]) = 103.0  (t+h-k=t+0, t+h=t+2)

        Wait: h=2, k=2. future_smooth uses columns [h, h+k] = [2, 4] → [102, 103, 104]
        past_smooth uses columns [0, k] = [0, 2] → [100, 101, 102]

        return = (103.0 - 101.0) / 101.0 * 10000 = 198.0198... bps
        """
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        fwd = _make_forward_prices(prices)
        result = LabelFactory.smoothed_return(fwd, horizon=2, smoothing_window=2)

        past_smooth = np.mean([100.0, 101.0, 102.0])  # 101.0
        future_smooth = np.mean([102.0, 103.0, 104.0])  # 103.0
        expected_bps = (future_smooth - past_smooth) / past_smooth * 10000.0

        np.testing.assert_allclose(result[0], expected_bps, rtol=1e-12)

    def test_flat_prices_returns_zero(self):
        """Flat prices → zero return."""
        fwd = np.full((10, 20), 150.0)
        result = LabelFactory.smoothed_return(fwd, horizon=5, smoothing_window=3)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_batch_shape(self):
        """Output shape matches input N."""
        fwd = np.random.uniform(100, 200, size=(500, 20))
        result = LabelFactory.smoothed_return(fwd, horizon=5, smoothing_window=3)
        assert result.shape == (500,)

    def test_positive_trend_positive_return(self):
        """Upward trend → positive return (sign convention: >0 = bullish)."""
        prices = [100.0, 101.0, 102.0, 105.0, 110.0]
        fwd = _make_forward_prices(prices)
        result = LabelFactory.smoothed_return(fwd, horizon=2, smoothing_window=2)
        assert result[0] > 0, f"Expected positive, got {result[0]}"


# ---------------------------------------------------------------------------
# Point Return
# ---------------------------------------------------------------------------

class TestPointReturn:
    def test_hand_calculated(self):
        """(110 - 100) / 100 * 10000 = 1000 bps."""
        # k=2: columns [t-2, t-1, t, t+1, t+2]
        prices = [98.0, 99.0, 100.0, 105.0, 110.0]
        fwd = _make_forward_prices(prices)
        result = LabelFactory.point_return(fwd, horizon=2, smoothing_window=2)
        expected = (110.0 - 100.0) / 100.0 * 10000.0  # 1000.0 bps
        np.testing.assert_allclose(result[0], expected, rtol=1e-12)

    def test_negative_return(self):
        """Price decrease → negative return."""
        prices = [100.0, 100.0, 100.0, 95.0, 90.0]
        fwd = _make_forward_prices(prices)
        result = LabelFactory.point_return(fwd, horizon=2, smoothing_window=2)
        expected = (90.0 - 100.0) / 100.0 * 10000.0  # -1000.0 bps
        np.testing.assert_allclose(result[0], expected, rtol=1e-12)

    def test_zero_return(self):
        """Same price → zero return."""
        fwd = np.full((5, 10), 150.0)
        result = LabelFactory.point_return(fwd, horizon=3, smoothing_window=2)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Smoothed vs Point Divergence
# ---------------------------------------------------------------------------

class TestSmoothedVsPointDivergence:
    def test_non_linear_path_produces_different_results(self):
        """Non-monotonic price path: smoothed ≠ point.

        Prices: [100, 100, 100, 110, 90]  (k=2, h=2)
        Point return at h=2: (90 - 100) / 100 = -1000 bps
        Smoothed: past=mean([100,100,100])=100, future=mean([100,110,90])=100 → 0 bps

        These MUST differ — this is exactly the label-execution mismatch.
        """
        prices = [100.0, 100.0, 100.0, 110.0, 90.0]
        fwd = _make_forward_prices(prices)

        smoothed = LabelFactory.smoothed_return(fwd, horizon=2, smoothing_window=2)
        point = LabelFactory.point_return(fwd, horizon=2, smoothing_window=2)

        assert smoothed[0] != point[0], (
            f"Smoothed ({smoothed[0]:.2f}) should differ from "
            f"point ({point[0]:.2f}) on non-linear path"
        )

    def test_linear_path_produces_similar_results(self):
        """Perfectly linear path: smoothed ≈ point (not exactly equal due to formula)."""
        prices = [100.0, 102.0, 104.0, 106.0, 108.0]
        fwd = _make_forward_prices(prices)

        smoothed = LabelFactory.smoothed_return(fwd, horizon=2, smoothing_window=2)
        point = LabelFactory.point_return(fwd, horizon=2, smoothing_window=2)

        # Both should be positive and in the same direction
        assert smoothed[0] > 0
        assert point[0] > 0


# ---------------------------------------------------------------------------
# Mean Return
# ---------------------------------------------------------------------------

class TestMeanReturn:
    def test_hand_calculated(self):
        """mean([105, 110]) = 107.5; (107.5 - 100) / 100 * 10000 = 750 bps."""
        prices = [98.0, 99.0, 100.0, 105.0, 110.0]
        fwd = _make_forward_prices(prices)
        result = LabelFactory.mean_return(fwd, horizon=2, smoothing_window=2)
        future_mean = np.mean([105.0, 110.0])  # 107.5
        expected = (future_mean - 100.0) / 100.0 * 10000.0
        np.testing.assert_allclose(result[0], expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Peak Return
# ---------------------------------------------------------------------------

class TestPeakReturn:
    def test_selects_positive_dominant(self):
        """Max positive > |min negative| → returns positive peak."""
        # k=1: [t-1, t, t+1, t+2, t+3]
        prices = [99.0, 100.0, 105.0, 95.0, 103.0]
        fwd = _make_forward_prices(prices)
        result = LabelFactory.peak_return(fwd, horizon=3, smoothing_window=1)
        # max = 105 → ret = 500 bps, min = 95 → ret = -500 bps
        # |500| >= |-500| → dominant = +500 bps
        assert result[0] > 0

    def test_selects_negative_dominant(self):
        """|min negative| > max positive → returns negative peak."""
        prices = [99.0, 100.0, 101.0, 80.0, 99.0]
        fwd = _make_forward_prices(prices)
        result = LabelFactory.peak_return(fwd, horizon=3, smoothing_window=1)
        # max = 101 → ret = 100 bps, min = 80 → ret = -2000 bps
        # |-2000| > |100| → dominant = -2000 bps
        assert result[0] < 0


# ---------------------------------------------------------------------------
# Multi-Horizon
# ---------------------------------------------------------------------------

class TestMultiHorizon:
    def test_shape(self):
        fwd = np.random.uniform(100, 200, size=(50, 20))
        result = LabelFactory.multi_horizon(
            fwd, horizons=[2, 5, 10], smoothing_window=3, return_type="point_return",
        )
        assert result.shape == (50, 3)

    def test_invalid_return_type(self):
        fwd = np.random.uniform(100, 200, size=(10, 20))
        with pytest.raises(ValueError, match="Unknown return_type"):
            LabelFactory.multi_horizon(fwd, horizons=[2], smoothing_window=0, return_type="invalid")


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------

class TestClassify:
    def test_boundaries(self):
        """Threshold ± epsilon → correct classification."""
        threshold = 5.0
        eps = 1e-12
        returns = np.array([
            threshold + eps,   # UP
            threshold - eps,   # STABLE (just below)
            -threshold - eps,  # DOWN
            -threshold + eps,  # STABLE (just above)
            0.0,               # STABLE
        ])
        labels = LabelFactory.classify(returns, threshold_bps=threshold)
        expected = np.array([1, 0, -1, 0, 0], dtype=np.int8)
        np.testing.assert_array_equal(labels, expected)

    def test_multi_horizon_input(self):
        """Works with 2D input [N, H]."""
        returns = np.array([[10.0, -10.0], [0.0, 5.1]])
        labels = LabelFactory.classify(returns, threshold_bps=5.0)
        expected = np.array([[1, -1], [0, 1]], dtype=np.int8)
        np.testing.assert_array_equal(labels, expected)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_eps_guard_zero_base_price(self):
        """Base price = 0 → returns 0.0, not Inf."""
        prices = [0.0, 0.0, 0.0, 100.0, 200.0]
        fwd = _make_forward_prices(prices)
        result = LabelFactory.point_return(fwd, horizon=2, smoothing_window=2)
        assert np.isfinite(result[0])
        assert result[0] == 0.0

    def test_eps_guard_near_zero_base(self):
        """Near-zero base (below EPS) → returns 0.0."""
        tiny = DIVISION_GUARD_EPS / 10.0
        prices = [tiny, tiny, tiny, 100.0, 200.0]
        fwd = _make_forward_prices(prices)
        result = LabelFactory.smoothed_return(fwd, horizon=2, smoothing_window=2)
        assert np.isfinite(result[0])
        assert result[0] == 0.0

    def test_deterministic(self):
        """Same input → identical output across runs."""
        fwd = np.random.RandomState(42).uniform(100, 200, size=(100, 20))
        r1 = LabelFactory.smoothed_return(fwd, horizon=5, smoothing_window=3)
        r2 = LabelFactory.smoothed_return(fwd, horizon=5, smoothing_window=3)
        np.testing.assert_array_equal(r1, r2)

    def test_consistent_signature(self):
        """All return-type functions accept the same (fwd, h, k) signature."""
        fwd = np.random.uniform(100, 200, size=(10, 20))
        for fn_name in ["smoothed_return", "point_return", "mean_return", "peak_return"]:
            fn = getattr(LabelFactory, fn_name)
            result = fn(fwd, horizon=5, smoothing_window=3)
            assert result.shape == (10,), f"{fn_name} returned wrong shape"
            assert result.dtype == np.float64, f"{fn_name} returned wrong dtype"


# ---------------------------------------------------------------------------
# Input Validation (Pre-T9 regression tests — Bug B1)
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Pre-T9 regression tests: LabelFactory input hardening (Bug B1).

    Before this fix, out-of-bounds horizons silently produced NaN
    (smoothed_return, mean_return), uninformative IndexError (point_return),
    or ValueError from np.max on empty slice (peak_return).

    Production callers hit the primitives directly (not multi_horizon):
        - lob-dataset-analyzer/.../label_execution_mismatch.py:235-236
        - scripts/e8_model_diagnostic.py:149, 153, 222, 454
    """

    # -- multi_horizon dispatcher validation --

    def test_multi_horizon_horizon_exceeds_cols_raises(self):
        """Horizon far beyond array bounds must raise ValueError."""
        fp = np.zeros((10, 16))  # k=5, max_H=10 → 16 cols
        with pytest.raises(ValueError, match="exceeds forward_prices cols"):
            LabelFactory.multi_horizon(fp, horizons=[5000], smoothing_window=5)

    def test_multi_horizon_exactly_at_boundary_raises(self):
        """h + k == n_cols is out of bounds (needs h + k < n_cols)."""
        fp = np.zeros((10, 16))  # 16 cols
        # h=11, k=5 → h+k=16 == n_cols → invalid
        with pytest.raises(ValueError, match="exceeds forward_prices cols"):
            LabelFactory.multi_horizon(fp, horizons=[11], smoothing_window=5)

    def test_multi_horizon_one_below_boundary_succeeds(self):
        """h + k == n_cols - 1 is valid (last accessible column)."""
        rng = np.random.default_rng(42)
        fp = rng.uniform(100, 200, size=(10, 16))
        # h=10, k=5 → h+k=15 < 16 → valid
        result = LabelFactory.multi_horizon(fp, horizons=[10], smoothing_window=5)
        assert result.shape == (10, 1)
        assert np.isfinite(result).all()

    def test_multi_horizon_negative_horizon_raises(self):
        """Negative horizon must raise."""
        fp = np.zeros((10, 16))
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            LabelFactory.multi_horizon(fp, horizons=[-1], smoothing_window=5)

    def test_multi_horizon_zero_horizon_raises(self):
        """h=0 is degenerate (empty slice for mean/peak) and rejected."""
        fp = np.zeros((10, 16))
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            LabelFactory.multi_horizon(fp, horizons=[0], smoothing_window=5)

    def test_multi_horizon_empty_horizons_raises(self):
        """Empty horizons list must raise."""
        fp = np.zeros((10, 16))
        with pytest.raises(ValueError, match="horizons list is empty"):
            LabelFactory.multi_horizon(fp, horizons=[], smoothing_window=5)

    def test_multi_horizon_smoothing_window_too_large_raises(self):
        """smoothing_window >= n_cols must raise."""
        fp = np.zeros((10, 5))  # only 5 cols
        with pytest.raises(ValueError, match="smoothing_window .* must be < forward_prices"):
            LabelFactory.multi_horizon(fp, horizons=[1], smoothing_window=10)

    def test_multi_horizon_fp_not_2d_raises(self):
        """3D forward_prices must raise."""
        fp = np.zeros((10, 16, 1))
        with pytest.raises(ValueError, match="must be 2D"):
            LabelFactory.multi_horizon(fp, horizons=[1], smoothing_window=5)

    # -- primitive-level validation (covers production callers) --

    def test_smoothed_return_horizon_exceeds_raises(self):
        """smoothed_return with out-of-bounds horizon raises ValueError."""
        fp = np.zeros((10, 16))
        with pytest.raises(ValueError, match="exceeds forward_prices cols"):
            LabelFactory.smoothed_return(fp, horizon=5000, smoothing_window=5)

    def test_point_return_horizon_exceeds_raises(self):
        """point_return with out-of-bounds: now ValueError (was IndexError)."""
        fp = np.zeros((10, 16))
        with pytest.raises(ValueError, match="exceeds forward_prices cols"):
            LabelFactory.point_return(fp, horizon=5000, smoothing_window=5)

    def test_mean_return_horizon_exceeds_raises(self):
        """mean_return with out-of-bounds: now ValueError (was silent NaN)."""
        fp = np.zeros((10, 16))
        with pytest.raises(ValueError, match="exceeds forward_prices cols"):
            LabelFactory.mean_return(fp, horizon=5000, smoothing_window=5)

    def test_peak_return_horizon_exceeds_raises(self):
        """peak_return with out-of-bounds: now ValueError (was silent NaN/crash)."""
        fp = np.zeros((10, 16))
        with pytest.raises(ValueError, match="exceeds forward_prices cols"):
            LabelFactory.peak_return(fp, horizon=5000, smoothing_window=5)

    def test_primitive_negative_smoothing_window_raises(self):
        """Negative smoothing_window raises on any primitive."""
        fp = np.zeros((10, 16))
        with pytest.raises(ValueError, match="smoothing_window must be >= 0"):
            LabelFactory.smoothed_return(fp, horizon=1, smoothing_window=-1)

    # -- defense in depth --

    def test_multi_horizon_raises_on_inf_input(self):
        """forward_prices with Inf in future window triggers defense-in-depth.

        The EPS guard handles NaN base prices (returns 0.0), but Inf in the
        future window with valid base prices produces Inf results.
        """
        fp = np.full((10, 16), 100.0)
        fp[:, 6:] = np.inf  # Inf in future window
        # h=1, k=5: past=mean(100×6)=100, future=mean([100,100,100,100,100,Inf])=Inf
        # result = (Inf - 100) / 100 * 10000 = Inf → caught by isfinite check
        with pytest.raises(ValueError, match="non-finite values"):
            LabelFactory.multi_horizon(fp, horizons=[1], smoothing_window=5)
