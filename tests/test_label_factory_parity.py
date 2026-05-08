"""CR6 / NEW-CRITICAL-2 — Python ↔ Rust LabelFactory parity test.

This test consumes a fixture produced by the Rust integration test at
``feature-extractor-MBO-LOB/crates/hft-labeling/tests/parity_fixture_gen.rs``
and asserts that Python's ``LabelFactory.{smoothed,point,mean,peak}_return``
produces the same labels as Rust's ``MultiHorizonLabelGenerator`` and
``MagnitudeGenerator`` to within f64 rounding tolerance.

Closes the long-standing gap that root ``CLAUDE.md`` cited "max diff 7.56e-12"
between the two implementations without an executable test backing the claim.

Architecture:
    1. Rust regenerates the fixture on demand:
       ``cargo test -p hft-labeling --test parity_fixture_gen -- --ignored --nocapture``
    2. The fixture is a checked-in JSON at
       ``hft-contracts/tests/fixtures/python_rust_parity/v1/labels.json``.
    3. This test loads the fixture and parametrizes over scenarios × 4
       ReturnType variants × horizons.
    4. Rust outputs are RAW DECIMAL fractions; this test multiplies by
       10000.0 before comparing to Python's basis-points outputs.

Tolerance: ``rtol=1e-12, atol=1e-15`` matches existing convention at
``hft-contracts/tests/test_label_factory.py:118`` for hand-calculated tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from hft_contracts.label_factory import LabelFactory


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "python_rust_parity" / "v1" / "labels.json"

# Fixture schema version — bump in lockstep with the Rust generator's
# ``schema_version: 1`` field. A mismatch indicates fixture/test drift.
EXPECTED_SCHEMA_VERSION = 1

# Tolerance — matches existing ``test_label_factory.py:118`` convention for
# the hand-calculated formula tests. f64 rounding noise is ~1e-15; the
# CLAUDE.md historical "7.56e-12" max-diff observation is ~30 ulps near 1.0,
# expected for sum-of-(k+1) differences.
PARITY_RTOL = 1e-12
PARITY_ATOL = 1e-15


@pytest.fixture(scope="module")
def fixture_data() -> Dict[str, Any]:
    """Load the checked-in JSON fixture once per test module.

    Skips with actionable message if the fixture is missing — operator runs
    the Rust regeneration command (cited in the skip message) to produce it.
    """
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"Parity fixture not found at {FIXTURE_PATH}. "
            f"Regenerate via: cd feature-extractor-MBO-LOB && "
            f"cargo test -p hft-labeling --test parity_fixture_gen "
            f"-- --ignored --nocapture"
        )
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["schema_version"] == EXPECTED_SCHEMA_VERSION, (
        f"Fixture schema_version drift: expected {EXPECTED_SCHEMA_VERSION}, "
        f"got {data['schema_version']}. Update EXPECTED_SCHEMA_VERSION in this "
        f"test AND verify that the Rust generator's schema bump is "
        f"backward-compatible with the test's parsing logic."
    )
    return data


def _scenarios(fixture_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract scenario list with at-least-one assertion."""
    scenarios = fixture_data["scenarios"]
    assert len(scenarios) >= 1, "Fixture must contain at least one scenario"
    return scenarios


def _scenario_horizon_pairs(scenarios: List[Dict[str, Any]]) -> List[tuple]:
    """Generate (scenario_idx, horizon) parametrize ids."""
    pairs = []
    for s_idx, s in enumerate(scenarios):
        for h in s["horizons"]:
            pairs.append((s_idx, h))
    return pairs


class TestProvenance:
    """Fixture provenance + format invariants."""

    def test_schema_version_matches(self, fixture_data: Dict[str, Any]) -> None:
        # Asserted in fixture loader; this test surfaces it explicitly for CI grep.
        assert fixture_data["schema_version"] == EXPECTED_SCHEMA_VERSION

    def test_rust_commit_recorded(self, fixture_data: Dict[str, Any]) -> None:
        """Rust commit SHA must be present (non-empty) for traceability.

        Empty string is allowed (rust generator runs outside a git checkout)
        but a missing key indicates fixture-format breakage.
        """
        assert "rust_commit" in fixture_data
        assert isinstance(fixture_data["rust_commit"], str)

    def test_at_least_one_scenario(self, fixture_data: Dict[str, Any]) -> None:
        assert len(fixture_data["scenarios"]) >= 1

    def test_scenario_invariants(self, fixture_data: Dict[str, Any]) -> None:
        """Each scenario must satisfy: total = n_samples + 2k + max_h.

        This is the price-stream-length invariant required for both Rust
        generators (multi_horizon needs 2k+max_h padding around n_samples
        valid t indices; magnitude with smoothing_window=None needs max_h).
        """
        for s in _scenarios(fixture_data):
            n = s["n_samples"]
            k = s["k"]
            max_h = max(s["horizons"])
            expected_total = n + 2 * k + max_h
            assert s["prices_total"] == expected_total, (
                f"Scenario '{s['name']}': prices_total={s['prices_total']} "
                f"!= n_samples({n}) + 2k({k}) + max_h({max_h}) = {expected_total}"
            )
            assert len(s["prices_flat"]) == expected_total, (
                f"Scenario '{s['name']}': prices_flat length mismatch"
            )
            assert len(s["forward_prices_2d"]) == n, (
                f"Scenario '{s['name']}': forward_prices_2d row count mismatch"
            )
            assert all(len(row) == k + max_h + 1 for row in s["forward_prices_2d"]), (
                f"Scenario '{s['name']}': forward_prices_2d col count mismatch"
            )


class TestParity:
    """Cross-language formula parity for all 4 ReturnType variants.

    Each test:
        1. Loads scenario inputs (forward_prices_2d, k, horizon)
        2. Runs Python LabelFactory.{return_type}(forward_prices_2d, h, k)
        3. Reads Rust output (raw decimals) for the same horizon
        4. Multiplies Rust output × 10000.0 to convert to basis points
        5. Asserts np.allclose(rust_bps, python_bps, rtol, atol)
    """

    def _run_method(
        self,
        method_name: str,
        scenario: Dict[str, Any],
        horizon: int,
    ) -> tuple:
        """Run Python LabelFactory method + load Rust output for parametrize.

        Returns (rust_bps, python_bps) — both `[n_samples]` float64 numpy arrays
        in basis points.
        """
        forward_prices = np.asarray(scenario["forward_prices_2d"], dtype=np.float64)
        k = scenario["k"]
        rust_decimal_key = f"h{horizon}"
        rust_decimal = np.asarray(
            scenario["rust_labels_decimal"][method_name][rust_decimal_key],
            dtype=np.float64,
        )
        # Rust → bps via × 10000.0 (Wave4 finding: Python returns bps, Rust raw).
        rust_bps = rust_decimal * 10000.0

        # Dispatch on Python LabelFactory method.
        if method_name == "smoothed_return":
            python_bps = LabelFactory.smoothed_return(
                forward_prices, horizon=horizon, smoothing_window=k
            )
        elif method_name == "point_return":
            python_bps = LabelFactory.point_return(
                forward_prices, horizon=horizon, smoothing_window=k
            )
        elif method_name == "mean_return":
            python_bps = LabelFactory.mean_return(
                forward_prices, horizon=horizon, smoothing_window=k
            )
        elif method_name == "dominant_return":
            # Rust dominant_return ↔ Python peak_return (verified Wave4 Q5:
            # `if |max| >= |min| then max else min` — same tie-break, same sign).
            python_bps = LabelFactory.peak_return(
                forward_prices, horizon=horizon, smoothing_window=k
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")

        return rust_bps, np.asarray(python_bps, dtype=np.float64)

    @pytest.mark.parametrize(
        "method_name", ["smoothed_return", "point_return", "mean_return", "dominant_return"]
    )
    def test_parity_all_methods(
        self, fixture_data: Dict[str, Any], method_name: str
    ) -> None:
        """Per-method parametrized test running across ALL scenarios × horizons."""
        for scenario in _scenarios(fixture_data):
            for horizon in scenario["horizons"]:
                rust_bps, python_bps = self._run_method(method_name, scenario, horizon)
                assert rust_bps.shape == python_bps.shape, (
                    f"Shape mismatch for {method_name} h={horizon} "
                    f"in scenario '{scenario['name']}': "
                    f"rust={rust_bps.shape} vs python={python_bps.shape}"
                )
                np.testing.assert_allclose(
                    python_bps,
                    rust_bps,
                    rtol=PARITY_RTOL,
                    atol=PARITY_ATOL,
                    err_msg=(
                        f"Cross-language parity FAILED for "
                        f"{method_name} h={horizon} in scenario "
                        f"'{scenario['name']}'.\n"
                        f"Python (bps): {python_bps}\n"
                        f"Rust × 10000 (bps): {rust_bps}\n"
                        f"Max abs diff: {np.max(np.abs(python_bps - rust_bps)):.3e}"
                    ),
                )

    def test_smoothed_return_at_least_one_horizon(
        self, fixture_data: Dict[str, Any]
    ) -> None:
        """Sanity: at least one scenario tests smoothed_return."""
        any_tested = any(
            "smoothed_return" in s["rust_labels_decimal"]
            and any(s["rust_labels_decimal"]["smoothed_return"].values())
            for s in _scenarios(fixture_data)
        )
        assert any_tested, "Fixture must include at least one smoothed_return horizon"


class TestUnitConversion:
    """Lock the Python-bps vs Rust-decimal unit-conversion convention."""

    def test_python_returns_basis_points(self) -> None:
        """Python LabelFactory.smoothed_return returns bps (×10000), not decimals.

        Lock: a 1% return on synthetic prices [100, 101, 102] (k=0, h=2)
        should produce ≈100 bps (NOT 0.01).
        """
        # k=0 means past = prices[t], future = prices[t+h]
        forward_prices = np.array([[100.0, 101.0, 102.0]], dtype=np.float64)
        result = LabelFactory.smoothed_return(forward_prices, horizon=2, smoothing_window=0)
        # (102 - 100) / 100 = 0.02 = 200 bps
        np.testing.assert_allclose(result[0], 200.0, rtol=PARITY_RTOL)

    def test_rust_decimal_to_bps_conversion(self) -> None:
        """Lock the conversion: Rust 0.02 (= 2%) × 10000 = 200 bps."""
        rust_decimal = 0.02
        bps = rust_decimal * 10000.0
        np.testing.assert_allclose(bps, 200.0, rtol=PARITY_RTOL)
