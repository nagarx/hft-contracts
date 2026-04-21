"""Phase 0 fixture integrity tests.

Verifies:
  1. Committed fixtures match regeneration (bit-deterministic under seed=42).
  2. Golden values JSON schema is well-formed.
  3. Array shapes and dtypes match the declared Phase 0 contract.
  4. Metadata JSON matches the real export contract shape.

These tests do NOT require torch / lob-models / lob-model-trainer — they validate the
fixture layer itself. Downstream integration tests (forward-pass golden, E2E pipeline)
live in their respective consumer modules and use `pytest.importorskip`.

Run:
  pytest hft-contracts/tests/test_phase0_fixtures.py -v
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "phase0_benchmark"
GENERATE_SCRIPT = FIXTURE_DIR / "generate.py"
GOLDEN_PATH = FIXTURE_DIR / "golden_values.json"


def _load_generate_module():
    """Import generate.py as a module (it's not on sys.path normally)."""
    spec = importlib.util.spec_from_file_location("phase0_generate", GENERATE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase0_generate"] = module
    spec.loader.exec_module(module)
    return module


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class TestFixturesExist:
    """Committed fixtures are present at expected paths."""

    def test_generate_script_exists(self):
        assert GENERATE_SCRIPT.exists(), "generate.py missing — Phase 0 fixture generator"

    def test_readme_exists(self):
        assert (FIXTURE_DIR / "README.md").exists()

    def test_mbo_npz_exists(self):
        assert (FIXTURE_DIR / "synthetic_mbo.npz").exists()

    def test_basic_npz_exists(self):
        assert (FIXTURE_DIR / "synthetic_basic.npz").exists()

    def test_mbo_metadata_exists(self):
        assert (FIXTURE_DIR / "fixture_metadata_mbo.json").exists()

    def test_basic_metadata_exists(self):
        assert (FIXTURE_DIR / "fixture_metadata_basic.json").exists()

    def test_golden_values_exists(self):
        assert GOLDEN_PATH.exists()


class TestGoldenValuesSchema:
    """golden_values.json has the expected top-level shape."""

    @pytest.fixture
    def golden(self) -> dict:
        return json.loads(GOLDEN_PATH.read_text())

    def test_schema_version(self, golden):
        assert golden["schema_version"] == "1"

    def test_seed_is_42(self, golden):
        assert golden["seed"] == 42

    def test_has_mbo_block(self, golden):
        assert "mbo" in golden
        for key in ("npz_file", "metadata_file", "array_hashes", "array_shapes", "array_dtypes", "metadata_sha256"):
            assert key in golden["mbo"], f"mbo.{key} missing"

    def test_has_basic_block(self, golden):
        assert "basic" in golden
        for key in ("npz_file", "metadata_file", "array_hashes", "array_shapes", "array_dtypes", "metadata_sha256"):
            assert key in golden["basic"], f"basic.{key} missing"

    def test_forward_pass_block_present(self, golden):
        """forward_pass block reserved for Phase I.A / I.B pinning; may be empty at Phase 0."""
        assert "forward_pass" in golden

    def test_signal_boundary_block_present(self, golden):
        """signal_boundary block reserved for Phase II pinning; may be empty at Phase 0."""
        assert "signal_boundary" in golden


class TestMBOFixture:
    """MBO synthetic fixture matches declared Phase 0 contract."""

    @pytest.fixture
    def arrays(self):
        with np.load(FIXTURE_DIR / "synthetic_mbo.npz") as npz:
            return {k: npz[k] for k in npz.files}

    @pytest.fixture
    def metadata(self):
        return json.loads((FIXTURE_DIR / "fixture_metadata_mbo.json").read_text())

    @pytest.fixture
    def golden_mbo(self):
        return json.loads(GOLDEN_PATH.read_text())["mbo"]

    def test_sequences_shape(self, arrays):
        assert arrays["sequences"].shape == (10, 20, 98)

    def test_sequences_dtype(self, arrays):
        assert arrays["sequences"].dtype == np.float32

    def test_sequences_no_nan_inf(self, arrays):
        assert np.all(np.isfinite(arrays["sequences"]))

    def test_regression_labels_shape(self, arrays):
        assert arrays["regression_labels"].shape == (10, 3)  # H = [10, 60, 300]

    def test_regression_labels_dtype(self, arrays):
        assert arrays["regression_labels"].dtype == np.float64

    def test_forward_prices_shape(self, arrays):
        # smoothing_offset (5) + max_horizon (300) + 1 = 306 columns
        assert arrays["forward_prices"].shape == (10, 306)

    def test_forward_prices_dtype(self, arrays):
        assert arrays["forward_prices"].dtype == np.float64

    def test_forward_prices_positive(self, arrays):
        """Forward prices represent USD — must be positive."""
        assert np.all(arrays["forward_prices"] > 0)

    def test_mid_price_at_index_40_positive(self, arrays):
        assert np.all(arrays["sequences"][:, :, 40] > 0)

    def test_spread_at_index_42_positive(self, arrays):
        assert np.all(arrays["sequences"][:, :, 42] > 0)

    def test_array_hashes_match_golden(self, arrays, golden_mbo):
        for key, arr in arrays.items():
            expected = golden_mbo["array_hashes"][key]
            actual = _sha256_bytes(arr.tobytes())
            assert actual == expected, (
                f"Array hash drift on mbo.{key}: committed={expected[:16]}... "
                f"actual={actual[:16]}... Fix: either the environment is "
                f"non-deterministic OR the generator changed. Run "
                f"`python generate.py --verify` to diagnose."
            )

    def test_metadata_contract_version(self, metadata):
        assert metadata["contract_version"] == "2.2"

    def test_metadata_schema_version(self, metadata):
        assert metadata["schema_version"] == "2.2"

    def test_metadata_n_features(self, metadata):
        assert metadata["n_features"] == 98

    def test_metadata_horizons(self, metadata):
        assert metadata["labeling"]["horizons"] == [10, 60, 300]

    def test_metadata_label_strategy(self, metadata):
        assert metadata["label_strategy"] == "regression"

    def test_metadata_has_forward_prices_block(self, metadata):
        fp = metadata["forward_prices"]
        assert fp["exported"] is True
        assert fp["smoothing_window_offset"] == 5
        assert fp["max_horizon"] == 300
        assert fp["n_columns"] == 306


class TestBASICFixture:
    """BASIC synthetic fixture matches declared Phase 0 contract."""

    @pytest.fixture
    def arrays(self):
        with np.load(FIXTURE_DIR / "synthetic_basic.npz") as npz:
            return {k: npz[k] for k in npz.files}

    @pytest.fixture
    def metadata(self):
        return json.loads((FIXTURE_DIR / "fixture_metadata_basic.json").read_text())

    @pytest.fixture
    def golden_basic(self):
        return json.loads(GOLDEN_PATH.read_text())["basic"]

    def test_sequences_shape(self, arrays):
        assert arrays["sequences"].shape == (10, 20, 34)

    def test_sequences_dtype(self, arrays):
        assert arrays["sequences"].dtype == np.float32

    def test_sequences_no_nan_inf(self, arrays):
        assert np.all(np.isfinite(arrays["sequences"]))

    def test_labels_shape(self, arrays):
        assert arrays["labels"].shape == (10, 8)  # H = [1, 2, 3, 5, 10, 20, 30, 60]

    def test_labels_dtype_is_float64(self, arrays):
        """BASIC labels are POINT RETURN regression (float64 bps), NOT int8 class."""
        assert arrays["labels"].dtype == np.float64

    def test_forward_prices_shape(self, arrays):
        # max_horizon (60) + 1 = 61 columns (BASIC has no smoothing offset)
        assert arrays["forward_prices"].shape == (10, 61)

    def test_forward_prices_positive(self, arrays):
        assert np.all(arrays["forward_prices"] > 0)

    def test_array_hashes_match_golden(self, arrays, golden_basic):
        for key, arr in arrays.items():
            expected = golden_basic["array_hashes"][key]
            actual = _sha256_bytes(arr.tobytes())
            assert actual == expected, (
                f"Array hash drift on basic.{key}: committed={expected[:16]}... "
                f"actual={actual[:16]}... Fix: run `python generate.py --verify` to diagnose."
            )

    def test_metadata_data_source(self, metadata):
        """BASIC fixture declares off-exchange data source."""
        assert metadata["data_source"] == "off_exchange_phase0_synthetic"

    def test_metadata_n_features(self, metadata):
        assert metadata["n_features"] == 34

    def test_metadata_label_strategy(self, metadata):
        assert metadata["label_strategy"] == "point_return"

    def test_metadata_horizons(self, metadata):
        assert metadata["horizons"] == [1, 2, 3, 5, 10, 20, 30, 60]


class TestRegenerationDeterminism:
    """Regeneration of fixtures produces bit-identical output as committed.

    This is the Phase 0 determinism contract: under seed=42 on the reference
    environment (numpy≥1.26, CPython 3.10+), running the generator twice
    MUST produce identical file bytes.
    """

    def test_regenerate_matches_committed_array_hashes(self, tmp_path):
        """Full regenerate into tmp; compare array hashes to committed golden."""
        gen = _load_generate_module()
        regenerated = gen.main(output_dir=tmp_path)
        committed = json.loads(GOLDEN_PATH.read_text())

        for pipeline in ("mbo", "basic"):
            for array_name, expected_hash in committed[pipeline]["array_hashes"].items():
                actual_hash = regenerated[pipeline]["array_hashes"][array_name]
                assert actual_hash == expected_hash, (
                    f"Drift in {pipeline}.{array_name}: "
                    f"committed={expected_hash[:16]}... regenerated={actual_hash[:16]}..."
                )

    def test_regenerate_matches_committed_metadata_hashes(self, tmp_path):
        gen = _load_generate_module()
        regenerated = gen.main(output_dir=tmp_path)
        committed = json.loads(GOLDEN_PATH.read_text())

        for pipeline in ("mbo", "basic"):
            assert regenerated[pipeline]["metadata_sha256"] == committed[pipeline]["metadata_sha256"], (
                f"Metadata hash drift on {pipeline}"
            )

    def test_same_seed_same_arrays_twice(self, tmp_path):
        """Running the generator twice into two tmp dirs must produce identical bytes."""
        gen = _load_generate_module()
        a_dir = tmp_path / "a"
        b_dir = tmp_path / "b"
        a = gen.main(output_dir=a_dir)
        b = gen.main(output_dir=b_dir)

        assert a["mbo"]["array_hashes"] == b["mbo"]["array_hashes"]
        assert a["basic"]["array_hashes"] == b["basic"]["array_hashes"]
        assert a["mbo"]["metadata_sha256"] == b["mbo"]["metadata_sha256"]
        assert a["basic"]["metadata_sha256"] == b["basic"]["metadata_sha256"]


class TestGeneratorDeterminismContract:
    """Static contract: generate.py MUST use only numpy primitives with frozen
    cross-version / cross-platform bit-stability.

    Any future edit that introduces an unstable primitive (``standard_t``,
    ``standard_gamma``, ``standard_cauchy``, BLAS-backed reductions on large
    arrays, etc.) silently breaks CI on a different numpy-version + platform
    combination than where the fixture was committed.

    This static grep-style check fails loud at code review rather than waiting
    for the post-merge CI run. The explicitly-AVOIDED list in the module
    docstring is the SSoT; this test mirrors it.

    Rationale (plan v2.0, §Architectural Invariants #6 "Contract Enforcement
    Pattern"): every cross-environment contract requires (a) declaration
    (docstring), (b) producer-side validator (generator itself), (c) consumer-
    side validator (this test). Closes the class of silent-CI-breakage-on-
    different-numpy-version bugs.
    """

    # Primitives documented as NOT bit-stable across numpy versions. Adding
    # one of these to generate.py SHOULD fail CI immediately.
    _FORBIDDEN_PRIMITIVES = (
        "standard_t",
        "standard_gamma",
        "standard_cauchy",
        "standard_exponential",  # conservative: cross-version-stable for default_rng only
        ".mean(",  # BLAS-backed reductions: SIMD-order-sensitive on large arrays
        ".matmul(",
        "@ ",  # matmul operator (conservative glob — would also flag "@staticmethod" inside strings,
        # but generator has no decorators on reduction paths)
    )

    def test_generator_uses_only_stable_primitives(self):
        """Static scan of generate.py source. No forbidden primitive appears
        outside comments/docstrings.

        Strategy: for each line in the source, strip line-comments + skip
        lines that are clearly inside a string literal (triple-quoted block
        comments). Then check for forbidden substrings.
        """
        gen_path = FIXTURE_DIR / "generate.py"
        source = gen_path.read_text().splitlines()

        in_triple_string = False
        triple_delim = None
        offenders: list[tuple[int, str, str]] = []

        for lineno, line in enumerate(source, start=1):
            stripped = line.strip()
            # Track triple-quoted string blocks (module docstring + function docstrings).
            if in_triple_string:
                if triple_delim in line:
                    in_triple_string = False
                    triple_delim = None
                continue  # Inside docstring — skip.
            if stripped.startswith('"""') or stripped.startswith("'''"):
                delim = stripped[:3]
                # Single-line triple-quote (open and close on same line)?
                rest = stripped[3:]
                if delim in rest:
                    continue  # whole thing on one line — skip
                in_triple_string = True
                triple_delim = delim
                continue
            # Strip inline comment.
            code = line.split("#", 1)[0]
            for forbidden in self._FORBIDDEN_PRIMITIVES:
                if forbidden in code:
                    offenders.append((lineno, forbidden, line.rstrip()))

        assert not offenders, (
            "Phase 0 generate.py uses a forbidden unstable primitive. "
            "Each listed primitive has drifted between numpy minor versions in the "
            "past (see module docstring 'Explicitly AVOIDED' list). If you must use "
            "one, document the numpy version pin + fixture-regeneration policy FIRST."
            f"\n\nOffenders:\n" + "\n".join(
                f"  line {n}: forbidden={f!r}  source: {s}" for n, f, s in offenders
            )
        )

    def test_contract_docstring_mentions_avoidance(self):
        """The module docstring must enumerate the AVOIDED primitives so the
        contract is discoverable during code review (not just in this test).
        """
        gen_path = FIXTURE_DIR / "generate.py"
        source = gen_path.read_text()
        # Minimal discoverability invariant — must mention the key categories.
        assert "standard_t" in source, "Docstring must call out standard_t as forbidden"
        assert "Explicitly AVOIDED" in source or "avoided" in source.lower(), (
            "Docstring must have an Explicitly-AVOIDED-primitives section"
        )
        assert "standard_normal" in source, "Docstring must call out stable primitive"
