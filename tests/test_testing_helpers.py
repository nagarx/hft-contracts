"""Tests for hft_contracts._testing — the cross-module fixture-path resolver.

Locks the contract that consumer integration tests depend on:
  - phase0_fixture_dir() returns a concrete Path
  - the returned directory contains the canonical Phase 0 fixtures
  - the error path is actionable when fixtures are absent
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hft_contracts._testing import phase0_fixture_dir


class TestPhase0FixtureDir:
    def test_returns_existing_directory(self):
        fixture_dir = phase0_fixture_dir()
        assert isinstance(fixture_dir, Path)
        assert fixture_dir.is_dir()

    def test_contains_mbo_fixture(self):
        assert (phase0_fixture_dir() / "synthetic_mbo.npz").exists()

    def test_contains_basic_fixture(self):
        assert (phase0_fixture_dir() / "synthetic_basic.npz").exists()

    def test_contains_golden_values(self):
        assert (phase0_fixture_dir() / "golden_values.json").exists()

    def test_contains_readme(self):
        assert (phase0_fixture_dir() / "README.md").exists()

    def test_contains_generate_script(self):
        assert (phase0_fixture_dir() / "generate.py").exists()

    def test_contains_metadata_jsons(self):
        d = phase0_fixture_dir()
        assert (d / "fixture_metadata_mbo.json").exists()
        assert (d / "fixture_metadata_basic.json").exists()

    def test_resolves_absolute(self):
        """Resolver must return an absolute path — relative paths break when tests cd."""
        assert phase0_fixture_dir().is_absolute()

    def test_idempotent(self):
        """Repeated calls return the same path (no cache staleness)."""
        a = phase0_fixture_dir()
        b = phase0_fixture_dir()
        assert a == b
