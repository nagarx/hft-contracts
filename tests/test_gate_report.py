"""Lock the cross-stage gate-report contract (TB-5).

``GATE_STATUS_VALUES`` is the canonical lower-case status vocabulary every gate
runner MUST emit under ``captured_metrics["gate_report"]["status"]``;
``ExperimentRecord.index_entry()`` projects it for fast
``hft-ops ledger list --gate-status`` filtering. A typo/addition/removal in this
SSoT set silently breaks cross-stage gate routing. These are the hft-rules §6
golden tests that freeze the contract (the set was previously asserted nowhere).
"""

from hft_contracts.gate_report import GATE_STATUS_VALUES, GateReportDict


class TestGateStatusValues:
    def test_canonical_membership_frozen(self):
        # Adding / removing / renaming a status is a cross-stage contract change
        # that MUST update this assertion in the same commit.
        assert GATE_STATUS_VALUES == frozenset({"pass", "warn", "fail", "abort"})

    def test_is_frozenset(self):
        # Immutable by type so a consumer cannot mutate the shared SSoT set.
        assert isinstance(GATE_STATUS_VALUES, frozenset)

    def test_all_lowercase(self):
        # The validation-stage adapter lowercases ``verdict`` -> ``status`` before
        # writing; the canonical set must therefore be entirely lower-case.
        assert all(v == v.lower() for v in GATE_STATUS_VALUES)


class TestGateReportDict:
    def test_documented_keys_present(self):
        # ``status`` (required-in-practice) + ``summary`` (recommended) are the
        # documented keys; TypedDict(total=False) tolerates stage-specific extras.
        annotations = GateReportDict.__annotations__
        assert "status" in annotations
        assert "summary" in annotations
