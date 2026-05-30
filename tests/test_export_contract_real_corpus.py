"""End-to-end contract test against the REAL on-disk v3p0 export corpus.

Every other ``validate_export_contract`` test uses synthetic dict fixtures.
This one locks the actual shipped producer output: a real
``{day}_metadata.json`` from the v3p0 baseline export must pass the full
contract validator without raising. It is the only test that exercises the
genuine producer → contract path rather than a hand-built replica.

Skipped (not failed) when the external data volume is not mounted, so the
suite stays green on clones / CI without the ``data`` symlink — mirroring the
data-absent idiom at ``tests/test_label_factory_parity.py`` (in-body
``pytest.skip``). The specific day file is chosen dynamically (glob) so the
test survives a re-export with different day boundaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hft_contracts.validation import validate_export_contract

# ``data`` is a symlink to the external volume; e5_timebased_60s_v3p0 is the
# recommended clean v3p0 baseline corpus (98-feature, schema_version 3.0).
_TRAIN_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "exports"
    / "e5_timebased_60s_v3p0"
    / "train"
)


def _pick_real_metadata() -> Path | None:
    if not _TRAIN_DIR.exists():
        return None
    files = sorted(_TRAIN_DIR.glob("*_metadata.json"))
    return files[0] if files else None


def test_real_v3p0_metadata_passes_export_contract():
    meta_path = _pick_real_metadata()
    if meta_path is None:
        pytest.skip(
            f"v3p0 export corpus absent at {_TRAIN_DIR} "
            "(external data volume not mounted)"
        )

    metadata = json.loads(meta_path.read_text())

    # The real shipped export must pass the full contract validator WITHOUT
    # raising. A warnings list is permitted (optional / provenance-completeness
    # fields), so we assert "did not raise" + positive schema sanity — NOT
    # ``warnings == []`` (which would be brittle against provenance warnings).
    warnings = validate_export_contract(metadata)
    assert isinstance(warnings, list)
    assert metadata["schema_version"] == "3.0", (
        f"{meta_path.name}: expected schema_version '3.0', got "
        f"{metadata.get('schema_version')!r}"
    )
    assert metadata["n_features"] in (98, 148), (
        f"{meta_path.name}: unexpected n_features {metadata.get('n_features')!r}"
    )
