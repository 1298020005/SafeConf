"""Raw count ID normalization is explicit and fail-closed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_e258_raw_header import count_matrix_id  # noqa: E402


def test_author_metadata_cell_id_maps_to_count_header():
    assert count_matrix_id("PC-P4-D3_I73_AGGCCGTTCGCAGGCT-1") == \
        "P4_I73_AGGCCGTTCGCAGGCT-1"


def test_unexpected_id_fails_closed():
    with pytest.raises(ValueError, match="unexpected metadata cell ID"):
        count_matrix_id("P4_I73_AGGCCGTTCGCAGGCT-1")
