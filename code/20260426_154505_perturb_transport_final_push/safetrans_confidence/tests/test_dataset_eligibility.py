from pathlib import Path

import pandas as pd
import pytest

from safetrans_confidence.data.eligibility import audit_h5ad

ATLAS = Path("/home/yyf/data/singlecell_perturbation_atlas")
SCAN = ATLAS / "metadata" / "h5ad_scan.tsv"


def _path(study_family: str) -> Path:
    scan = pd.read_csv(SCAN, sep="\t")
    sub = scan[scan["study_family"].astype(str) == study_family]
    assert len(sub) >= 1
    path = Path(str(sub.iloc[0]["local_path"]))
    if not path.exists():
        old_root = Path("/home/yyf/datasets/singlecell_perturbation_atlas")
        try:
            path = ATLAS / path.relative_to(old_root)
        except ValueError:
            pass
    return path


@pytest.mark.skipif(not SCAN.exists(), reason="atlas missing")
def test_norman_generalization_not_cross_context_eligible():
    row = audit_h5ad(_path("Norman"), "Norman")
    assert row.cross_context_eligible is False
    assert "no_context" in row.reason_if_not or row.n_contexts < 2


@pytest.mark.skipif(not SCAN.exists(), reason="atlas missing")
def test_haber_eligible():
    row = audit_h5ad(_path("Haber"), "Haber")
    assert row.cross_context_eligible is True


@pytest.mark.skipif(not SCAN.exists(), reason="atlas missing")
def test_kcp_eligible():
    row = audit_h5ad(_path("KaggleCrossPatient"), "KaggleCrossPatient")
    assert row.cross_context_eligible is True


@pytest.mark.skipif(not SCAN.exists(), reason="atlas missing")
def test_crosspatient_and_frangieh_eligible():
    assert audit_h5ad(_path("crossPatient"), "crossPatient").cross_context_eligible is True
    assert audit_h5ad(_path("Frangieh"), "Frangieh").cross_context_eligible is True
