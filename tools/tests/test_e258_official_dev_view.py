"""The E258 engineering cache must never interpret held-out donor values."""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import build_e258_official_dev_view as view  # noqa: E402


def test_held_out_numeric_fields_are_skipped(tmp_path, monkeypatch):
    metadata = pd.DataFrame({"Cell_Line": ["pipw_4", "eipl_1", "tolg_4"]})
    monkeypatch.setattr(view, "task_contract", lambda path: (
        metadata, [("eipl_1", "AARS"), ("pipw_4", "AARS")]))
    source = tmp_path / "official.tsv.gz"
    rows = ["Target\tExpressed_Gene_Symbol\tExpressed_Gene_Ens_ID\t"
            "Cell_Line\twt_expr\tlfc\tpval_adj\n"]
    for line, values in (("pipw_4", ("0.1", "0.2")),
                         ("eipl_1", ("0.3", "0.4")),
                         ("tolg_4", ("secret", "secret"))):
        for gene, effect in zip(("G1", "G2"), values):
            rows.append(f"AARS\t{gene}\tENSG\t{line}\t1.0\t{effect}\t0.1\n")
    with gzip.open(source, "wt") as stream:
        stream.writelines(rows)
    output = tmp_path / "dev.npz"
    result = view.build(tmp_path / "metadata.tsv.gz", source, output,
                        verify_hash=False, min_common_genes=2)
    assert result["test_rows_mechanically_skipped"] == 2
    assert result["test_numeric_values_parsed"] == 0
    with np.load(output, allow_pickle=False) as data:
        assert data["gene"].tolist() == ["G1", "G2"]
        np.testing.assert_allclose(data["effect"], [[0.3, 0.4], [0.1, 0.2]])
        assert data["task_line"].tolist() == ["eipl_1", "pipw_4"]


def test_task_missing_from_official_summary_is_only_dropped_in_engineering_view(tmp_path, monkeypatch):
    metadata = pd.DataFrame({"Cell_Line": ["pipw_4", "eipl_1", "tolg_4"]})
    monkeypatch.setattr(view, "task_contract", lambda path: (
        metadata, [("eipl_1", "AARS"), ("eipl_1", "BRAF"), ("pipw_4", "AARS")]))
    source = tmp_path / "official.tsv.gz"
    header = "Target\tExpressed_Gene_Symbol\tExpressed_Gene_Ens_ID\tCell_Line\twt_expr\tlfc\tpval_adj\n"
    with gzip.open(source, "wt") as stream:
        stream.write(header)
        for line in ("pipw_4", "eipl_1"):
            for gene in ("G1", "G2"):
                stream.write(f"AARS\t{gene}\tENSG\t{line}\t1.0\t0.2\t0.1\n")
    output = tmp_path / "dev.npz"
    view.build(tmp_path / "metadata.tsv.gz", source, output,
               verify_hash=False, min_common_genes=2)
    with np.load(output, allow_pickle=False) as data:
        assert list(zip(data["task_line"], data["task_target"])) == [
            ("eipl_1", "AARS"), ("pipw_4", "AARS")]
