"""A train-only view must contain controls and training labels, never held-out X."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


SCRIPT = Path(__file__).resolve().parents[1] / "tools/scripts/build_e247_train_only_view.py"
SPEC = importlib.util.spec_from_file_location("build_e247_train_only_view", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_view_never_indexes_heldout_expression(tmp_path: Path) -> None:
    source = tmp_path / "source.h5ad"
    split = tmp_path / "split.csv"
    output = tmp_path / "train_only.h5ad"
    labels = ["control", "t1", "v1", "t2", "e1", "t1", "control"]
    values = np.asarray(
        [[1, 0, 2], [2, 3, 0], [999, 999, 999], [4, 0, 5],
         [888, 888, 888], [0, 6, 7], [8, 0, 9]], dtype=np.float32
    )
    obs = pd.DataFrame(
        {"perturbation": pd.Categorical(labels)},
        index=[f"cell{i}" for i in range(len(labels))],
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    ad.AnnData(X=csr_matrix(values), obs=obs, var=var).write_h5ad(source)
    with split.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["perturbation", "n_cells", "eligible_ge30", "split"])
        for label, count, split_name in (
            ("control", 2, "excluded"), ("t1", 2, "train"),
            ("t2", 1, "train"), ("v1", 1, "validation"), ("e1", 1, "test")
        ):
            writer.writerow([label, count, int(split_name != "excluded"), split_name])

    result = module.build_view(
        source, split, output, source_sha=module.sha256_file(source),
        split_sha=module.sha256_file(split),
        expected_counts={"train": 2, "validation": 1, "test": 1},
    )
    viewed = ad.read_h5ad(output)
    assert result["n_rows"] == 5
    assert result["n_controls"] == 2
    assert result["heldout_expression_rows_indexed"] == 0
    assert set(viewed.obs["perturbation"].cat.categories) == {"control", "t1", "t2"}
    assert list(viewed.obs_names) == ["cell0", "cell1", "cell3", "cell5", "cell6"]
    np.testing.assert_array_equal(viewed.X.toarray(), values[[0, 1, 3, 5, 6]])
    assert not np.any(viewed.X.toarray() >= 888)
