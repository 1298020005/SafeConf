import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "scripts"
    / "build_e216_prediction_controls.py"
)
SPEC = importlib.util.spec_from_file_location("build_e216_prediction_controls", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_control_selection_is_capped_and_never_selects_perturbed(monkeypatch):
    monkeypatch.setattr(MODULE, "EXPECTED_STATES", 2)
    obs = pd.DataFrame(
        {
            "condition": ["control", "control", "G1", "control", "G2"],
            "cell_type": ["a", "a", "a", "b", "b"],
            "treatment": ["x", "x", "x", "y", "y"],
        },
        index=["c1", "c2", "p1", "c3", "p2"],
    )
    tasks = pd.DataFrame(
        {
            "condition": ["G1", "G2"],
            "cell_type": ["a", "b"],
            "treatment": ["x", "y"],
        }
    )
    positions, counts = MODULE.select_control_positions(obs, tasks, cap=1)
    assert len(positions) == 2
    assert set(positions).issubset({0, 1, 3})
    assert counts.n_selected_controls.tolist() == [1, 1]


def test_sparse_reader_reads_only_selected_rows_and_genes(tmp_path):
    dense = np.asarray(
        [[1, 0, 2, 0], [0, 3, 0, 4], [5, 0, 6, 7]], dtype=np.float64
    )
    sparse = csr_matrix(dense)
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.create_dataset("data", data=sparse.data)
        group.create_dataset("indices", data=sparse.indices)
        group.create_dataset("indptr", data=sparse.indptr)
    result = MODULE.read_csr_rows_on_gene_panel(
        path,
        [0, 2],
        np.asarray([0, 2], dtype=np.int64),
        n_total_genes=4,
    )
    np.testing.assert_allclose(result.toarray(), [[1, 2], [5, 6]])
