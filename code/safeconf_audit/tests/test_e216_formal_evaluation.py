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
    / "run_e216_formal_evaluation.py"
)
SPEC = importlib.util.spec_from_file_location("run_e216_formal_evaluation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_consecutive_sparse_truth_reader(tmp_path):
    dense = np.asarray(
        [[1, 0, 2], [0, 3, 4], [5, 0, 6], [7, 8, 0]], dtype=np.float64
    )
    sparse = csr_matrix(dense)
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.create_dataset("data", data=sparse.data)
        group.create_dataset("indices", data=sparse.indices)
        group.create_dataset("indptr", data=sparse.indptr)
    result = MODULE.read_selected_csr_blocks(
        path, [0, 1, 3], np.asarray([0, 2]), n_total_genes=3, max_rows_per_block=2
    )
    np.testing.assert_allclose(result.toarray(), [[1, 2], [0, 4], [7, 0]])


def test_review_metric_rewards_oracle_order():
    outcome = np.asarray([1, 2, 3, 4, 5], dtype=float)
    tasks = np.asarray(["a", "b", "c", "d", "e"])
    result = MODULE.review_metrics(outcome, outcome, tasks)
    assert result["high_error_capture"] == 1.0
    assert np.isclose(result["oracle_normalized_utility"], 1.0)


def test_bootstrap_is_stratified_and_returns_only_evaluable_draws(monkeypatch):
    rows = []
    for treatment_offset, treatment in enumerate(["IFNG", "INS", "TGFB"]):
        for condition_offset, condition in enumerate(["A", "B", "C", "D"]):
            for cell_offset, cell_type in enumerate(["c1", "c2", "c3", "c4"]):
                value = 10 * treatment_offset + 2 * condition_offset + 0.1 * cell_offset
                rows.append(
                    {
                        "condition": f"{treatment}_{condition}",
                        "cell_type": cell_type,
                        "treatment": treatment,
                        "task_id": f"{treatment}_{condition}|{cell_type}|{treatment}",
                        "safeconf_m_4to1": value,
                        "predicted_magnitude": -value + 0.3 * condition_offset,
                        "latent_centroid_error": value + 0.2 * cell_offset,
                    }
                )
    monkeypatch.setattr(MODULE, "N_BOOTSTRAP", 25)
    result = MODULE.bootstrap_increments(pd.DataFrame(rows))
    assert len(result) == 25
    assert result.draw.tolist() == list(range(25))
    assert result.attempt.is_monotonic_increasing
    assert np.isfinite(result[["delta_macro_spearman", "delta_utility_20"]]).all().all()
