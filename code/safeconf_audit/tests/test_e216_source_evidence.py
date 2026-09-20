import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "scripts"
    / "build_e216_source_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("build_e216_source_evidence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_source_effect_is_perturbation_minus_matching_control():
    obs = pd.DataFrame(
        {
            "condition": ["control", "G", "G", "control", "G"],
            "cell_type": ["a", "a", "a", "b", "b"],
            "treatment": ["x", "x", "x", "y", "y"],
        }
    )
    expression = csr_matrix(
        np.asarray([[1, 2], [3, 6], [5, 8], [10, 20], [14, 26]], dtype=float)
    )
    deltas, metadata = MODULE.aggregate_source_effects(expression, obs, {"G"})
    np.testing.assert_allclose(deltas, [[3, 5], [4, 6]])
    assert metadata.n_source_cells.tolist() == [2, 1]


def test_task_summary_excludes_target_state():
    tasks = pd.DataFrame(
        {"condition": ["G"], "cell_type": ["a"], "treatment": ["x"]}
    )
    source = pd.DataFrame(
        {
            "condition": ["G", "G", "G"],
            "cell_type": ["a", "b", "c"],
            "treatment": ["x", "y", "z"],
            "n_source_cells": [10, 11, 12],
        }
    )
    summary = MODULE.task_source_summary(tasks, source)
    assert summary.n_source_contexts.tolist() == [2]
    assert summary.n_source_cells.tolist() == [23]
