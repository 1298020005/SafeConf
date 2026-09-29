import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PATH = Path(__file__).parents[1] / "tools/scripts/analyze_public_history_error_memory.py"
SPEC = importlib.util.spec_from_file_location("phe", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def frame(n=20):
    rng = np.random.default_rng(9)
    data = {column: rng.normal(size=n) for column in MOD.P}
    data.update(task_id=[f"t{i}" for i in range(n)], true_error_rmse=np.arange(n, dtype=float))
    return pd.DataFrame(data)


def test_leave_one_out_error_memory_excludes_self_label():
    data = frame()
    features = MOD.error_memory_features(data, data, leave_self_out=True)
    # The largest label cannot be its own nearest-neighbour contribution.
    assert features.loc[len(data) - 1, "error_memory_knn_mean"] < data.true_error_rmse.max()
    assert np.isfinite(features.to_numpy()).all()


def test_error_memory_query_does_not_need_query_truth():
    memory = frame(20)
    query = frame(4).drop(columns="true_error_rmse")
    features = MOD.error_memory_features(memory, query, leave_self_out=False)
    assert features.shape == (4, len(MOD.ERROR_MEMORY))
    assert np.isfinite(features.to_numpy()).all()
