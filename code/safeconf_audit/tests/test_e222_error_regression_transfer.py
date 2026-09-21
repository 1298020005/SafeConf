"""Small deterministic checks for E222's scoring and candidate contract."""
from pathlib import Path
import runpy
import numpy as np
import pandas as pd

NS = runpy.run_path(str(Path(__file__).parents[3] / "tools/scripts/run_e222_error_regression_transfer.py"))


def test_feature_groups_are_matched():
    assert NS["FEATURES"]["magnitude"] == ["m"]
    assert NS["FEATURES"]["full"] == ["m", "s", "d"]
    assert len(NS["CANDIDATES"]) == 3


def test_training_arrays_use_fold_normalized_target():
    rows = []
    for study in ("a", "b"):
        for fold in ("f1", "f2"):
            for i in range(4):
                rows.append({"dataset": study, "fold_id": study + fold, "m": i / 4 + .1,
                             "s": .2, "d": .3,
                             "error_two_predictor_mean_rmse": float(i + 1)})
    x, y, w = NS["training_arrays"](pd.DataFrame(rows), ["m"])
    assert x.shape == (16, 1)
    assert np.allclose(y[:4], np.array([.4, .8, 1.2, 1.6]))
    assert np.isclose(w.sum(), len(rows))


def test_full_model_only_receives_prediction_time_columns():
    assert NS["ERROR"] not in NS["FEATURES"]["full"]
