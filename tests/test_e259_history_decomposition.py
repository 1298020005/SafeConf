"""Small E259 guards for source-only history and fair fold-wise ranking."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/scripts"))
from run_e259_feng_fixed_decomposition import source_mean  # noqa: E402
from run_e259_cui_fold_label_budget import rank_within_folds  # noqa: E402


def test_feng_source_mean_excludes_held_donor() -> None:
    view = {
        "task_target": np.asarray(["T", "T", "T", "T"]),
        "task_donor": np.asarray(["a", "b", "c", "d"]),
        "gene": np.asarray(["g1", "g2"]),
    }
    full = {("T", donor): np.asarray(values, dtype=np.float32)
            for donor, values in {
                "a": [999, 999], "b": [2, 4], "c": [4, 8], "d": [6, 12]
            }.items()}
    mean, eligible = source_mean(np.asarray([0]), view, full,
                                 ("a", "b", "c", "d"))
    assert eligible.tolist() == [True]
    np.testing.assert_allclose(mean[0], [4, 8])


def test_cui_percentiles_are_computed_within_outer_fold() -> None:
    frame = pd.DataFrame({"fold_id": [0, 0, 1, 1],
                          "prediction_l2_norm": [10, 20, 100, 200],
                          "true_error_rmse": [2, 1, 20, 10]})
    ranked = rank_within_folds(frame, ("prediction_l2_norm",))
    np.testing.assert_allclose(ranked.rank_prediction_l2_norm,
                               [.25, .75, .25, .75])
    np.testing.assert_allclose(ranked.rank_true_error_rmse,
                               [.75, .25, .75, .25])
