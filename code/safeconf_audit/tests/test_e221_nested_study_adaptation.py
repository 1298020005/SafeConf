"""Small contract checks for E221; run with the project Python environment."""
from pathlib import Path
import runpy
import numpy as np
import pandas as pd

NS = runpy.run_path(str(Path(__file__).parents[3] / "tools/scripts/run_e221_nested_study_adaptation.py"))


def test_candidates_predeclared_and_unique():
    items = NS["candidates"]()
    assert len(items) == len({x[0] for x in items})
    assert len(items) > 20


def test_metric_oracle_bounds():
    m = NS["metrics"]
    y = np.arange(1.0, 11.0)
    assert m(y, y)["utility_20"] == 1.0
    assert m(y[::-1], y)["utility_20"] < 0


def test_inner_selection_does_not_require_outer_study():
    rows = []
    for study in ("a", "b", "c"):
        for fold in ("f1", "f2"):
            for i in range(10):
                rows.append({"dataset": study, "fold_id": f"{study}_{fold}", "m": i / 10 + .001,
                             "s": (10-i) / 10 + .001, "d": .5, "error_two_predictor_mean_rmse": float(i+1)})
    frame = pd.DataFrame(rows)
    name, spec, table = NS["select_candidate"](frame)
    assert name and len(table) == 3 * len(NS["candidates"]())
