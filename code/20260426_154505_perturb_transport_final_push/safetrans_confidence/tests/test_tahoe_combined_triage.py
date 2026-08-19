from __future__ import annotations

import numpy as np
import pandas as pd

from safetrans_confidence.cli.run_tahoe_combined_triage import (
    add_fixed_rank_blends,
    bootstrap_top10,
    decide_gate,
    point_summary,
)


def _synthetic_scores(n_tasks: int = 60) -> pd.DataFrame:
    rows = []
    for task in range(n_tasks):
        for predictor_index, predictor in enumerate(("p1", "p2")):
            error = float(task + predictor_index / 10)
            rows.append(
                {
                    "task_key": f"task_{task}",
                    "fold_id": task % 5,
                    "predictor_name": predictor,
                    "true_error_rmse": error,
                    "safeconf_full": error + (task % 3),
                    "predicted_magnitude": error + ((task + 1) % 4),
                }
            )
    return pd.DataFrame(rows)


def test_fixed_blends_are_bounded_and_use_frozen_weights() -> None:
    scored = add_fixed_rank_blends(_synthetic_scores())
    assert scored["safeconf_rank"].between(0, 1).all()
    assert scored["magnitude_rank"].between(0, 1).all()
    expected = 0.5 * scored["safeconf_rank"] + 0.5 * scored["magnitude_rank"]
    np.testing.assert_allclose(scored["combined_equal"], expected)


def test_score_construction_does_not_require_true_error() -> None:
    source = _synthetic_scores().drop(columns="true_error_rmse")
    scored = add_fixed_rank_blends(source)
    assert scored["combined_equal"].notna().all()


def test_point_summary_reports_all_fixed_scores() -> None:
    scored = add_fixed_rank_blends(_synthetic_scores())
    summary = point_summary(scored)
    assert set(summary["top_fraction"]) == {0.05, 0.10, 0.20}
    assert "combined_equal" in set(summary["score_name"])
    assert len(summary) == 15


def test_cluster_bootstrap_and_gate() -> None:
    scored = add_fixed_rank_blends(_synthetic_scores())
    summary, draws = bootstrap_top10(scored, n_bootstrap=30, seed=5201)
    assert int(summary.iloc[0]["n_task_clusters"]) == 60
    assert len(draws) == 30
    assert "combined_equal_minus_magnitude_ci_low" in summary.columns

    pass_adds = pd.DataFrame(
        [{"combined_equal_enrichment_ci_low": 2.0, "combined_equal_minus_magnitude_ci_low": 0.1}]
    )
    pass_useful = pd.DataFrame(
        [{"combined_equal_enrichment_ci_low": 2.0, "combined_equal_minus_magnitude_ci_low": -0.1}]
    )
    fail = pd.DataFrame(
        [{"combined_equal_enrichment_ci_low": 0.9, "combined_equal_minus_magnitude_ci_low": 0.1}]
    )
    assert decide_gate(pass_adds) == "PASS_ADDS_VALUE"
    assert decide_gate(pass_useful) == "PASS_USEFUL_NOT_BETTER"
    assert decide_gate(fail) == "FAIL"
