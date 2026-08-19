import pandas as pd

from confidence_task.run_confidence_mvp_v2_1 import build_main_table_from_eval


def test_single_dataset_summary_does_not_require_kagglecrosscell():
    eval_summary = pd.DataFrame(
        [
            {
                "level": "dataset",
                "dataset_name": "KaggleCrossPatient",
                "score_name": "simple_combined_confidence",
                "direction_aligned_spearman": 0.33,
                "risk_cov_80_improve_pct": 2.8,
            },
            {
                "level": "dataset",
                "dataset_name": "KaggleCrossPatient",
                "score_name": "model_disagreement_risk",
                "direction_aligned_spearman": 0.42,
                "risk_cov_80_improve_pct": 11.6,
            },
        ]
    )
    table = build_main_table_from_eval(eval_summary, ["KaggleCrossPatient"])
    assert table["dataset_name"].tolist() == ["KaggleCrossPatient"]
    assert float(table["simple_combined_confidence_aligned_rho"].iloc[0]) == 0.33
    assert "KaggleCrossCell" not in set(table["dataset_name"])

