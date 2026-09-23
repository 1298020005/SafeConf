import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_e235_formal_evaluation import macro_metrics, state_metrics
from run_e235_seal_evaluate_finalize import validate_pretruth, sha256


def test_perfect_risk_ranking_has_unit_normalized_utility():
    block = pd.DataFrame({
        "task_id": [f"task_{index}" for index in range(10)],
        "full_gene_rmse": np.arange(1, 11, dtype=float),
        "rank_M": np.arange(1, 11, dtype=float),
        "score_M_plus_H": np.arange(10, 0, -1, dtype=float),
    })
    perfect = state_metrics(block, "rank_M", 0.2)
    reversed_result = state_metrics(block, "score_M_plus_H", 0.2)
    assert np.isclose(perfect["utility"], 1.0)
    assert np.isclose(perfect["spearman"], 1.0)
    assert reversed_result["utility"] < 0.0
    assert np.isclose(perfect["capture"], 19 / 55)


def test_bootstrap_occurrences_do_not_change_untied_result():
    rows = []
    for state in range(12):
        for index in range(10):
            rows.append({
                "cell_type": f"cell_{state // 3}", "treatment": f"treat_{state % 3}",
                "task_id": f"{state}:{index}", "full_gene_rmse": float(index + 1),
                "rank_M": float(index + 1),
            })
    frame = pd.DataFrame(rows)
    no_occurrences = macro_metrics(frame, "rank_M", 0.2)
    with_occurrences = macro_metrics(frame, "rank_M", 0.2,
                                      np.arange(len(frame), dtype=np.int64))
    assert np.isclose(no_occurrences["utility"], 1.0)
    assert no_occurrences == with_occurrences


def test_pretruth_gate_rejects_truth_columns(tmp_path):
    pretruth = tmp_path / "pretruth"
    pretruth.mkdir()
    rows = []
    for index in range(224):
        rows.append({"task_id": f"task_{index}", "condition": f"gene_{index % 53}",
                     "cell_type": f"cell_{index % 4}", "treatment": f"treat_{index % 3}",
                     "test_perturbed_expression_rows_read": 0,
                     **{name: float(index) for name in (
                         "rank_M", "score_M_plus_H", "score_M_plus_D", "score_M_plus_G",
                         "score_M_plus_N", "score_M_plus_C",
                         "score_original_five_80_20", "score_random_fixed")}})
    table = pretruth / "E235_PRETRUTH_SCORES.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    status = {"status": "SCORES_READY_AWAITING_REMOTE_SEAL", "score_sha256": sha256(table),
              "n_tasks": 224, "n_states": 12, "n_target_genes": 53,
              "test_perturbed_expression_rows_read": 0,
              "target_truth_access": "NOT_AUTHORIZED"}
    (pretruth / "E235_PRETRUTH_SCORE_STATUS.json").write_text(json.dumps(status))
    assert validate_pretruth(pretruth, {"score_sha256": sha256(table)}) == status
    frame = pd.read_csv(table)
    frame["full_gene_rmse"] = 0.1
    frame.to_csv(table, index=False)
    status["score_sha256"] = sha256(table)
    (pretruth / "E235_PRETRUTH_SCORE_STATUS.json").write_text(json.dumps(status))
    with pytest.raises(RuntimeError, match="truth-like fields"):
        validate_pretruth(pretruth, {"score_sha256": sha256(table)})
