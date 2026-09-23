import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_e233_stage1_supervisor import evaluate_variant


def test_validation_gate_uses_mean_and_contexts(tmp_path):
    output = tmp_path / "prediction"
    output.mkdir()
    tasks = pd.DataFrame({
        "cell_type": [f"c{i}" for i in range(8)], "treatment": ["t"] * 8,
        "condition": [f"g{i}" for i in range(8)], "task_id": [f"x{i}" for i in range(8)],
    })
    tasks.to_csv(output / "E208_VALIDATION_TASKS.csv", index=False)
    controls = np.zeros((8, 2)); truth = np.ones((8, 2)); predictions = np.full((8, 2), 0.5)
    np.save(output / "E208_VALIDATION_CONTROL_CENTROIDS.npy", controls)
    np.save(output / "E208_VALIDATION_PREDICTION_CENTROIDS.npy", predictions)
    truth_path = tmp_path / "truth.npy"; np.save(truth_path, truth)
    _, _, summary = evaluate_variant(output, truth_path)
    assert summary["model_mean_mse"] < summary["no_change_mean_mse"]
    assert summary["contexts_not_worse"] == 8
    assert summary["competence_gate"] == "PASS"
