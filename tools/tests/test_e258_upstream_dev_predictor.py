"""E258 engineering predictor uses other donors as its source history."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_e258_official_dev_predictor as model  # noqa: E402


def test_source_mean_excludes_current_training_donor():
    donors = np.asarray(model.TRAIN + model.VALIDATION)
    lines = np.asarray([f"{donor}_1" for donor in donors])
    view = {
        "effect": np.asarray([[i, i, i] for i in range(1, 7)], dtype=np.float32),
        "task_donor": donors,
        "task_target": np.asarray(["A"] * len(donors)),
        "task_line": lines,
        "line": lines,
        "gene": np.asarray(["A", "G1", "G2"]),
        "control": np.asarray([[1, i, i + 2] for i in range(1, 7)], dtype=np.float32),
    }
    source_mean, similar, delta = model.baselines(view)
    np.testing.assert_allclose(source_mean[0], [3, 3, 3])  # (2+3+4)/3
    np.testing.assert_allclose(source_mean[4], [2.5, 2.5, 2.5])
    assert similar.shape == source_mean.shape == delta.shape
    assert model.gene_mask(view)[0].tolist() == [False, True, True]


def test_residual_model_starts_at_source_mean_and_backpropagates():
    network = model.ResidualMLP(3, (4,))
    x = torch.randn(2, 6)
    base = torch.randn(2, 3)
    torch.testing.assert_close(network(x, base), base)
    network(x, base).square().mean().backward()
    assert network.network[-1].weight.grad is not None


def test_train_only_shrinkage_fits_scalar_without_validation():
    base = np.asarray([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    truth = base * 0.5
    alpha = model.train_only_shrinkage(base, truth,
                                       np.ones_like(base, dtype=bool),
                                       np.asarray(["train_1", "train_2"]))
    assert alpha == 0.5
