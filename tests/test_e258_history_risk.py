"""Small mathematical and donor-exclusion checks for E258 risk development."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/scripts"))
from run_e258_history_validation import percentile, utility20  # noqa: E402
from run_e258_train_oof_supervised_risk import source_mean_for_rows  # noqa: E402
from prepare_e258_split_half_groups import split  # noqa: E402


def test_within_line_percentiles_and_utility() -> None:
    values = np.asarray([5.0, 1.0, 2.0, 9.0])
    lines = np.asarray(["a", "a", "b", "b"])
    np.testing.assert_allclose(percentile(values, lines),
                               [0.75, 0.25, 0.25, 0.75])
    error = np.arange(1, 21, dtype=float)
    assert utility20(error, error) == (20 + 19 + 18 + 17) / error.sum()


def test_historical_moment_identity_is_exact() -> None:
    prediction = np.asarray([0.1, -0.2, 0.3])
    history = np.asarray([[0.0, -0.4, 0.4], [0.2, -0.1, 0.1],
                          [-0.1, 0.0, 0.6]])
    gap2 = np.mean((prediction - history.mean(axis=0)) ** 2)
    dispersion2 = np.var(history, axis=0).mean()
    direct = np.mean((prediction[None, :] - history) ** 2)
    np.testing.assert_allclose(gap2 + dispersion2, direct, rtol=1e-12)


def test_oof_source_mean_does_not_include_heldout_donor() -> None:
    view = {
        "task_donor": np.asarray(["a", "b", "c", "d"]),
        "task_target": np.asarray(["T", "T", "T", "T"]),
        "gene": np.asarray(["g1", "g2"]),
    }
    history = {(target, donor): np.asarray(value, dtype=np.float32)
               for target, donor, value in [
                   ("T", "a", [999, 999]), ("T", "b", [2, 4]),
                   ("T", "c", [4, 8]), ("T", "d", [6, 12])]}
    mean, allowed = source_mean_for_rows(
        view, history, np.asarray([0]), ("b", "c", "d"))
    assert allowed.tolist() == [True]
    np.testing.assert_allclose(mean[0], [4, 8])


def test_split_half_mapping_never_selects_forbidden_columns() -> None:
    original = np.asarray([-1, 0, 1, -1, 0, 1, 0, 1], dtype=np.int32)
    halves, counts = split(original, 2)
    assert halves.tolist() == [-1, 0, 2, -1, 1, 3, 0, 2]
    assert counts.tolist() == [2, 1, 2, 1]
