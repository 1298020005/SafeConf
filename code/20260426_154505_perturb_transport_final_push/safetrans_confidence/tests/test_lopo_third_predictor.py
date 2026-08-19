from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from safetrans_confidence.cli.run_lopo_third_predictor import (
    DISAGREEMENT_FEATURES,
    PREDICTOR_OUTPUT_FEATURES,
    _feature_columns,
    _fit_learned_scores,
    _normalization_audit,
    _write_report,
    predict_control_1nn,
    predict_pert_mean,
)
from safetrans_confidence.features.normalize import QNORM_SUFFIX


def _task(
    key: str,
    split: str,
    context: str,
    effect: list[float],
    control: list[float],
) -> dict:
    return {
        "task_key": key,
        "fold_id": 0,
        "split": split,
        "context": context,
        "perturbation": "P1",
        "true_effect": np.asarray(effect, dtype=np.float32),
        "control_mean": np.asarray(control, dtype=np.float32),
    }


def test_pert_mean_excludes_train_target_from_its_own_prediction() -> None:
    tasks = pd.DataFrame(
        [
            _task("a", "train", "A", [1.0, 1.0], [1.0, 0.0]),
            _task("b", "train", "B", [3.0, 3.0], [0.0, 1.0]),
            _task("c", "test", "C", [2.0, 2.0], [1.0, 1.0]),
        ]
    )
    result = predict_pert_mean(tasks).set_index("task_key")
    np.testing.assert_allclose(result.loc["a", "predicted_effect"], [3.0, 3.0])
    np.testing.assert_allclose(result.loc["b", "predicted_effect"], [1.0, 1.0])
    np.testing.assert_allclose(result.loc["c", "predicted_effect"], [2.0, 2.0])


def test_control_1nn_uses_nearest_other_context() -> None:
    tasks = pd.DataFrame(
        [
            _task("a", "train", "A", [1.0, 0.0], [1.0, 0.0]),
            _task("b", "train", "B", [0.0, 3.0], [0.0, 1.0]),
            _task("c", "test", "C", [1.0, 1.0], [0.99, 0.01]),
        ]
    )
    result = predict_control_1nn(tasks).set_index("task_key")
    np.testing.assert_allclose(result.loc["c", "predicted_effect"], [1.0, 0.0])


def test_pre_model_feature_set_excludes_predictor_outputs() -> None:
    raw = [
        "context_similarity_max",
        "perturbation_support_count",
        "historical_residual_risk",
        *DISAGREEMENT_FEATURES,
        *PREDICTOR_OUTPUT_FEATURES,
    ]
    normalized = [f"{column}{QNORM_SUFFIX}" for column in raw]
    selected = _feature_columns(normalized, "pre_model_task_only")
    selected_raw = {column.removesuffix(QNORM_SUFFIX) for column in selected}
    assert "context_similarity_max" in selected_raw
    assert "historical_residual_risk" in selected_raw
    assert selected_raw.isdisjoint(
        set(DISAGREEMENT_FEATURES + PREDICTOR_OUTPUT_FEATURES)
    )


def test_normalization_audit_rejects_test_only_group() -> None:
    base = pd.DataFrame(
        {
            "dataset_name": ["D"],
            "fold_id": [0],
            "predictor_name": ["PertMeanPredictor"],
            "split": ["test"],
        }
    )
    with pytest.raises(ValueError, match="lack train/val"):
        _normalization_audit(base)


def test_lodo_lopo_excludes_heldout_dataset_and_third_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[int] = []

    class DummyModel:
        def fit(self, x, y):
            captured.append(len(x))

        def predict(self, x):
            return np.full(len(x), 0.5)

    monkeypatch.setattr(
        "safetrans_confidence.cli.run_lopo_third_predictor._make_model",
        lambda *args, **kwargs: DummyModel(),
    )
    rows = []
    for dataset in ["D1", "D2"]:
        for predictor in [
            "V0StrongBaseline",
            "ContextSimBaseline",
            "PertMeanPredictor",
        ]:
            for split in ["train", "val", "test"]:
                rows.append(
                    {
                        "record_id": f"{dataset}-{predictor}-{split}",
                        "dataset_name": dataset,
                        "dataset_family": "gene_main",
                        "task_key": f"{dataset}-{split}",
                        "fold_id": 0,
                        "split": split,
                        "context": "C",
                        "perturbation": "P",
                        "predictor_name": predictor,
                        "true_error_rmse": 0.1,
                        "true_effect_l2_norm": 1.0,
                        "context_similarity_max_qnorm": 0.5,
                    }
                )
    base = pd.DataFrame(rows)
    scores, provenance = _fit_learned_scores(
        base,
        "PertMeanPredictor",
        "pre_model_task_only",
        "lodo_lopo",
        ["context_similarity_max_qnorm"],
        5201,
    )
    assert len(scores) == 2
    assert captured == [4, 4]
    for _, row in provenance.iterrows():
        assert row["heldout_dataset"] not in row["training_datasets"].split(";")
        assert row["training_predictors"] == (
            "ContextSimBaseline;V0StrongBaseline"
        )
        assert not bool(row["third_predictor_error_used_for_fit"])
        assert not bool(row["heldout_dataset_error_used_for_fit"])


def test_lopo_fits_each_outer_fold_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_train_rows: list[int] = []

    class DummyModel:
        def fit(self, x, y):
            captured_train_rows.append(len(x))

        def predict(self, x):
            return np.full(len(x), 0.5)

    monkeypatch.setattr(
        "safetrans_confidence.cli.run_lopo_third_predictor._make_model",
        lambda *args, **kwargs: DummyModel(),
    )
    rows = []
    for dataset in ["D1", "D2"]:
        for fold_id in [0, 1]:
            for predictor in [
                "V0StrongBaseline",
                "ContextSimBaseline",
                "PertMeanPredictor",
            ]:
                for split in ["train", "val", "test"]:
                    rows.append(
                        {
                            "record_id": (
                                f"{dataset}-{fold_id}-{predictor}-{split}"
                            ),
                            "dataset_name": dataset,
                            "dataset_family": "gene_main",
                            "task_key": f"{dataset}-{fold_id}-{split}",
                            "fold_id": fold_id,
                            "split": split,
                            "context": "C",
                            "perturbation": "P",
                            "predictor_name": predictor,
                            "true_error_rmse": 0.1 + 0.01 * fold_id,
                            "true_effect_l2_norm": 1.0,
                            "context_similarity_max_qnorm": 0.5,
                        }
                    )
    scores, provenance = _fit_learned_scores(
        pd.DataFrame(rows),
        "PertMeanPredictor",
        "pre_model_task_only",
        "lopo",
        ["context_similarity_max_qnorm"],
        5201,
    )
    assert captured_train_rows == [8, 8]
    assert len(scores) == 4
    assert sorted(provenance["fold_id"].tolist()) == [0, 1]
    assert provenance["source_target_task_overlap"].eq(0).all()


def test_smoke_report_uses_observed_denominator_without_formal_gate(
    tmp_path,
) -> None:
    ladder = pd.DataFrame(
        [
            {
                "dataset_name": dataset,
                "predictor_name": "PertMeanPredictor",
                "validation_mode": mode,
                "feature_set": feature_set,
                "partial_rho_control_magnitude": 0.2,
            }
            for dataset in ["D1", "D2"]
            for mode, feature_set in [
                ("lopo", "pre_model_task_only"),
                ("lodo_lopo", "full"),
            ]
        ]
    )
    diversity = pd.DataFrame(
        [
            {
                "dataset_name": dataset,
                "third_predictor": "Control1NNPredictor",
                "independent_under_50pct_near_identical": True,
            }
            for dataset in ["D1", "D2"]
        ]
    )
    report_path = tmp_path / "report.md"
    decision = _write_report(report_path, ladder, diversity)
    report = report_path.read_text(encoding="utf-8")
    assert "2/2" in report
    assert "/7" not in report
    assert decision["pre_model_task_only_gate"] == "smoke_not_evaluated"
    assert decision["lodo_lopo_gate"] == "smoke_not_evaluated"
    assert decision["control1nn_role"] == "smoke_not_evaluated"
    assert not decision["formal_seven_dataset_gate_evaluated"]
