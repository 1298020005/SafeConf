from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from safetrans_confidence.scoring.protocol_v0_2 import (
    CHEM_DATASETS,
    PRIMARY_SCORE_NAME,
    build_protocol_v0_2_primary_scores,
    build_protocol_v0_2_scores,
    assign_dataset_family,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "safetrans_confidence" / "config" / "scoring" / "protocol_v0_2.yaml"


def test_protocol_v02_formula_is_frozen():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["name"] == "protocol_v0_2_family_confidence"
    assert cfg["gene_main_formula"] == "context_similarity + log_support - model_disagreement"
    assert cfg["chem_robust_formula"] == "log_support - model_disagreement; stability_weight=0"
    assert "KaggleCrossPatient" in cfg["chem_datasets"]
    assert set(cfg["chem_datasets"]) == set(CHEM_DATASETS)


def test_main_chemical_datasets_use_chem_robust_formula():
    assert assign_dataset_family("McFarlandTsherniak2020") == "chem_robust"
    assert assign_dataset_family("SrivatsanTrapnell2020_sciplex3") == "chem_robust"


def test_main_gene_datasets_use_gene_main_formula():
    assert assign_dataset_family("CuiHacohen2023") == "gene_main"
    assert assign_dataset_family("Frangieh") == "gene_main"
    assert assign_dataset_family("LaraAstiasoHuntly2023_invivo") == "gene_main"
    assert assign_dataset_family("SantinhaPlatt2023") == "gene_main"


def _blind_primary_base() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": ["r1", "r2", "r3", "r4"],
            "dataset_name": ["Frangieh"] * 4,
            "fold_id": [0, 0, 0, 0],
            "split": ["train", "train", "val", "test"],
            "context": ["c1", "c2", "c3", "c4"],
            "perturbation": ["p1", "p2", "p3", "p4"],
            "predictor_name": ["V0StrongBaseline"] * 4,
            "context_similarity_max": [0.2, 0.4, 0.3, 0.6],
            "perturbation_support_count": [1, 3, 2, 5],
            "model_disagreement_rmse": [0.8, 0.4, 0.5, 0.2],
        }
    )


def test_primary_protocol_score_is_label_blind():
    scores, formulas = build_protocol_v0_2_primary_scores(_blind_primary_base())

    assert set(scores["score_name"]) == {PRIMARY_SCORE_NAME}
    assert "true_error_rmse" not in scores.columns
    assert "true_effect_key" not in scores.columns
    assert len(scores) == 4
    assert np.isfinite(scores.loc[scores["split"].eq("test"), "score_value"]).all()
    assert formulas["protocol_formula"].iloc[0] == "context_similarity + log_support - model_disagreement"


def test_evaluation_protocol_scores_keep_true_error_labels():
    base = _blind_primary_base()
    base["true_error_rmse"] = [1.0, 0.8, 0.6, 0.4]
    base["historical_residual_risk"] = [0.7, 0.6, 0.5, 0.4]
    base["ood_nearest_distance"] = [0.1, 0.2, 0.3, 0.4]
    base["prediction_magnitude_deviation"] = [0.4, 0.3, 0.2, 0.1]
    base["perturbation_effect_stability"] = [0.1, 0.2, 0.3, 0.4]

    blind_scores, _ = build_protocol_v0_2_primary_scores(base)
    scores, _ = build_protocol_v0_2_scores(base)
    eval_primary = scores[scores["score_name"].eq(PRIMARY_SCORE_NAME)]
    paired = eval_primary.merge(
        blind_scores[["record_id", "score_value"]],
        on="record_id",
        suffixes=("_evaluation", "_blind"),
    )

    assert "true_error_rmse" in scores.columns
    assert PRIMARY_SCORE_NAME in set(scores["score_name"])
    assert np.allclose(paired["score_value_evaluation"], paired["score_value_blind"], equal_nan=True)


def test_evaluation_protocol_scores_require_true_error_by_default():
    try:
        build_protocol_v0_2_scores(_blind_primary_base())
    except ValueError as exc:
        assert "true_error_rmse" in str(exc)
    else:
        raise AssertionError("evaluation scoring should require true_error_rmse by default")
