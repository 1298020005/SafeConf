import pandas as pd

from safetrans_confidence.features.schema import build_feature_provenance_table


def test_feature_provenance_marks_frozen_primary_as_blind_and_allowed():
    table = pd.DataFrame(
        {
            "context_similarity_max": [0.8],
            "perturbation_support_count": [3],
            "model_disagreement_rmse": [0.2],
        }
    )
    provenance = build_feature_provenance_table(table)
    primary = provenance[provenance["allowed_for_frozen_primary_score"]].copy()

    assert set(primary["feature_name"]) == {
        "context_similarity_max",
        "perturbation_support_count",
        "model_disagreement_rmse",
    }
    assert primary["allowed_for_blind_primary_features"].all()
    assert not primary["label_derived"].any()
    assert not primary["uses_heldout_true_effects"].any()
    assert primary["leakage_status"].eq("pass_frozen_primary").all()


def test_feature_provenance_keeps_label_derived_diagnostics_out_of_primary_score():
    provenance = build_feature_provenance_table()
    label_derived = provenance[provenance["label_derived"]].copy()

    assert {
        "historical_residual_risk",
        "perturbation_effect_stability",
        "prediction_magnitude_deviation",
        "fold_train_median_effect_norm",
    }.issubset(set(label_derived["feature_name"]))
    assert not label_derived["allowed_for_frozen_primary_score"].any()
    assert label_derived["allowed_for_evaluation_diagnostics"].all()
    assert label_derived["leakage_status"].eq("evaluation_only_label_derived").all()


def test_feature_provenance_present_column_tracks_input_table():
    table = pd.DataFrame({"dataset_name": ["d1", "d1", "d2"], "prediction_l2_norm": [1.0, None, 2.0]})
    provenance = build_feature_provenance_table(table)
    indexed = provenance.set_index("feature_name")
    present = indexed["present"].to_dict()

    assert bool(present["prediction_l2_norm"]) is True
    assert bool(present["model_disagreement_rmse"]) is False
    assert int(indexed.loc["prediction_l2_norm", "n_rows"]) == 3
    assert int(indexed.loc["prediction_l2_norm", "n_missing"]) == 1
    assert int(indexed.loc["prediction_l2_norm", "n_datasets_present"]) == 2
    assert int(indexed.loc["prediction_l2_norm", "n_datasets_with_missing"]) == 1
