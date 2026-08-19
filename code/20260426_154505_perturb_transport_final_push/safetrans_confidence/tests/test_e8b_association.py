from __future__ import annotations

import numpy as np
import pandas as pd

from safetrans_confidence.cli.run_e8b_association import (
    _partial_spearman,
    _primary_gate,
    _prepare_sciplex3_errors,
    _per_method_association,
    aggregate_benchmark_seeds,
    build_frangieh_dry_audit,
    compute_safeconf_risk_by_context,
    compute_safeconf_risk_per_perturbation,
    run_association,
    run_permutation_null,
)


def _synthetic_feature_matrix(n_perturbations: int = 8) -> pd.DataFrame:
    rows = []
    for fold in range(3):
        for index in range(n_perturbations):
            split = ("train", "val", "test")[(index + fold) % 3]
            for predictor in ("V0StrongBaseline", "ContextSimBaseline"):
                rows.append(
                    {
                        "record_id": f"r::{fold}::{index}::{predictor}",
                        "dataset_name": "Frangieh",
                        "dataset_family": "gene_main",
                        "fold_id": fold,
                        "split": split,
                        "context": f"c{index % 2}",
                        "perturbation": f"g{index:03d}",
                        "predictor_name": predictor,
                        "context_similarity_max": 0.2 + 0.05 * index,
                        "perturbation_support_count": index + 1,
                        "model_disagreement_rmse": 1.0 - 0.05 * index,
                    }
                )
    return pd.DataFrame(rows)


def _synthetic_errors(n_perturbations: int = 8) -> pd.DataFrame:
    rows = []
    for method_index, method in enumerate(("m1", "m2", "m3")):
        for index in range(n_perturbations):
            rows.append(
                {
                    "method": method,
                    "perturbation": f"g{index:03d}",
                    "error": index + 0.01 * method_index,
                }
            )
    return pd.DataFrame(rows)


def test_score_aggregation_excludes_test_rows() -> None:
    base = _synthetic_feature_matrix()
    observed = compute_safeconf_risk_per_perturbation(base, "Frangieh", "gene_main")
    mutated = base.copy()
    test = mutated["split"].eq("test")
    mutated.loc[test, "context_similarity_max"] = 1e9
    mutated.loc[test, "perturbation_support_count"] = 10**9
    mutated.loc[test, "model_disagreement_rmse"] = -1e9
    rescored = compute_safeconf_risk_per_perturbation(mutated, "Frangieh", "gene_main")
    pd.testing.assert_series_equal(observed, rescored)


def test_risk_direction_positive_with_error() -> None:
    risk = pd.Series(
        np.arange(8, dtype=float),
        index=[f"g{i:03d}" for i in range(8)],
        name="risk",
    )
    result = _per_method_association(risk, _synthetic_errors())
    assert result["spearman_rho"].gt(0.99).all()


def test_bootstrap_ci_contains_observed() -> None:
    risk = pd.Series(
        np.arange(8, dtype=float),
        index=[f"g{i:03d}" for i in range(8)],
        name="risk",
    )
    result = run_association(risk, _synthetic_errors(), n_bootstrap=200, seed=5201)
    assert result["median_spearman_ci_low"] <= result["median_spearman"]
    assert result["median_spearman"] <= result["median_spearman_ci_high"]


def test_permutation_null_centered_near_zero() -> None:
    risk = pd.Series(
        np.arange(30, dtype=float),
        index=[f"g{i:03d}" for i in range(30)],
        name="risk",
    )
    null = run_permutation_null(risk, _synthetic_errors(30), n_perm=500, seed=5201)
    assert abs(float(np.nanmedian(null))) < 0.12


def test_frangieh_74_perturbations_all_join() -> None:
    features = _synthetic_feature_matrix(74)
    benchmark = pd.DataFrame(
        {
            "method": "m1",
            "perturbation": [f"g{i:03d}" for i in range(74)],
            "error": np.linspace(0.1, 1.0, 74),
            "seed": 1,
            "Nstimulated": np.arange(10, 84),
        }
    )
    audit = build_frangieh_dry_audit(features, benchmark)
    assert audit["joined_perturbations"] == 74
    assert audit["join_complete"]
    assert audit["score_coverage_complete"]
    assert audit["association_computed"] is False


def test_available_seed_aggregation_uses_median() -> None:
    frame = pd.DataFrame(
        {
            "method": ["m1", "m1", "m1"],
            "perturbation": ["g1", "g1", "g1"],
            "error": [1.0, 9.0, 3.0],
            "seed": [1, 2, 3],
            "Nstimulated": [20, 20, 20],
        }
    )
    result = aggregate_benchmark_seeds(frame)
    assert result.loc[0, "error"] == 3.0
    assert result.loc[0, "n_available_seeds"] == 3


def test_context_risk_is_preserved_before_final_aggregation() -> None:
    result = compute_safeconf_risk_by_context(
        _synthetic_feature_matrix(),
        "Frangieh",
        "gene_main",
    )
    assert set(result.columns) == {"context", "perturbation", "risk"}
    assert result.groupby("perturbation")["context"].nunique().max() == 1


def test_primary_gate_uses_ci_and_positive_method_fraction() -> None:
    assert _primary_gate(0.2, 0.01, 0.60) == "PASS"
    assert _primary_gate(0.2, -0.01, 0.80) == "PARTIAL"
    assert _primary_gate(0.2, 0.01, 0.50) == "PARTIAL"
    assert _primary_gate(0.0, -0.1, 0.80) == "FAIL"


def test_partial_spearman_removes_shared_nuisance_rank() -> None:
    nuisance = np.repeat(np.arange(10, dtype=float), 2)
    risk = 10 * nuisance + np.tile([-1.0, 1.0], 10)
    error = 10 * nuisance + np.tile([1.0, -1.0], 10)
    raw_rho = pd.Series(risk).corr(pd.Series(error), method="spearman")
    partial = _partial_spearman(risk, error, nuisance)
    assert raw_rho > 0.95
    assert partial < 0


def test_sciplex3_error_preparation_uses_only_exact_alias_and_aggregates_doses(
    tmp_path,
) -> None:
    raw = pd.DataFrame(
        {
            "Unnamed: 0": range(8),
            "performance": [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0],
            "metric": ["mse"] * 8,
            "DataSet": ["sciplex3_A549"] * 8,
            "method": ["m1"] * 8,
            "perturb": [
                "A549_DrugA_0.1",
                "A549_DrugA_0.1",
                "A549_DrugA_1.0",
                "A549_DrugA_1.0",
                "A549_DrugB_0.1",
                "A549_DrugB_0.1",
                "A549_DrugB_1.0",
                "A549_DrugB_1.0",
            ],
            "DEG": [5000] * 8,
            "Ncontrol": [10] * 8,
            "Nimputed": [10] * 8,
            "Nstimulated": [10] * 8,
            "seed": [1, 2] * 4,
        }
    )
    csv_path = tmp_path / "chemical.csv"
    raw.to_csv(csv_path, index=False)
    alias = pd.DataFrame(
        {
            "safeconf_perturbation": ["SafeA", "UnsafeB"],
            "scperturbench_drug": ["DrugA", "DrugB"],
            "match_type": ["exact", "manual"],
            "notes": ["", ""],
        }
    )
    result = _prepare_sciplex3_errors(csv_path, alias)
    assert result["perturbation"].tolist() == ["SafeA"]
    assert result.loc[0, "error"] == 4.0
    assert result.loc[0, "n_doses"] == 2
