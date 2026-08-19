from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safetrans_confidence.cli.run_tahoe_d1_postprocess import (
    _merge_summary_and_ci,
    _primary_gate,
    bootstrap_task_clusters,
    compute_summary,
    run,
)
from safetrans_confidence.cli.run_tahoe_pseudobulk_smoke import (
    assign_context_folds,
    build_context_holdout_split,
    select_shards,
)
from safetrans_confidence.cli.run_tahoe_triage_audit import (
    bootstrap_top10,
    point_summary,
    top_fraction_enrichment,
)


def _synthetic_scores(n_tasks: int = 16) -> pd.DataFrame:
    rows: list[dict] = []
    for task_index in range(n_tasks):
        split = "test" if task_index < n_tasks - 4 else "train"
        magnitude = float((task_index % 5) + 1)
        base_risk = float(task_index) + 0.2 * magnitude
        error = float(task_index) + 0.1 * ((task_index % 3) - 1)
        for predictor_index, predictor in enumerate(("V0ExactDoseMean", "V0DrugMeanAcrossDose")):
            risk = base_risk + 0.05 * predictor_index
            rows.append(
                {
                    "dataset_name": "Tahoe100M_pseudobulk_D1",
                    "fold_id": task_index % 5,
                    "split": split,
                    "context": f"cell_{task_index % 4}",
                    "perturbation": f"drug_{task_index:03d}@1uM",
                    "drug": f"drug_{task_index:03d}",
                    "task_key": f"cell_{task_index % 4}||drug_{task_index:03d}@1uM",
                    "predictor_name": predictor,
                    "score_name": "tahoe_protocol_v0_2_confidence",
                    "score_type": "confidence",
                    "score_value": -risk,
                    "true_error_rmse": error + 0.02 * predictor_index,
                    "true_effect_l2_norm": magnitude,
                }
            )
    return pd.DataFrame(rows)


def test_summary_uses_test_rows_only() -> None:
    scores = _synthetic_scores()
    observed = compute_summary(scores)

    mutated = scores.copy()
    train = mutated["split"].eq("train")
    mutated.loc[train, "score_value"] = -9999.0
    mutated.loc[train, "true_error_rmse"] = 9999.0
    rescored = compute_summary(mutated)

    pd.testing.assert_frame_equal(observed, rescored)
    overall = observed[observed["level"].eq("overall")].iloc[0]
    assert overall["n_records"] == 24
    assert overall["n_task_clusters"] == 12


def test_task_cluster_bootstrap_produces_ci_columns() -> None:
    scores = _synthetic_scores()
    summary = compute_summary(scores)
    ci, draws = bootstrap_task_clusters(scores, n_bootstrap=120, seed=5201)
    merged = _merge_summary_and_ci(summary, ci)

    overall = merged[merged["level"].eq("overall")].iloc[0]
    assert len(draws[draws["level"].eq("overall")]) == 120
    assert np.isfinite(overall["partial_rho_control_magnitude_ci_low"])
    assert np.isfinite(overall["partial_rho_control_magnitude_ci_high"])
    assert overall["partial_rho_control_magnitude_ci_low"] <= overall["partial_rho_control_magnitude_ci_high"]


def test_gate_uses_partial_rho_point_and_task_cluster_ci() -> None:
    assert _primary_gate(
        pd.DataFrame(
            [
                {
                    "level": "overall",
                    "group_id": "overall",
                    "partial_rho_control_magnitude": 0.25,
                    "partial_rho_control_magnitude_ci_low": 0.01,
                }
            ]
        )
    ) == "PASS"
    assert _primary_gate(
        pd.DataFrame(
            [
                {
                    "level": "overall",
                    "group_id": "overall",
                    "partial_rho_control_magnitude": 0.25,
                    "partial_rho_control_magnitude_ci_low": -0.01,
                }
            ]
        )
    ) == "PARTIAL"
    assert _primary_gate(
        pd.DataFrame(
            [
                {
                    "level": "overall",
                    "group_id": "overall",
                    "partial_rho_control_magnitude": -0.02,
                    "partial_rho_control_magnitude_ci_low": -0.20,
                }
            ]
        )
    ) == "FAIL"


def test_run_writes_postprocess_outputs(tmp_path) -> None:
    run_dir = tmp_path / "tahoe_d1"
    tables = run_dir / "tables"
    tables.mkdir(parents=True)
    _synthetic_scores().to_csv(tables / "TAHOE_PROTOCOL_SCORES_SMOKE.csv", index=False)
    overlap = tmp_path / "D1_TAHOE_MAIN7_OVERLAP_AUDIT.csv"
    overlap.write_text("field,value\nG8,PASS_WITH_CAVEAT\n", encoding="utf-8")

    status = run(
        run_dir=run_dir,
        scores_csv=None,
        out_dir=None,
        n_bootstrap=80,
        seed=5201,
        overlap_audits=[overlap],
    )

    assert status["status"] == "ok"
    assert status["bootstrap_unit"] == "task_key"
    assert status["n_overlap_audits_copied"] == 1
    assert (tables / "TAHOE_D1_FORMAL_SUMMARY.csv").exists()
    assert (tables / "TAHOE_D1_TASK_CLUSTER_BOOTSTRAP_CI.csv").exists()
    assert (tables / "TAHOE_D1_TASK_CLUSTER_BOOTSTRAP_DRAWS.csv").exists()
    assert (tables / "TAHOE_D1_OVERLAP_AUDIT_MANIFEST.csv").exists()
    assert (tables / overlap.name).exists()
    assert (run_dir / "reports" / "TAHOE_D1_POSTPROCESS_REPORT.md").exists()
    loaded = json.loads((run_dir / "TAHOE_D1_POSTPROCESS_STATUS.json").read_text(encoding="utf-8"))
    assert loaded["n_task_clusters"] == 12


def test_run_backfills_task_key_from_prediction_records(tmp_path) -> None:
    run_dir = tmp_path / "tahoe_d1"
    tables = run_dir / "tables"
    tables.mkdir(parents=True)
    scores = _synthetic_scores().copy()
    scores["record_id"] = np.arange(len(scores))
    records = scores[["record_id", "task_key"]].copy()
    scores.drop(columns=["task_key"]).to_csv(tables / "TAHOE_PROTOCOL_SCORES_SMOKE.csv", index=False)
    records.to_csv(tables / "TAHOE_PREDICTION_RECORDS_SMOKE.csv", index=False)

    status = run(
        run_dir=run_dir,
        scores_csv=None,
        out_dir=None,
        n_bootstrap=30,
        seed=5201,
        overlap_audits=[],
    )

    assert status["status"] == "ok"
    assert status["n_task_clusters"] == 12
    summary = pd.read_csv(tables / "TAHOE_D1_FORMAL_SUMMARY.csv")
    assert int(summary.loc[summary["level"].eq("overall"), "n_task_clusters"].iloc[0]) == 12


def test_missing_required_columns_raise_clear_error() -> None:
    scores = _synthetic_scores().drop(columns=["task_key"])
    with pytest.raises(ValueError, match="missing required columns"):
        compute_summary(scores)


def test_nested_shard_sampling_preserves_base_sample() -> None:
    shards = [Path(f"shard-{index:04d}.parquet") for index in range(1026)]
    base = select_shards(shards, max_shards=100)
    expanded = select_shards(shards, max_shards=300, base_shards=100)

    assert len(base) == 100
    assert len(expanded) == 300
    assert set(base).issubset(set(expanded))


def _synthetic_tahoe_tasks() -> pd.DataFrame:
    rows: list[dict] = []
    task_id = 0
    for context_index in range(10):
        for perturbation_index in range(12):
            rows.append(
                {
                    "task_id": task_id,
                    "task_key": f"cell_{context_index}||drug_{perturbation_index}@1uM",
                    "context": f"cell_{context_index}",
                    "perturbation": f"drug_{perturbation_index}@1uM",
                    "drug": f"drug_{perturbation_index}",
                    "concentration": 1.0,
                    "concentration_unit": "uM",
                    "plate": f"plate_{context_index % 3}",
                }
            )
            task_id += 1
    return pd.DataFrame(rows)


def test_context_holdout_split_holds_out_whole_contexts() -> None:
    tasks = _synthetic_tahoe_tasks()
    assignment = assign_context_folds(tasks, n_folds=5)
    split = build_context_holdout_split(tasks, n_folds=5)
    test = split[split["split"].eq("test")]

    assert len(assignment) == 10
    assert test["task_key"].nunique() == len(tasks)
    assert not test["pair_seen_in_train"].any()
    assert not test["context_seen_in_train"].any()
    assert test["perturbation_seen_in_train"].all()
    assert test.groupby("context")["fold_id"].nunique().eq(1).all()

    fold_tasks = test.groupby("fold_id")["task_key"].nunique()
    assert int(fold_tasks.max() - fold_tasks.min()) <= 12


def test_top_fraction_enrichment_recovers_perfect_ranking() -> None:
    errors = np.arange(100, dtype=float)
    result = top_fraction_enrichment(errors, errors.copy(), 0.10)

    assert result["precision"] == 1.0
    assert result["random_expected"] == 0.10
    assert result["enrichment"] == 10.0


def test_tahoe_triage_bootstrap_uses_task_clusters() -> None:
    rows: list[dict] = []
    for task_index in range(40):
        for predictor in ("V0ExactDoseMean", "V0DrugMeanAcrossDose"):
            rows.append(
                {
                    "task_key": f"task_{task_index}",
                    "predictor_name": predictor,
                    "true_error_rmse": float(task_index),
                    "safeconf_full": float(task_index),
                    "predicted_magnitude": float(39 - task_index),
                    "support_only": float(task_index),
                    "disagreement_only": float(task_index),
                    "random": float((task_index * 17) % 40),
                    "oracle_error": float(task_index),
                }
            )
    scores = pd.DataFrame(rows)
    point = point_summary(scores)
    ci, draws = bootstrap_top10(scores, n_bootstrap=40, seed=5201)

    full = point[
        point["level"].eq("overall")
        & point["score_name"].eq("safeconf_full")
        & np.isclose(point["top_fraction"], 0.10)
    ].iloc[0]
    assert full["enrichment"] == 10.0
    assert int(ci.iloc[0]["n_task_clusters"]) == 40
    assert len(draws) == 40
    assert float(ci.iloc[0]["safeconf_minus_magnitude_ci_low"]) > 0
