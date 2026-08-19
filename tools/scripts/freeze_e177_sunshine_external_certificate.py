#!/usr/bin/env python3
"""Freeze E177 external processed-data identities before any target truth run."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
    "SunshineHein2023.h5ad"
)
SCGPT_VOCAB = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/"
    "codex_scgpt_attnres_workspace/checkpoints/whole-human/vocab.json"
)
SCGPT_CHECKPOINT = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/"
    "codex_scgpt_attnres_workspace/checkpoints/whole-human"
)
GEARS_GO = Path("/home/yyf/data/gears_formal_baselines_v2/frangieh_local_atlas/go.csv")
OUT = ROOT / "docs/实验结果/E177_sunshine_external_certificate_20260719"
MANIFESTS = OUT / "manifests"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

EXPERIMENT = "E177_sunshine_external_certificate"
TECH_GROUPS = tuple(str(value) for value in range(1, 9))
TARGET_SALT = "E177_SUNSHINE_EXTERNAL_TARGET_IDENTITY_V1"
SPLIT_SALT = "E177_SUNSHINE_EXTERNAL_SPLIT_V1"
N_SELECTED_TARGETS = 144
SPLIT_COUNTS = {
    "train": 54,
    "validation": 10,
    "calibration": 30,
    "evaluation": 50,
}
MODEL_SEEDS = (3407, 3408, 3409, 3410, 3411)
GENE_PANEL_SIZE = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(salt: str, *values: object) -> str:
    payload = b"\0".join([salt.encode(), *[str(value).encode() for value in values]])
    return hashlib.sha256(payload).hexdigest()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_metadata() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    adata = ad.read_h5ad(SOURCE, backed="r")
    try:
        obs = adata.obs.copy()
        for column in obs.columns:
            obs[column] = obs[column].astype(str)
        obs.insert(0, "source_row_index", np.arange(adata.n_obs, dtype=int))
        var_names = [str(value) for value in adata.var_names]
        schema = {
            "source_path": str(SOURCE),
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "obs_columns": [str(value) for value in adata.obs.columns],
            "var_name_sample": var_names[:20],
            "backed_mode": True,
            "X_read_during_freeze": False,
            "layers_read_during_freeze": False,
        }
    finally:
        adata.file.close()
    return obs, var_names, schema


def load_vocab() -> set[str]:
    payload = json.loads(SCGPT_VOCAB.read_text())
    return set(str(key) for key in payload)


def build_target_audit(obs: pd.DataFrame, var_names: list[str], vocab: set[str]) -> pd.DataFrame:
    exact = obs.loc[obs.match_type.eq("exact_match")].copy()
    exact["is_control"] = exact.perturbation.eq("control")
    single = exact.loc[
        ~exact.perturbation.isin(["control", "nan", ""])
        & ~exact.perturbation.str.contains("_", regex=False)
    ].copy()

    counts = single.groupby("perturbation", observed=True).size().rename("n_exact_cells")
    group_counts = (
        single.groupby(["perturbation", "gem_group"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for group in TECH_GROUPS:
        if group not in group_counts.columns:
            group_counts[group] = 0
    group_counts = group_counts[list(TECH_GROUPS)].astype(int)

    audit = counts.reset_index().merge(
        group_counts.reset_index(), on="perturbation", how="left"
    )
    for group in TECH_GROUPS:
        audit[group] = audit[group].fillna(0).astype(int)
    audit["in_expression_var"] = audit.perturbation.isin(set(var_names))
    audit["in_scgpt_vocab"] = audit.perturbation.isin(vocab)
    audit["is_single_gene_label"] = True
    audit["min_exact_cells_per_technical_group"] = audit[list(TECH_GROUPS)].min(axis=1)
    audit["n_technical_groups_with_cells"] = (audit[list(TECH_GROUPS)] > 0).sum(axis=1)
    audit["eligible_e177_target"] = (
        audit.in_expression_var
        & audit.in_scgpt_vocab
        & audit.n_exact_cells.ge(20)
        & audit.min_exact_cells_per_technical_group.ge(3)
    )
    audit["e177_selection_sha256"] = audit.perturbation.map(
        lambda value: stable_hash(TARGET_SALT, value)
    )
    return audit.sort_values(["eligible_e177_target", "perturbation"], ascending=[False, True])


def split_targets(selected: pd.DataFrame) -> pd.DataFrame:
    work = selected.copy().sort_values("n_exact_cells", kind="stable").reset_index(drop=True)
    work["cell_count_quartile"] = pd.qcut(
        work.n_exact_cells.rank(method="first"), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"]
    ).astype(str)
    per_quartile = {
        "Q1_low": {"train": 14, "validation": 3, "calibration": 7, "evaluation": 12},
        "Q2": {"train": 13, "validation": 2, "calibration": 8, "evaluation": 13},
        "Q3": {"train": 13, "validation": 3, "calibration": 7, "evaluation": 13},
        "Q4_high": {"train": 14, "validation": 2, "calibration": 8, "evaluation": 12},
    }
    blocks: list[pd.DataFrame] = []
    for quartile, block in work.groupby("cell_count_quartile", sort=True):
        counts = per_quartile[str(quartile)]
        ordered = block.copy()
        ordered["e177_split_sha256"] = ordered.perturbation.map(
            lambda value: stable_hash(SPLIT_SALT, quartile, value)
        )
        ordered = ordered.sort_values(["e177_split_sha256", "perturbation"], kind="stable")
        split = (
            ["train"] * counts["train"]
            + ["validation"] * counts["validation"]
            + ["calibration"] * counts["calibration"]
            + ["evaluation"] * counts["evaluation"]
        )
        if len(split) != len(ordered):
            raise RuntimeError(f"E177 split count failed for {quartile}")
        ordered["target_split"] = split
        blocks.append(ordered)
    result = pd.concat(blocks, ignore_index=True)
    result = result.sort_values(["target_split", "perturbation"], kind="stable").reset_index(drop=True)
    result.insert(0, "e177_target_rank", np.arange(1, len(result) + 1))
    observed = result.target_split.value_counts().to_dict()
    if observed != SPLIT_COUNTS:
        raise RuntimeError(f"E177 split counts changed: {observed}")
    return result


def select_targets(audit: pd.DataFrame) -> pd.DataFrame:
    pool = audit.loc[audit.eligible_e177_target].copy()
    if len(pool) < N_SELECTED_TARGETS:
        raise RuntimeError(f"E177 eligible pool too small: {len(pool)}")
    selected = pool.sort_values(
        ["e177_selection_sha256", "perturbation"], kind="stable"
    ).head(N_SELECTED_TARGETS)
    selected = split_targets(selected)
    selected["selection_used_expression_matrix_or_truth_error"] = False
    selected["technical_group_label_is_biological_context"] = False
    return selected


def build_tasks(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, target in selected.iterrows():
        for group in TECH_GROUPS:
            split = str(target["target_split"])
            perturbation = str(target["perturbation"])
            if split == "train":
                phase = "PRETRUTH_TRAIN_X"
            elif split == "validation":
                phase = "PRETRUTH_VALIDATION_X"
            elif split == "calibration":
                phase = "POSTGATE_CALIBRATION_TRUTH_X"
            elif split == "evaluation":
                phase = "POSTCALIBRATION_EVALUATION_TRUTH_X"
            else:
                raise RuntimeError(f"unknown split: {split}")
            rows.append(
                {
                    "task_id": f"E177::G{group}::{perturbation}",
                    "dataset_id": EXPERIMENT,
                    "technical_group": group,
                    "perturbation": perturbation,
                    "target_split": split,
                    "truth_access_phase": phase,
                    "prediction_query_required": split in {
                        "validation", "calibration", "evaluation"
                    },
                    "supervised_training_truth_available_pretruth": split in {"train", "validation"},
                    "primary_calibration_task": split == "calibration",
                    "primary_evaluation_task": split == "evaluation",
                    "n_exact_cells_for_task": int(target[group]),
                    "technical_group_label_is_biological_context": False,
                }
            )
    tasks = pd.DataFrame(rows)
    expected = {
        "train": SPLIT_COUNTS["train"] * len(TECH_GROUPS),
        "validation": SPLIT_COUNTS["validation"] * len(TECH_GROUPS),
        "calibration": SPLIT_COUNTS["calibration"] * len(TECH_GROUPS),
        "evaluation": SPLIT_COUNTS["evaluation"] * len(TECH_GROUPS),
    }
    if tasks.target_split.value_counts().to_dict() != expected:
        raise RuntimeError("E177 task split counts changed")
    if (tasks.n_exact_cells_for_task < 3).any():
        raise RuntimeError("E177 selected target has a sparse technical-group task")
    return tasks


def build_row_access(obs: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    exact = obs.loc[obs.match_type.eq("exact_match")].copy()
    controls = exact.loc[exact.perturbation.eq("control")].copy()
    controls["target_split"] = "control"
    controls["truth_access_phase"] = "PRETRUTH_CONTROL_X"
    controls["purpose"] = "build_same_group_control_profile"
    control_rows = controls[
        ["source_row_index", "gem_group", "perturbation", "target_split", "truth_access_phase", "purpose"]
    ].rename(columns={"gem_group": "technical_group"})

    split_map = selected.set_index("perturbation").target_split.to_dict()
    target_rows = exact.loc[exact.perturbation.isin(split_map)].copy()
    target_rows["target_split"] = target_rows.perturbation.map(split_map)
    phase_map = {
        "train": "PRETRUTH_TRAIN_X",
        "validation": "PRETRUTH_VALIDATION_X",
        "calibration": "POSTGATE_CALIBRATION_TRUTH_X",
        "evaluation": "POSTCALIBRATION_EVALUATION_TRUTH_X",
    }
    purpose_map = {
        "train": "fit_reference_predictors",
        "validation": "pretruth_gate_and_early_stopping",
        "calibration": "postgate_split_conformal_calibration",
        "evaluation": "postcalibration_hidden_evaluation",
    }
    target_rows["truth_access_phase"] = target_rows.target_split.map(phase_map)
    target_rows["purpose"] = target_rows.target_split.map(purpose_map)
    target_rows = target_rows[
        ["source_row_index", "gem_group", "perturbation", "target_split", "truth_access_phase", "purpose"]
    ].rename(columns={"gem_group": "technical_group"})
    result = pd.concat([control_rows, target_rows], ignore_index=True)
    result.insert(0, "e177_access_row_id", np.arange(1, len(result) + 1))
    result["logical_x_row_read_count_when_materialized"] = 1
    result["source_is_public_processed_h5ad"] = True
    return result.sort_values(
        ["truth_access_phase", "technical_group", "perturbation", "source_row_index"],
        kind="stable",
    ).reset_index(drop=True)


def write_docs(
    selected: pd.DataFrame,
    audit: pd.DataFrame,
    tasks: pd.DataFrame,
    row_access: pd.DataFrame,
    schema: dict[str, Any],
    source_sha: str,
) -> None:
    plan = f"""# E177 external certificate preregistration

## Purpose

E176 confirmed the risk certificate on a same-study multi-donor Primary CD4 setting. E177 moves to an independent public processed single-cell perturbation dataset. The goal is narrow: test whether the certificate-style conclusion still behaves sensibly outside the Primary CD4 line, without claiming wet-lab validation or deployment readiness.

## Frozen data boundary

The source is used only as a processed expression matrix. The metadata freeze reads `obs` and `var_names` in backed mode and does not decode `X` or layers. The selected population is exact single-gene CRISPRi labels with at least 20 exact-match cells overall, present in the expression matrix and scGPT vocabulary, and at least 3 cells in each of the eight `gem_group` technical groups.

`gem_group` is treated only as a technical repeat label. It is not a donor, patient, biological context, or independent study by itself.

## Frozen split

Selected targets: {len(selected)}. Split by target identity before expression access:

- train targets: {SPLIT_COUNTS['train']} ({SPLIT_COUNTS['train'] * len(TECH_GROUPS)} tasks)
- validation targets: {SPLIT_COUNTS['validation']} ({SPLIT_COUNTS['validation'] * len(TECH_GROUPS)} tasks)
- calibration targets: {SPLIT_COUNTS['calibration']} ({SPLIT_COUNTS['calibration'] * len(TECH_GROUPS)} tasks)
- final evaluation targets: {SPLIT_COUNTS['evaluation']} ({SPLIT_COUNTS['evaluation'] * len(TECH_GROUPS)} tasks)

The split is balanced across target cell-count quartiles and sorted by salted SHA-256 identities. Calibration and evaluation target expression is not available to model training or the pretruth gate.

## Model and gate

Use scGPT and GEARS with seeds {', '.join(map(str, MODEL_SEEDS))}. Before calibration truth is opened, the run must pass a truth-blind stability gate on train/validation/query predictions. The deployed object is the five-seed family mean for each predictor.

The primary certificate keeps the E176 conclusion style:

- deterministic lower bound: RMSE(p1, p2) / 2 for the two-predictor mean and max errors
- split conformal upper bound: target-level clusters, one cluster contains all eight technical groups of the same perturbation
- target coverage: 90 percent

Ranking metrics are diagnostics only. A failure to beat predicted magnitude is reported directly and does not get repaired after evaluation truth is visible.
"""
    (OUT / "PREREG_ANALYSIS_PLAN.md").write_text(plan, encoding="utf-8")

    readme = """# E177 先看这个

E177 是当前主线的独立外部数据计算验证。它不是新的湿实验，也不包含实验操作细节。

最重要的边界有三条：

1. 只用公开处理后的单细胞扰动表达数据做计算验证。
2. `gem_group` 只作为技术重复，不写成供体或生物学背景。
3. 先提交元数据冻结，再跑模型；校准和最终评价真值分阶段开放。
"""
    (OUT / "README_先看这个.md").write_text(readme, encoding="utf-8")

    report = f"""# E177 metadata freeze report

- Source shape: {schema['n_obs']:,} cells x {schema['n_vars']:,} genes.
- Exact-match cells: {int((row_access.truth_access_phase != 'IGNORED').sum()):,} planned rows include controls plus selected targets.
- Exact controls: {int(row_access.perturbation.eq('control').sum()):,}.
- Single-gene eligible targets before final hash selection: {int(audit.eligible_e177_target.sum()):,}.
- Frozen targets: {len(selected):,}; frozen tasks: {len(tasks):,}.
- Calibration targets/tasks: {SPLIT_COUNTS['calibration']:,}/{SPLIT_COUNTS['calibration'] * len(TECH_GROUPS):,}.
- Evaluation targets/tasks: {SPLIT_COUNTS['evaluation']:,}/{SPLIT_COUNTS['evaluation'] * len(TECH_GROUPS):,}.
- Expression matrix decoded during freeze: **0**.
- Source SHA-256: `{source_sha}`.
"""
    (REPORTS / "E177_METADATA_FREEZE_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"append-only E177 output exists: {OUT}")
    obs, var_names, schema = load_metadata()
    vocab = load_vocab()
    audit = build_target_audit(obs, var_names, vocab)
    selected = select_targets(audit)
    tasks = build_tasks(selected)
    row_access = build_row_access(obs, selected)

    expected_phases = {
        "PRETRUTH_CONTROL_X": int(obs.match_type.eq("exact_match").mul(obs.perturbation.eq("control")).sum()),
        "PRETRUTH_TRAIN_X": int(selected.loc[selected.target_split.eq("train"), list(TECH_GROUPS)].to_numpy().sum()),
        "PRETRUTH_VALIDATION_X": int(selected.loc[selected.target_split.eq("validation"), list(TECH_GROUPS)].to_numpy().sum()),
        "POSTGATE_CALIBRATION_TRUTH_X": int(selected.loc[selected.target_split.eq("calibration"), list(TECH_GROUPS)].to_numpy().sum()),
        "POSTCALIBRATION_EVALUATION_TRUTH_X": int(selected.loc[selected.target_split.eq("evaluation"), list(TECH_GROUPS)].to_numpy().sum()),
    }
    observed_phases = row_access.truth_access_phase.value_counts().to_dict()
    if observed_phases != expected_phases:
        raise RuntimeError(f"E177 row access counts changed: {observed_phases} != {expected_phases}")

    for directory in (OUT, MANIFESTS, TABLES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    audit.to_csv(TABLES / "E177_TARGET_AUDIT.csv", index=False)
    selected.to_csv(MANIFESTS / "E177_SELECTED_TARGETS.csv", index=False)
    tasks.to_csv(MANIFESTS / "E177_TASK_MANIFEST.csv", index=False)
    row_access.to_csv(MANIFESTS / "E177_ROW_ACCESS_MANIFEST.csv", index=False)

    source_sha = sha256_file(SOURCE)
    write_docs(selected, audit, tasks, row_access, schema, source_sha)

    source_lock = {
        "schema": "safeconf_e177_source_lock_v1",
        "experiment": EXPERIMENT,
        "source_path": str(SOURCE),
        "source_bytes": SOURCE.stat().st_size,
        "source_sha256": source_sha,
        "source_shape": [schema["n_obs"], schema["n_vars"]],
        "obs_columns": schema["obs_columns"],
        "selection_used_obs_columns": [
            "source_row_index",
            "match_type",
            "perturbation",
            "gem_group",
        ],
        "selection_used_var_names": True,
        "selection_used_expression_matrix_X": False,
        "selection_used_layers": False,
        "selection_used_target_errors_or_predictions": False,
        "public_processed_data_only": True,
        "operational_wetlab_protocol_in_scope": False,
    }
    (OUT / "SOURCE_LOCK.json").write_text(
        json.dumps(source_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    checkpoint_files = sorted(path for path in SCGPT_CHECKPOINT.iterdir() if path.is_file())
    model_lock = {
        "schema": "safeconf_e177_model_lock_v1",
        "experiment": EXPERIMENT,
        "gene_panel_size": GENE_PANEL_SIZE,
        "model_seeds": list(MODEL_SEEDS),
        "scgpt_checkpoint_files": {str(path): sha256_file(path) for path in checkpoint_files},
        "scgpt_vocab": {"path": str(SCGPT_VOCAB), "sha256": sha256_file(SCGPT_VOCAB)},
        "gears_go_prior": {"path": str(GEARS_GO), "sha256": sha256_file(GEARS_GO)},
        "selected_targets": N_SELECTED_TARGETS,
        "technical_groups": list(TECH_GROUPS),
        "technical_group_label_is_biological_context": False,
        "primary_lower_certificate": "rmse(p1,p2)/2",
        "primary_upper_certificate": "target_cluster_split_conformal",
        "deployed_seed_estimator": "five_seed_family_mean",
        "pretruth_gate_required_before_calibration": True,
    }
    (OUT / "MODEL_INPUT_LOCK.json").write_text(
        json.dumps(model_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    stat_lock = {
        "schema": "safeconf_e177_statistical_lock_v1",
        "experiment": EXPERIMENT,
        "target_identity_split_counts": SPLIT_COUNTS,
        "n_technical_groups_per_target": len(TECH_GROUPS),
        "primary_calibration_targets": SPLIT_COUNTS["calibration"],
        "primary_calibration_tasks": SPLIT_COUNTS["calibration"] * len(TECH_GROUPS),
        "primary_evaluation_targets": SPLIT_COUNTS["evaluation"],
        "primary_evaluation_tasks": SPLIT_COUNTS["evaluation"] * len(TECH_GROUPS),
        "target_cluster_definition": "one perturbation target across all eight technical groups",
        "target_coverage": 0.90,
        "finite_sample_order_rank_one_based": math.ceil((SPLIT_COUNTS["calibration"] + 1) * 0.90),
        "evaluation_truth_may_select_or_modify_method": False,
        "ranking_claim_is_diagnostic_only": True,
        "independent_public_processed_study": True,
        "technical_repeat_not_donor": True,
        "wetlab_or_clinical_validation": False,
    }
    (OUT / "STATISTICAL_ANALYSIS_LOCK.json").write_text(
        json.dumps(stat_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    access_log = {
        "schema": "safeconf_e177_freeze_access_log_v1",
        "experiment": EXPERIMENT,
        "stage": "F1_METADATA_FREEZE",
        "X_values_read": 0,
        "layers_values_read": 0,
        "obs_rows_read": int(len(obs)),
        "var_names_read": int(len(var_names)),
        "planned_row_access_by_phase": expected_phases,
    }
    (OUT / "HDF5_VALUE_ACCESS_LOG.json").write_text(
        json.dumps(access_log, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    artifacts = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "RUN_STATUS.json")
    status = {
        "schema": "safeconf_e177_metadata_freeze_v1",
        "experiment": EXPERIMENT,
        "stage": "F1_METADATA_FREEZE",
        "status": "PASS",
        "generated_at": now(),
        "python": platform.python_version(),
        "source_shape": [schema["n_obs"], schema["n_vars"]],
        "n_exact_match_cells": int(obs.match_type.eq("exact_match").sum()),
        "n_exact_control_cells": int(obs.match_type.eq("exact_match").mul(obs.perturbation.eq("control")).sum()),
        "n_single_gene_exact_labels": int(
            (
                obs.match_type.eq("exact_match")
                & ~obs.perturbation.isin(["control", "nan", ""])
                & ~obs.perturbation.str.contains("_", regex=False)
            ).groupby(obs.perturbation).any().sum()
        ),
        "n_eligible_targets_before_hash_selection": int(audit.eligible_e177_target.sum()),
        "n_selected_targets": int(len(selected)),
        "split_counts": SPLIT_COUNTS,
        "n_tasks": int(len(tasks)),
        "n_calibration_tasks": int(tasks.primary_calibration_task.sum()),
        "n_evaluation_tasks": int(tasks.primary_evaluation_task.sum()),
        "planned_row_access_by_phase": expected_phases,
        "expression_matrix_x_values_read_during_freeze": 0,
        "technical_group_label_is_biological_context": False,
        "public_processed_data_only": True,
        "artifact_sha256": {
            path.relative_to(OUT).as_posix(): sha256_file(path) for path in artifacts
        },
    }
    (OUT / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
